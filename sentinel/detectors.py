"""Online changepoint detectors.

All detectors consume a series that has already been standardised with
robust statistics (median and MAD) from a changepoint-free warmup window,
and all of them are strictly online: the score at time t only uses
observations up to and including t.

Two Bayesian online changepoint detectors (Adams and MacKay, 2007) are
implemented with conjugate observation models:

* ``bocpd_iid_score``: Normal-Gamma model, observations i.i.d. inside a
  regime. This is the textbook BOCPD and the one most open source
  implementations ship.
* ``bocpd_ar_score``: Normal-Inverse-Gamma Bayesian linear regression on
  ``[1, y_{t-1}]``, i.e. each regime is an AR(1) process with its own
  intercept, coefficient and noise variance.

The score returned by both is the posterior probability that the current
run length is short, P(r_t <= w | y_{1:t}). A regime change moves mass to
short run lengths, so a threshold on this probability is an alarm rule.
"""

from __future__ import annotations

import math

import numpy as np

_LOG_PI = math.log(math.pi)


def robust_standardise(y: np.ndarray, warmup: int) -> np.ndarray:
    ref = y[:warmup]
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med))) * 1.4826
    return (y - med) / max(mad, 1e-9)


def _lgamma_diff_table(a0: float, n_max: int) -> np.ndarray:
    """lgamma(a_n + 1/2) - lgamma(a_n) for a_n = a0 + n/2, n = 0..n_max."""
    out = np.empty(n_max + 1)
    for n in range(n_max + 1):
        a = a0 + 0.5 * n
        out[n] = math.lgamma(a + 0.5) - math.lgamma(a)
    return out


def _student_logpdf(x, loc, scale2, a, lg_diff):
    """Log density of a Student-t with df = 2a, location loc, squared scale scale2."""
    df = 2.0 * a
    z = (x - loc) ** 2 / (df * scale2)
    return lg_diff - 0.5 * (np.log(df * scale2) + _LOG_PI) - (a + 0.5) * np.log1p(z)


def _robust_logpred(x, loc, scale2, a, lg_diff, max_df):
    """Posterior predictive with its degrees of freedom capped at ``max_df``.

    The conjugate models assume Gaussian noise, so the exact predictive
    tightens towards a Gaussian as a regime accumulates data. Real telemetry
    has heavy tailed noise, and one large spike is then far more probable
    under a fresh regime than under a well estimated one, which reads as a
    changepoint. Capping the predictive degrees of freedom keeps every
    regime equally tolerant of spikes. The posterior itself is untouched;
    this is a tempered likelihood, not an exact Student-t noise model.
    """
    if max_df is None:
        return _student_logpdf(x, loc, scale2, a, lg_diff)
    a_cap = 0.5 * max_df
    capped = a > a_cap
    if not np.any(capped):
        return _student_logpdf(x, loc, scale2, a, lg_diff)
    a_eff = np.where(capped, a_cap, a)
    lg_cap = math.lgamma(a_cap + 0.5) - math.lgamma(a_cap)
    lg_eff = np.where(capped, lg_cap, lg_diff)
    return _student_logpdf(x, loc, scale2, a_eff, lg_eff)


def _logsumexp(v: np.ndarray) -> float:
    m = float(np.max(v))
    return m + math.log(float(np.sum(np.exp(v - m))))


def bocpd_iid_score(
    y: np.ndarray,
    hazard: float = 1.0 / 250.0,
    window: int = 60,
    r_max: int = 400,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    max_df: float | None = 5.0,
) -> np.ndarray:
    """BOCPD with a Normal-Gamma (i.i.d. Gaussian, unknown mean and variance) model."""
    T = len(y)
    lg = _lgamma_diff_table(alpha0, r_max + 1)
    log_h, log_1mh = math.log(hazard), math.log1p(-hazard)

    log_r = np.array([0.0])
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])
    score = np.zeros(T)

    for t in range(T):
        x = y[t]
        n = np.arange(len(log_r))
        scale2 = beta * (kappa + 1.0) / (alpha * kappa)
        log_pred = _robust_logpred(x, mu, scale2, alpha, lg[n], max_df)

        joint = log_r + log_pred
        growth = joint + log_1mh
        cp = _logsumexp(joint + log_h)
        log_r = np.concatenate(([cp], growth))
        log_r -= _logsumexp(log_r)

        new_kappa = kappa + 1.0
        new_mu = (kappa * mu + x) / new_kappa
        new_alpha = alpha + 0.5
        new_beta = beta + kappa * (x - mu) ** 2 / (2.0 * new_kappa)
        mu = np.concatenate(([mu0], new_mu))
        kappa = np.concatenate(([kappa0], new_kappa))
        alpha = np.concatenate(([alpha0], new_alpha))
        beta = np.concatenate(([beta0], new_beta))

        if len(log_r) > r_max:
            log_r, mu, kappa, alpha, beta = (a[:r_max] for a in (log_r, mu, kappa, alpha, beta))
            log_r -= _logsumexp(log_r)

        score[t] = float(np.sum(np.exp(log_r[: window + 1])))
    return score


def bocpd_ar_score(
    y: np.ndarray,
    hazard: float = 1.0 / 250.0,
    window: int = 60,
    r_max: int = 400,
    m0: tuple[float, float] = (0.0, 0.5),
    p0: tuple[float, float] = (1.0, 4.0),
    a0: float = 1.0,
    b0: float = 0.5,
    max_df: float | None = 5.0,
    return_map: bool = False,
):
    """BOCPD where every regime is an AR(1) process.

    Observation model inside a regime: y_t = c + phi * y_{t-1} + e_t with
    e_t ~ N(0, s2). Prior: (c, phi) | s2 ~ N(m0, s2 * diag(p0)^-1) and
    s2 ~ InvGamma(a0, b0). The sufficient statistics per run length are
    X'X (three numbers), X'y (two numbers), y'y and n, so the posterior and
    the Student-t predictive are closed form and fully vectorised over run
    lengths.
    """
    T = len(y)
    lg = _lgamma_diff_table(a0, r_max + 1)
    log_h, log_1mh = math.log(hazard), math.log1p(-hazard)
    m0v = np.array(m0, dtype=float)
    P0 = np.diag(np.array(p0, dtype=float))
    h0 = P0 @ m0v
    q0 = float(m0v @ P0 @ m0v)

    log_r = np.array([0.0])
    # sufficient statistics, one entry per run length
    s11 = np.zeros(1)  # n
    s12 = np.zeros(1)  # sum y_prev
    s22 = np.zeros(1)  # sum y_prev^2
    xy1 = np.zeros(1)  # sum y
    xy2 = np.zeros(1)  # sum y_prev * y
    yy = np.zeros(1)   # sum y^2
    score = np.zeros(T)
    r_map = np.zeros(T, dtype=int)
    y_prev = y[0]

    for t in range(T):
        x = y[t]
        n = np.arange(len(log_r))
        # posterior precision P = P0 + X'X
        p11 = P0[0, 0] + s11
        p12 = P0[0, 1] + s12
        p22 = P0[1, 1] + s22
        det = p11 * p22 - p12 * p12
        v11, v12, v22 = p22 / det, -p12 / det, p11 / det
        g1 = h0[0] + xy1
        g2 = h0[1] + xy2
        mn1 = v11 * g1 + v12 * g2
        mn2 = v12 * g1 + v22 * g2
        an = a0 + 0.5 * s11
        quad = mn1 * g1 + mn2 * g2  # mn' P mn
        bn = b0 + 0.5 * (yy + q0 - quad)
        bn = np.maximum(bn, 1e-9)

        loc = mn1 + mn2 * y_prev
        lev = v11 + 2.0 * v12 * y_prev + v22 * y_prev * y_prev
        scale2 = bn / an * (1.0 + lev)
        log_pred = _robust_logpred(x, loc, scale2, an, lg[n], max_df)

        joint = log_r + log_pred
        growth = joint + log_1mh
        cp = _logsumexp(joint + log_h)
        log_r = np.concatenate(([cp], growth))
        log_r -= _logsumexp(log_r)

        z = np.zeros(1)
        s11 = np.concatenate((z, s11 + 1.0))
        s12 = np.concatenate((z, s12 + y_prev))
        s22 = np.concatenate((z, s22 + y_prev * y_prev))
        xy1 = np.concatenate((z, xy1 + x))
        xy2 = np.concatenate((z, xy2 + y_prev * x))
        yy = np.concatenate((z, yy + x * x))

        if len(log_r) > r_max:
            log_r, s11, s12, s22, xy1, xy2, yy = (
                a[:r_max] for a in (log_r, s11, s12, s22, xy1, xy2, yy)
            )
            log_r -= _logsumexp(log_r)

        score[t] = float(np.sum(np.exp(log_r[: window + 1])))
        r_map[t] = int(np.argmax(log_r))
        y_prev = x
    if return_map:
        return score, r_map
    return score


def zscore_score(y: np.ndarray, window: int = 60, persist: int = 3) -> np.ndarray:
    """Rolling z-score baseline, the rule most dashboards actually use.

    z_t compares y_t with the mean and standard deviation of the previous
    ``window`` points. The score is min(|z|) over the last ``persist``
    points, so an alarm needs ``persist`` consecutive excursions.
    """
    T = len(y)
    z = np.zeros(T)
    c = np.concatenate(([0.0], np.cumsum(y)))
    c2 = np.concatenate(([0.0], np.cumsum(y * y)))
    for t in range(window, T):
        s = c[t] - c[t - window]
        s2 = c2[t] - c2[t - window]
        m = s / window
        v = max(s2 / window - m * m, 1e-9)
        z[t] = abs(y[t] - m) / math.sqrt(v)
    out = np.zeros(T)
    for t in range(persist - 1, T):
        out[t] = float(np.min(z[t - persist + 1 : t + 1]))
    return out


def alarms_from_score(score: np.ndarray, threshold: float, start: int, refractory: int = 100) -> list[int]:
    """Turn a score stream into alarm times with a refractory period."""
    idx = np.flatnonzero(score[start:] > threshold) + start
    out: list[int] = []
    nxt = -1
    for t in idx.tolist():
        if t >= nxt:
            out.append(t)
            nxt = t + refractory
    return out


def long_run_sd_factor(seg: np.ndarray, max_rho: float = 0.95) -> float:
    """sqrt((1 + rho) / (1 - rho)) from the lag one autocorrelation of ``seg``.

    For an AR(1) process the variance of a sum of n consecutive values is
    about n * sigma^2 * (1 + rho) / (1 - rho), not n * sigma^2. CUSUM is a
    running sum, so on autocorrelated data its excursions are larger by this
    factor and a threshold tuned on i.i.d. data fires far too often.
    """
    x = seg - np.mean(seg)
    den = float(np.dot(x, x))
    if den <= 0:
        return 1.0
    rho = float(np.dot(x[1:], x[:-1])) / den
    rho = min(max(rho, 0.0), max_rho)
    return math.sqrt((1.0 + rho) / (1.0 - rho))


def cusum_alarms(
    y: np.ndarray,
    h: float,
    start: int,
    k: float = 0.5,
    refractory: int = 100,
    autocorr_correction: bool = False,
) -> list[int]:
    """Two-sided CUSUM (Page, 1954) with re-baselining after every alarm.

    The baseline mean and standard deviation come from the ``start`` points
    before monitoring begins; after an alarm the detector waits
    ``refractory`` points and re-estimates the baseline from them, which is
    how CUSUM is run in practice on telemetry that does not return to its
    old level.

    With ``autocorr_correction`` the standard deviation is replaced by the
    long run standard deviation (see ``long_run_sd_factor``), estimated from
    the same baseline window. Nothing else changes.
    """
    T = len(y)

    def scale(seg: np.ndarray) -> float:
        sd = float(np.std(seg)) or 1.0
        return sd * long_run_sd_factor(seg) if autocorr_correction else sd

    mu = float(np.mean(y[:start]))
    sd = scale(y[:start])
    sp = sn = 0.0
    out: list[int] = []
    t = start
    while t < T:
        z = (y[t] - mu) / sd
        sp = max(0.0, sp + z - k)
        sn = max(0.0, sn - z - k)
        if sp > h or sn > h:
            out.append(t)
            end = min(T, t + refractory)
            seg = y[t + 1 : end]
            if len(seg) >= 10:
                mu = float(np.mean(seg))
                sd = scale(seg)
            sp = sn = 0.0
            t = end
            continue
        t += 1
    return out
