import json
import math
import os

import numpy as np
import pytest

from sentinel.detectors import (
    _lgamma_diff_table,
    _robust_logpred,
    _student_logpdf,
    alarms_from_score,
    bocpd_ar_score,
    bocpd_iid_score,
    cusum_alarms,
    long_run_sd_factor,
    robust_standardise,
    zscore_score,
)
from sentinel.evaluate import Prepared, calibrate, match
from sentinel.generate import make_series

HERE = os.path.dirname(__file__)


def gaussian_ar(phi, n, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, math.sqrt(1 - phi**2), n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.8])
def test_generator_marginal_sd_is_one(phi):
    s = make_series(1, phi, n=60_000, null=True)
    assert abs(np.std(s.y[1000:]) - 1.0) < 0.06


def test_generator_is_deterministic_and_respects_gap():
    a = make_series(7, 0.5)
    b = make_series(7, 0.5)
    assert np.array_equal(a.y, b.y)
    assert a.changepoints == b.changepoints
    assert len(a.changepoints) == 3
    assert min(np.diff(a.changepoints)) >= 250
    assert a.changepoints[0] >= 400


def test_lgamma_table():
    tab = _lgamma_diff_table(1.0, 10)
    for n in range(11):
        a = 1.0 + n / 2
        assert tab[n] == pytest.approx(math.lgamma(a + 0.5) - math.lgamma(a))


def test_student_logpdf_integrates_to_one():
    a = 2.5
    lg = math.lgamma(a + 0.5) - math.lgamma(a)
    xs = np.linspace(-200, 200, 400_001)
    dens = np.exp(_student_logpdf(xs, 0.3, 1.7, a, lg))
    assert np.trapezoid(dens, xs) == pytest.approx(1.0, abs=2e-3)


def test_robust_logpred_caps_only_large_df():
    a = np.array([1.0, 2.0, 10.0])
    lg = np.array([math.lgamma(x + 0.5) - math.lgamma(x) for x in a])
    exact = _student_logpdf(6.0, 0.0, 1.0, a, lg)
    capped = _robust_logpred(6.0, 0.0, 1.0, a, lg, max_df=5.0)
    assert capped[0] == pytest.approx(exact[0])
    assert capped[1] == pytest.approx(exact[1])
    assert capped[2] > exact[2]  # a spike is less surprising under the capped tail
    assert np.allclose(_robust_logpred(6.0, 0.0, 1.0, a, lg, None), exact)


def test_bocpd_iid_detects_large_shift_quickly():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 1000)
    y[600:] += 4.0
    s = bocpd_iid_score(y)
    al = alarms_from_score(s, 0.9, start=300)
    assert any(600 <= a <= 615 for a in al)
    assert not any(300 <= a < 600 for a in al)


def test_bocpd_ar_detects_large_shift_and_estimates_start():
    y = gaussian_ar(0.5, 1200, 3)
    y[700:] += 4.0
    s, r = bocpd_ar_score(y, return_map=True)
    al = alarms_from_score(s, 0.9, start=300)
    hits = [a for a in al if 700 <= a <= 760]
    assert hits
    assert abs((hits[0] - r[hits[0]]) - 700) <= 5


def test_iid_model_saturates_on_autocorrelated_null_but_ar_model_does_not():
    """The core finding in miniature: no changepoint, Gaussian AR(1) with phi 0.8."""
    y = gaussian_ar(0.8, 1500, 11)
    iid = bocpd_iid_score(y)[300:]
    ar = bocpd_ar_score(y)[300:]
    assert np.mean(iid > 0.9) > 0.3
    assert np.mean(ar > 0.9) < 0.01


def test_zscore_needs_persistence():
    rng = np.random.default_rng(4)
    y = rng.normal(0, 1, 400)
    y[200] = 50.0  # one isolated spike
    y[300:] += 10.0  # a persistent shift
    s = zscore_score(y, window=60, persist=3)
    assert max(s[200:203]) < 3.0
    assert s[302] > 3.0


def test_alarm_refractory():
    s = np.zeros(1000)
    s[400:420] = 1.0
    s[600] = 1.0
    assert alarms_from_score(s, 0.5, start=300, refractory=100) == [400, 600]


def test_cusum_rebaselines_after_a_persistent_shift():
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, 1500)
    y[800:] += 3.0
    al = cusum_alarms(y, h=8.0, start=300)
    assert any(800 <= a <= 820 for a in al)
    assert len([a for a in al if a > 820]) <= 1


def test_match_is_greedy_and_counts_false_alarms():
    delays, fa = match([350, 510, 515, 900], [500, 880])
    assert delays == [10, 20]
    assert fa == 2


def test_robust_standardise_uses_warmup_only():
    y = np.concatenate([np.full(300, 5.0) + np.arange(300) % 3, np.full(100, 1000.0)])
    z = robust_standardise(y, 300)
    assert abs(np.median(z[:300])) < 1e-9


def test_calibrate_reports_infeasible_budget():
    s = make_series(1, 0.0, null=True)
    z = robust_standardise(s.y, 300)
    p = Prepared(series=s, z=z, scores={"bocpd_iid": np.ones_like(z)})
    thr, feasible = calibrate([p], "bocpd_iid", budget=1.0)
    assert feasible is False


def test_results_json_supports_readme_claims():
    path = os.path.join(HERE, "..", "eval", "results.json")
    r = json.load(open(path))
    budget = r["config"]["budget_fa_per_1000"]
    hi = r["regimes"]["0.8"]
    # Tuned once on clean data: everything except the AR-aware model blows the budget.
    for m in ("zscore", "cusum", "bocpd_iid"):
        assert hi[m]["naive"]["fa_per_1000_null"] > 3 * budget
    assert hi["bocpd_ar"]["naive"]["fa_per_1000_null"] <= 1.5 * budget
    # The i.i.d. BOCPD cannot be calibrated at all on autocorrelated data.
    assert hi["bocpd_iid"]["per_regime"]["calibration_feasible"] is False
    # And the AR-aware model pays for its stability in recall on mean shifts.
    lo = r["regimes"]["0.0"]
    assert hi["bocpd_ar"]["naive"]["detection"]["mean_3"]["recall"] < lo["bocpd_ar"]["naive"]["detection"]["mean_3"]["recall"]
    assert hi["cusum"]["per_regime"]["detection"]["all"]["recall"] > hi["bocpd_ar"]["naive"]["detection"]["all"]["recall"]


def test_long_run_sd_factor_matches_theory():
    y = gaussian_ar(0.8, 50_000, 2)
    assert long_run_sd_factor(y) == pytest.approx(3.0, rel=0.05)
    rng = np.random.default_rng(1)
    assert long_run_sd_factor(rng.normal(size=50_000)) == pytest.approx(1.0, abs=0.02)


def test_results_json_one_line_fix_holds_budget_everywhere():
    path = os.path.join(HERE, "..", "eval", "results.json")
    r = json.load(open(path))
    budget = r["config"]["budget_fa_per_1000"]
    for phi in ("0.0", "0.5", "0.8"):
        assert r["regimes"][phi]["cusum_lrv"]["naive"]["fa_per_1000_null"] <= budget
    hi = r["regimes"]["0.8"]
    assert hi["cusum_lrv"]["naive"]["detection"]["all"]["recall"] > hi["bocpd_ar"]["naive"]["detection"]["all"]["recall"]
