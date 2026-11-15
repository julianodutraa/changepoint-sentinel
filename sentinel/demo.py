"""End to end demo on one autocorrelated series.

Usage: python -m sentinel.demo [--seed N] [--phi 0.8]

Runs the AR(1)-aware BOCPD with the threshold calibrated once on i.i.d.
data and prints an alarm card for every alarm: when it fired, when the
model thinks the regime started (MAP run length), and the AR(1) regime
estimated before and after. The true changepoints are printed alongside so
the output can be checked by eye.
"""

from __future__ import annotations

import argparse

import numpy as np

from .detectors import alarms_from_score, bocpd_ar_score, robust_standardise
from .generate import make_series

NAIVE_THRESHOLD = 0.9  # calibrated on phi = 0 null data by eval/run_eval.py
WARMUP = 300


def fit_ar1(y: np.ndarray) -> tuple[float, float, float]:
    """OLS fit of y_t = c + phi * y_{t-1} + e; returns (level, phi, sigma_e)."""
    x, t = y[:-1], y[1:]
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, t, rcond=None)
    c, phi = float(coef[0]), float(coef[1])
    resid = t - X @ coef
    level = c / (1.0 - phi) if abs(1.0 - phi) > 1e-6 else float(np.mean(y))
    return level, phi, float(np.std(resid))


def run(seed: int, phi: float) -> str:
    s = make_series(seed, phi)
    z = robust_standardise(s.y, WARMUP)
    score, r_map = bocpd_ar_score(z, return_map=True)
    alarms = alarms_from_score(score, NAIVE_THRESHOLD, start=WARMUP)
    lines = [f"series seed={seed} phi={phi} length={len(z)}"]
    for cp, kind, size in zip(s.changepoints, s.kinds, s.sizes):
        desc = f"mean {size:+.0f} sd" if kind == "mean" else f"variance x{size}"
        lines.append(f"  true change at t={cp}: {desc}")
    for a in alarms:
        start = a - int(r_map[a])
        pre = z[max(0, start - 200) : start]
        post = z[start : min(len(z), a + 100)]
        lp, pp, sp = fit_ar1(pre) if len(pre) > 20 else (float("nan"),) * 3
        lq, pq, sq = fit_ar1(post) if len(post) > 20 else (float("nan"),) * 3
        lines.append(
            f"ALARM t={a}  P(run<=60)={score[a]:.3f}  regime start t~{start}\n"
            f"    before: level {lp:+.2f}  phi {pp:.2f}  noise {sp:.2f}\n"
            f"    after : level {lq:+.2f}  phi {pq:.2f}  noise {sq:.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=30_800_001)
    ap.add_argument("--phi", type=float, default=0.8)
    a = ap.parse_args()
    print(run(a.seed, a.phi))


if __name__ == "__main__":
    main()
