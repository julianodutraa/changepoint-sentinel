"""Calibration and scoring of detectors on synthetic telemetry.

The comparison is made at a fixed false alarm budget. For every detector
the alarm threshold is the most sensitive value on a grid whose false
alarm rate on changepoint-free calibration series stays within the budget.
Two calibration protocols are reported:

* ``naive``: thresholds tuned on i.i.d. data (phi = 0) and then deployed
  everywhere. This is what happens when a detector is tuned once on a
  clean benchmark and rolled out to real telemetry.
* ``per_regime``: thresholds re-tuned on null data with the same
  autocorrelation as the test data. This is the best a fixed threshold can
  do and isolates detection power from calibration error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .detectors import (
    alarms_from_score,
    bocpd_ar_score,
    bocpd_iid_score,
    cusum_alarms,
    robust_standardise,
    zscore_score,
)
from .generate import Series, make_series

WARMUP = 300
TOLERANCE = 150
REFRACTORY = 100

GRIDS: dict[str, np.ndarray] = {
    "zscore": np.round(np.arange(1.0, 12.01, 0.25), 3),
    "cusum": np.round(np.arange(2.0, 60.01, 1.0), 3),
    "cusum_lrv": np.round(np.arange(2.0, 60.01, 1.0), 3),
    "bocpd_iid": np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
    "bocpd_ar": np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
}
METHODS = list(GRIDS)


@dataclass
class Prepared:
    series: Series
    z: np.ndarray
    scores: dict[str, np.ndarray]


def prepare(series: Series) -> Prepared:
    z = robust_standardise(series.y, WARMUP)
    scores = {
        "zscore": zscore_score(z),
        "bocpd_iid": bocpd_iid_score(z),
        "bocpd_ar": bocpd_ar_score(z),
    }
    return Prepared(series=series, z=z, scores=scores)


def alarms(p: Prepared, method: str, thr: float) -> list[int]:
    if method == "cusum":
        return cusum_alarms(p.z, thr, start=WARMUP, refractory=REFRACTORY)
    if method == "cusum_lrv":
        return cusum_alarms(p.z, thr, start=WARMUP, refractory=REFRACTORY, autocorr_correction=True)
    return alarms_from_score(p.scores[method], thr, start=WARMUP, refractory=REFRACTORY)


def match(alarm_times: list[int], cps: list[int], tol: int = TOLERANCE):
    """Greedy matching: each changepoint takes the first unused alarm in [cp, cp + tol]."""
    used = set()
    delays: list[int | None] = []
    for cp in cps:
        hit = None
        for a in alarm_times:
            if a in used:
                continue
            if cp <= a <= cp + tol:
                hit = a
                break
        if hit is None:
            delays.append(None)
        else:
            used.add(hit)
            delays.append(hit - cp)
    false_alarms = len([a for a in alarm_times if a not in used])
    return delays, false_alarms


def fa_per_1000(prepared: list[Prepared], method: str, thr: float) -> float:
    total_alarms = 0
    total_points = 0
    for p in prepared:
        a = alarms(p, method, thr)
        _, fa = match(a, p.series.changepoints)
        total_alarms += fa
        total_points += len(p.z) - WARMUP
    return 1000.0 * total_alarms / max(total_points, 1)


def calibrate(null_prepared: list[Prepared], method: str, budget: float = 1.0) -> tuple[float, bool]:
    """Most sensitive threshold on the grid whose false alarm rate is within budget.

    Returns (threshold, feasible). When no grid value meets the budget the
    least sensitive value is returned with feasible = False. For a
    probability score that saturates near 1 this is not a grid problem: no
    threshold below 1 can meet the budget.
    """
    grid = GRIDS[method]
    for thr in grid:  # grids are ordered from most to least sensitive
        if fa_per_1000(null_prepared, method, float(thr)) <= budget:
            return float(thr), True
    return float(grid[-1]), False


def detection_report(change_prepared: list[Prepared], method: str, thr: float) -> dict:
    rows = []
    total_fa = 0
    total_points = 0
    for p in change_prepared:
        a = alarms(p, method, thr)
        delays, fa = match(a, p.series.changepoints)
        total_fa += fa
        total_points += len(p.z) - WARMUP
        for d, kind, size in zip(delays, p.series.kinds, p.series.sizes):
            rows.append((kind, abs(size), d))

    def summarise(sel: Callable[[tuple], bool]) -> dict:
        r = [x for x in rows if sel(x)]
        hits = [x[2] for x in r if x[2] is not None]
        return {
            "n": len(r),
            "recall": round(len(hits) / len(r), 4) if r else None,
            "median_delay": float(np.median(hits)) if hits else None,
        }

    return {
        "all": summarise(lambda x: True),
        "mean_1": summarise(lambda x: x[0] == "mean" and x[1] == 1.0),
        "mean_2": summarise(lambda x: x[0] == "mean" and x[1] == 2.0),
        "mean_3": summarise(lambda x: x[0] == "mean" and x[1] == 3.0),
        "variance": summarise(lambda x: x[0] == "variance"),
        "fa_per_1000_on_change_series": round(1000.0 * total_fa / max(total_points, 1), 3),
    }


def build_sets(phi: float, n_cal: int, n_null: int, n_change: int) -> dict[str, list[Prepared]]:
    off = int(round(phi * 1000)) * 1000
    cal = [prepare(make_series(10_000_000 + off + i, phi, null=True)) for i in range(n_cal)]
    null = [prepare(make_series(20_000_000 + off + i, phi, null=True)) for i in range(n_null)]
    chg = [prepare(make_series(30_000_000 + off + i, phi)) for i in range(n_change)]
    return {"cal": cal, "null": null, "change": chg}
