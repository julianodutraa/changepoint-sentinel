"""Reproduce every number in the README.

Usage: python eval/run_eval.py [--quick]

Writes eval/results.json. The full run takes about one to two minutes on a
laptop CPU and only needs NumPy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentinel.evaluate import (  # noqa: E402
    METHODS,
    build_sets,
    calibrate,
    detection_report,
    fa_per_1000,
)

PHIS = [0.0, 0.5, 0.8]
BUDGET = 1.0  # false alarms per 1000 monitored points


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_cal, n_null, n_change = (8, 8, 12) if args.quick else (30, 30, 60)

    t0 = time.time()
    sets = {phi: build_sets(phi, n_cal, n_null, n_change) for phi in PHIS}
    naive = {m: calibrate(sets[0.0]["cal"], m, BUDGET) for m in METHODS}
    naive_thr = {m: naive[m][0] for m in METHODS}

    results = {
        "config": {"phis": PHIS, "budget_fa_per_1000": BUDGET, "n_cal": n_cal, "n_null": n_null, "n_change": n_change},
        "naive_thresholds": naive_thr,
        "regimes": {},
    }
    for phi in PHIS:
        reg = {}
        for m in METHODS:
            thr_r, feasible = calibrate(sets[phi]["cal"], m, BUDGET)
            reg[m] = {
                "naive": {
                    "threshold": naive_thr[m],
                    "fa_per_1000_null": round(fa_per_1000(sets[phi]["null"], m, naive_thr[m]), 3),
                    "detection": detection_report(sets[phi]["change"], m, naive_thr[m]),
                },
                "per_regime": {
                    "threshold": thr_r,
                    "calibration_feasible": feasible,
                    "fa_per_1000_null": round(fa_per_1000(sets[phi]["null"], m, thr_r), 3),
                    "detection": detection_report(sets[phi]["change"], m, thr_r),
                },
            }
        results["regimes"][str(phi)] = reg
    results["runtime_seconds"] = round(time.time() - t0, 1)

    out = os.path.join(os.path.dirname(__file__), "results.json" if not args.quick else "results_quick.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"{'phi':>4} {'method':<10} {'protocol':<11} {'thr':>7} {'FA/1k':>7} {'recall':>7} {'m1':>6} {'m2':>6} {'m3':>6} {'var':>6} {'delay':>6}")
    for phi in PHIS:
        for m in METHODS:
            for proto in ("naive", "per_regime"):
                r = results["regimes"][str(phi)][m][proto]
                d = r["detection"]
                print(
                    f"{phi:>4} {m:<10} {proto:<11} {r['threshold']:>7} {r['fa_per_1000_null']:>7} "
                    f"{d['all']['recall']:>7} {d['mean_1']['recall']!s:>6} {d['mean_2']['recall']!s:>6} "
                    f"{d['mean_3']['recall']!s:>6} {d['variance']['recall']!s:>6} {d['all']['median_delay']!s:>6}"
                    + ("  (no feasible threshold)" if r.get("calibration_feasible") is False else "")
                )
    print(f"runtime {results['runtime_seconds']}s -> {out}")


if __name__ == "__main__":
    main()
