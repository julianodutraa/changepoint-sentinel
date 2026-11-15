"""Synthetic pipeline telemetry with known regime changes.

Each series is an AR(1) process with Student-t innovations, which is a
reasonable stand-in for per-minute stage latency or throughput of a data
pipeline: values are correlated in time and occasionally spiky. Regime
changes are either a mean shift (expressed in units of the stationary
standard deviation, so a "3 sigma" shift means the same thing at every
level of autocorrelation) or a variance increase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Series:
    y: np.ndarray
    phi: float
    changepoints: list[int] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    sizes: list[float] = field(default_factory=list)


def _innovations(rng: np.random.Generator, n: int, df: float) -> np.ndarray:
    # Student-t scaled to unit variance.
    e = rng.standard_t(df, size=n)
    return e * np.sqrt((df - 2.0) / df)


def make_series(
    seed: int,
    phi: float,
    n: int = 1500,
    warmup: int = 300,
    n_changes: int = 3,
    min_gap: int = 250,
    df: float = 5.0,
    null: bool = False,
) -> Series:
    """Generate one series.

    The innovation standard deviation is set so that the stationary
    (marginal) standard deviation is 1 for every phi. Mean shifts of size k
    therefore move the level by k marginal standard deviations.
    """
    rng = np.random.default_rng(seed)
    sigma_e = np.sqrt(1.0 - phi**2)
    cps: list[int] = []
    kinds: list[str] = []
    sizes: list[float] = []
    if not null:
        lo = warmup + 100
        hi = n - 150
        while len(cps) < n_changes:
            cand = sorted(rng.integers(lo, hi, size=n_changes).tolist())
            gaps = np.diff([lo - min_gap] + cand)
            if np.all(gaps >= min_gap):
                cps = cand
        for _ in cps:
            if rng.random() < 0.7:
                kinds.append("mean")
                sizes.append(float(rng.choice([1.0, 2.0, 3.0])) * float(rng.choice([-1.0, 1.0])))
            else:
                kinds.append("variance")
                sizes.append(2.5)

    level = np.zeros(n)
    scale = np.ones(n)
    for cp, kind, size in zip(cps, kinds, sizes):
        if kind == "mean":
            level[cp:] += size
        else:
            scale[cp:] *= size

    e = _innovations(rng, n, df) * sigma_e * scale
    x = np.zeros(n)
    x[0] = e[0] / sigma_e  # stationary start
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    y = level + x
    return Series(y=y, phi=phi, changepoints=cps, kinds=kinds, sizes=sizes)
