"""changepoint-sentinel: Bayesian online changepoint detection for autocorrelated telemetry."""

from .generate import Series, make_series
from .detectors import (
    bocpd_iid_score,
    bocpd_ar_score,
    zscore_score,
    alarms_from_score,
    cusum_alarms,
    long_run_sd_factor,
)

__all__ = [
    "Series",
    "make_series",
    "bocpd_iid_score",
    "bocpd_ar_score",
    "zscore_score",
    "alarms_from_score",
    "cusum_alarms",
    "long_run_sd_factor",
]
