# changepoint-sentinel

Online regime change detection for autocorrelated pipeline telemetry: a Bayesian online changepoint detector with an AR(1) regime model, benchmarked at a fixed false alarm budget against the detectors teams actually run, including the one line fix that ended up winning.

## Executive summary

Every data platform watches thousands of time series: stage latency, rows per minute, lag, queue depth, warehouse credits. The expensive failure mode is rarely a missed incident. It is the slow erosion of trust that happens when the alerting layer pages people for nothing, until real pages get ignored. Most of that noise has one technical root cause that rarely shows up in dashboards: operational metrics are autocorrelated (a slow minute is usually followed by another slow minute), and nearly every detector in production silently assumes they are not.

This project quantifies that cost and compares the remedies under one rule that a platform leader can hold a team to: **every detector gets the same false alarm budget, one per thousand monitored points, tuned once on clean data and then deployed unchanged.** Under that rule, on synthetic telemetry with the autocorrelation typical of per minute pipeline metrics (lag one autocorrelation 0.8):

| Detector, tuned once on i.i.d. data | False alarms per 1000 points (budget 1.0) | Changes detected |
|---|---|---|
| Rolling z score (dashboard style) | 6.39 | not meaningful at that alarm rate |
| CUSUM | 7.36 | not meaningful at that alarm rate |
| Bayesian online changepoint, textbook i.i.d. model | 9.86 | not meaningful at that alarm rate |
| Bayesian online changepoint, AR(1) regime model (this repo) | 0.61 | 35.0% |
| CUSUM with long run variance correction (one line) | 0.22 | 64.4% |

Three decisions follow. First, a detector tuned once and rolled out to autocorrelated metrics breaks its alarm budget by six to ten times; that is a pager problem, not a modelling nuance. Second, the principled Bayesian fix does hold the budget in every regime with zero retuning, but it pays for it with recall. Third, and this is the result worth acting on, a one line correction to the oldest detector in the table (scale CUSUM by the long run standard deviation, estimated from the same warmup window) holds the budget everywhere and detects nearly twice as many changes as the Bayesian model. The sophisticated method is still useful for what it explains (when a regime started and what changed), not for raising the page.

All numbers are measured, reproducible in about 80 seconds on a laptop with NumPy only, and locked by tests so the README cannot drift from the code.

## What is in the repository

```
sentinel/generate.py   synthetic AR(1) telemetry with Student-t noise and known regime changes
sentinel/detectors.py  BOCPD (i.i.d. and AR(1) regime models), rolling z score, CUSUM, long run variance correction
sentinel/evaluate.py   fixed budget calibration, greedy changepoint matching, detection report
sentinel/demo.py       alarm cards on one series: when, regime start estimate, regime before and after
eval/run_eval.py       reproduces every number in this README into eval/results.json
tests/                 19 tests, including two that assert the README claims against results.json
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                 # 19 tests, under a second
python eval/run_eval.py             # full benchmark, about 80 s, writes eval/results.json
python -m sentinel.demo             # alarm cards for one autocorrelated series
```

## Technical deep dive

### The data

Each series has 1500 points. Noise is AR(1), x_t = phi x_{t-1} + e_t, with Student-t innovations (5 degrees of freedom, scaled to unit variance) because real latency is spiky. The innovation scale is set so that the stationary standard deviation is exactly 1 for every phi, so a mean shift of k is k marginal standard deviations at every level of autocorrelation. Changes start after a 300 point warmup plus 100 points, at least 250 points apart, three per series; 70% are mean shifts of plus or minus 1, 2 or 3, 30% are a 2.5x increase in noise scale. Per regime phi in {0, 0.5, 0.8} there are 30 changepoint free calibration series, 30 changepoint free test series and 60 series with 180 changes in total, all from disjoint seeds.

### Bayesian online changepoint detection

BOCPD (Adams and MacKay, 2007) maintains the posterior over the run length r_t, the time since the last regime change:

```
P(r_t = r + 1, y_1:t) = P(r_{t-1} = r, y_1:t-1) * pred(y_t | regime of length r) * (1 - H)
P(r_t = 0,     y_1:t) = sum_r P(r_{t-1} = r, y_1:t-1) * pred(y_t | regime of length r) * H
```

with constant hazard H = 1/250 and the run length truncated at 400. The alarm score is P(r_t <= 60 | y_1:t), the posterior probability that the current regime is young.

Two conjugate regime models are implemented. The textbook one is Normal-Gamma, observations i.i.d. inside a regime. The one this repository argues for is an AR(1) regime, y_t = c + phi y_{t-1} + e_t, with a Normal-Inverse-Gamma prior on (c, phi, s^2). Its sufficient statistics per run length are X'X (three numbers), X'y (two), y'y and n, so the posterior and the Student-t predictive

```
loc    = x' m_n,                x = [1, y_{t-1}]
scale2 = (b_n / a_n) (1 + x' V_n x),   df = 2 a_n
```

are closed form and vectorised over all run lengths at once (about 0.1 s per 1500 point series).

One engineering decision matters more than it looks. The exact conjugate predictive tightens towards a Gaussian as a regime accumulates data, so one spike from heavy tailed noise is far more probable under a brand new regime than under a well estimated old one, and the detector reads it as a changepoint. In the first version this pushed the calibrated threshold to 0.995 on i.i.d. data and to 0.9999 at phi 0.8. The fix used here caps the predictive degrees of freedom at 5 for every regime (a tempered likelihood, not an exact Student-t noise model, and documented as such in the code), which brought the calibrated threshold back to 0.9 and made calibration transfer possible at all.

### Baselines

Rolling z score: |y_t minus mean| over the trailing 60 points divided by their standard deviation, alarm when three consecutive points exceed the threshold. Two sided CUSUM (Page, 1954) with k = 0.5 on values standardised by the warmup window, re-baselined from the 100 points after every alarm. CUSUM with long run variance correction: identical, except the standard deviation is multiplied by sqrt((1 + rho) / (1 - rho)), where rho is the lag one autocorrelation of the same baseline window. For an AR(1) process the variance of a sum of n points is about n sigma^2 (1 + rho)/(1 - rho), and CUSUM is a running sum.

### Protocol

All detectors see the same robustly standardised input (median and MAD from the warmup window), use a 100 point refractory period after each alarm, and are scored by greedy matching: a change counts as detected if an unused alarm lands within 150 points after it; every other alarm is false. Thresholds are the most sensitive grid value with at most 1 false alarm per 1000 monitored points on the calibration nulls. "Naive" tunes once on phi = 0 and deploys everywhere. "Per regime" retunes on nulls with the test phi, which is the best a fixed threshold can do if you know each metric's autocorrelation.

### Results

False alarms per 1000 points on held out null series, and recall / median delay (points) on the 180 changes per regime.

**Tuned once on i.i.d. data (naive)**

| phi | z score | CUSUM | CUSUM long run | BOCPD i.i.d. | BOCPD AR(1) |
|---|---|---|---|---|---|
| 0.0 | FA 0.81, recall 40.6%, delay 4 | FA 0.92, 89.4%, 8 | FA 0.83, 88.9%, 8.5 | FA 0.75, 76.1%, 13 | FA 0.81, 65.0%, 13 |
| 0.5 | FA 3.14 | FA 5.42 | FA 0.44, 84.4%, 15 | FA 5.86 | FA 1.31, 55.0%, 18 |
| 0.8 | FA 6.39 | FA 7.36 | FA 0.22, 64.4%, 26 | FA 9.86 | FA 0.61, 35.0%, 25 |

Recall is omitted where the detector broke the budget: at seven false alarms per 1000 points and a 150 point tolerance window, a random alarm lands in most windows, and the 90% "recall" those rows report in eval/results.json is mostly luck.

**Retuned per regime (oracle on autocorrelation)**

| phi | z score | CUSUM | CUSUM long run | BOCPD i.i.d. | BOCPD AR(1) |
|---|---|---|---|---|---|
| 0.5 | thr 2.25, FA 0.67, 34.4% | h 16, FA 1.03, 91.7% | h 6, FA 1.06, 86.7% | thr 0.9999, FA 0.39, 55.6% | thr 0.9, FA 1.31, 55.0% |
| 0.8 | thr 3.0, FA 0.67, 27.2% | h 35, FA 1.08, 83.9% | h 4, FA 0.56, 74.4% | no feasible threshold, FA 7.56 at 0.9999 | thr 0.9, FA 0.61, 35.0% |

### What the numbers say, including what is not flattering

The i.i.d. Bayesian detector cannot be calibrated at all at phi = 0.8: its score sits above 0.9 for most of a changepoint free series, because to a model that believes in independence, persistence looks like a sequence of regime changes. No threshold below 1 meets the budget. This is the clearest demonstration in the repository that model misspecification is a calibration problem before it is an accuracy problem.

The AR(1) Bayesian detector is the only method whose false alarm rate needs no retuning in any regime (0.81, 1.31, 0.61; the 1.31 at phi 0.5 is slightly over budget, reported as measured). But its recall on mean shifts collapses with autocorrelation: 95.1% to 27.8% on 3 sigma shifts from phi 0 to 0.8. Part of this is fundamental. Once a regime is described by c + phi y_{t-1}, a level shift of k marginal standard deviations only shows up in the one step residual with size k sqrt((1 - phi)/(1 + phi)), which is k/3 at phi = 0.8, so a 3 sigma shift looks like a 1 sigma shift per point. Part of it is this implementation: the regime model also has to re-learn phi and the noise scale, and a new regime with three free parameters is a weak competitor for many points after the change. Variance increases are the exception and are detected better as autocorrelation grows (37.7% to 59.6%).

The long run variance corrected CUSUM, tuned once with h = 7, stays within budget at every phi (0.83, 0.44, 0.22) and detects 88.9%, 84.4% and 64.4% of changes. It over corrects at phi = 0.8 (0.22 false alarms against a budget of 1), so retuning it per regime buys another ten points of recall (74.4%). A fully retuned plain CUSUM is the strongest detector in the table (83.9% at phi 0.8), but its threshold has to move from 7 to 35, which means per metric calibration on clean history for every series you monitor.

The demo shows the other side of the Bayesian model. On a phi = 0.8 series it fires 18 points after a 2.5x noise increase and dates the new regime to t = 598 against a true 590, with the before and after AR(1) fits (noise 0.57 to 1.37) that an on-call engineer actually wants to read. It also raises one false alarm at t = 350 and misses the third change. That is the recommended split: a corrected CUSUM to page, the AR(1) BOCPD to explain.

### Limitations

Everything is synthetic, with a single AR(1) noise model and no seasonality; real metrics have daily cycles that need to be removed before any of these detectors. The budget is matched on 36000 null points per regime, so false alarm rates carry sampling error of roughly plus or minus 0.2. The BOCPD hazard, window and prior were fixed by hand, not tuned, and a hierarchical prior or the beta divergence robust BOCPD of Knoblauch et al. (2018) would likely narrow the recall gap. The long run variance factor assumes AR(1) structure and is estimated from 300 points.

### References

R. P. Adams and D. J. C. MacKay, Bayesian Online Changepoint Detection, 2007. E. S. Page, Continuous Inspection Schemes, Biometrika, 1954. J. Knoblauch, J. Jewson and T. Damoulas, Doubly Robust Bayesian Inference for Non-Stationary Streaming Data with beta-Divergences, NeurIPS 2018. For autocorrelation and CUSUM, see the literature on residual and long run variance charts in statistical process control.

## License

MIT
