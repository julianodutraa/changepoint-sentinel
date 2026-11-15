# The detector that believed in independence

*Why changepoint detectors flood on-call when they meet real telemetry, what a Bayesian fix buys, and why a one line correction to a 1954 algorithm beat it.*

Every on-call engineer has a metric they have learned to ignore. It pages twice a week, someone opens the dashboard, the line wiggles a little higher than usual, and the page gets acknowledged and forgotten. Nobody decides to stop trusting the alerting layer. It happens one false page at a time, and the day a real regime change arrives, it gets the same shrug.

I wanted to know how much of that noise comes from one specific, well understood, and almost universally ignored property of operational data: autocorrelation. A pipeline stage that is slow this minute is very likely slow the next minute too. Lag, queue depth, rows per minute, and warehouse credit burn all behave this way. And nearly every detector teams deploy, from a rolling z score on a dashboard to the textbook version of Bayesian online changepoint detection, assumes that consecutive points are independent.

## Setting up a fair fight

Comparing detectors is easy to do badly. Any method looks great if you let it alarm often enough, and any method looks safe if you let it miss enough. So the experiment fixes the one quantity a platform team actually negotiates with its on-call rotation: the false alarm budget. Every detector gets exactly one false alarm per thousand monitored points, measured on changepoint free calibration series, and the threshold is the most sensitive value that stays within that budget. Then it is judged on changes it has never seen, and on separate null series for its real false alarm rate.

The data is synthetic, so ground truth is exact. Each series is an AR(1) process with heavy tailed Student-t noise, scaled so its stationary standard deviation is one regardless of autocorrelation, which means a "3 sigma" shift means the same thing everywhere. Each series carries three changes: mean shifts of one, two or three standard deviations, or a 2.5x increase in noise. I ran three regimes, lag one autocorrelation 0, 0.5 and 0.8, the last being typical of per minute infrastructure metrics. Calibration, null test and change series come from disjoint seeds.

The decisive design choice was to evaluate two protocols. In the naive one, every detector is tuned once on independent data and then deployed unchanged everywhere, which is how detectors get rolled out across thousands of metrics. In the per regime one, each detector is retuned on null data with the right autocorrelation, which is the best a fixed threshold can do but requires knowing every metric's structure in advance.

## What happens when you tune once

At autocorrelation 0.8, the rolling z score raised 6.4 false alarms per thousand points, plain CUSUM 7.4, and the textbook i.i.d. Bayesian online changepoint detector 9.9. Against a budget of one, that is six to ten times more pages than promised.

Those detectors also report around 90% recall in that setting, and it would be easy to put that number on a slide. It is mostly luck. At seven alarms per thousand points and a 150 point tolerance window, a random alarm lands near most real changes. A detector that breaks its budget has no meaningful recall, and the repository says so rather than quoting it.

The Bayesian detector is the most instructive failure. Retuned specifically for autocorrelated data, it still cannot meet the budget: there is no threshold below one that works. Its alarm score, the posterior probability that the current regime is younger than sixty points, sits above 0.9 for most of a series with no changes at all. To a model that believes observations are independent, persistence is indistinguishable from a sequence of small regime changes. Misspecification here is not an accuracy problem. It is a calibration problem, and it is unfixable by threshold.

## The principled fix

The natural response is to give the Bayesian detector the right model. In this repository each regime is an AR(1) process with its own intercept, autocorrelation and noise, with a conjugate Normal-Inverse-Gamma prior. The sufficient statistics per candidate run length are a handful of sums, so the posterior and the Student-t predictive stay closed form and vectorise over every run length at once. A 1500 point series runs in about a tenth of a second.

One detail cost me an iteration and is worth sharing. The exact conjugate predictive becomes almost Gaussian as a regime accumulates data. With heavy tailed noise, a single spike is then much more likely under a brand new regime than under a well estimated one, so the detector calls a changepoint on every spike. The first version needed a threshold of 0.995 on clean data and 0.9999 on autocorrelated data. Capping the predictive degrees of freedom at five for every regime, a tempered likelihood rather than an exact heavy tailed model, brought the threshold down to 0.9 everywhere.

With that, the AR(1) detector does exactly what it promises on false alarms: 0.81, 1.31 and 0.61 per thousand across the three regimes, with a single threshold tuned on independent data. It is the only Bayesian variant that transfers.

Then you look at recall. On three sigma mean shifts it drops from 95% at no autocorrelation to 28% at 0.8. Some of that is fundamental: once a regime is described as c plus phi times the previous value, a level shift shows up in the one step residual with size k times the square root of (1 minus phi) over (1 plus phi). At phi 0.8 that factor is one third, so a three sigma shift looks like a one sigma shift per point. The rest is the cost of flexibility: a new regime that must also relearn its autocorrelation and noise is a weak competitor for many points after a change.

## The one line fix

CUSUM, published by Page in 1954, sums standardised deviations and alarms when the sum crosses a threshold. On autocorrelated data, the variance of a running sum is not n times the variance of one point; for AR(1) it is approximately n times that variance times (1 plus rho) over (1 minus rho). So I changed one thing: scale by the long run standard deviation, with rho estimated from the same warmup window CUSUM already uses for its baseline.

Tuned once on independent data with the same threshold of seven, this corrected CUSUM stayed within budget at every level of autocorrelation, at 0.83, 0.44 and 0.22 false alarms per thousand, and detected 89%, 84% and 64% of changes. At high autocorrelation that is nearly double the Bayesian detector's 35%. It over corrects, since 0.22 is well under the budget, and retuning it per regime buys another ten points of recall. A plain CUSUM retuned per regime is the strongest detector in the table at 84%, but its threshold has to move from 7 to 35, which is a per metric calibration program, not a configuration change.

## Where the Bayesian model still earns its place

Paging is a binary decision. Diagnosis is not. On one autocorrelated test series, the AR(1) detector fires 18 points after a 2.5x noise increase, dates the start of the new regime to t equals 598 against a true 590, and reports the regime before and after: same autocorrelation, noise from 0.57 to 1.37. That is the sentence an on-call engineer wants to read when the page arrives. It also produces a false alarm early in the series and misses a later change, and the demo prints those too.

So the recommendation is a split, not a winner: a long run variance corrected CUSUM to decide whether to page, and the AR(1) Bayesian detector to explain what changed and when.

## What I would not claim

This is synthetic data with one noise model and no seasonality. Real metrics have daily cycles that must be removed first. False alarm rates carry sampling error of roughly 0.2 per thousand at this sample size. The Bayesian hyperparameters were fixed by hand, and robust variants such as beta divergence BOCPD might close part of the gap. The correction factor assumes AR(1) structure.

The broader lesson is about where sophistication should go. The expensive mistake was not choosing the wrong algorithm. It was evaluating detectors without a fixed alarm budget and on data without the one property that defines operational telemetry. Once those two things are in place, the cheap fix and the principled model each find their job.

The code, the benchmark and the tests that lock every number above are at https://github.com/julianodutraa/changepoint-sentinel.
