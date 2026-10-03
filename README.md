# Powers of Ten, Applied to AI Scaling

A small PyTorch script that uses the "Powers of Ten" idea (thinking in orders of magnitude) to demonstrate a technique used in real AI work: **fitting a scaling law to small training runs and extrapolating to a bigger model**.

> **Read this first:** all loss data here is **synthetic**, generated from Chinchilla-like constants plus noise. In the main experiment the data comes from *exactly the formula being fitted*, so the results show that the fitting procedure works **when the functional form is right**. They say nothing about how any real model behaves. Section 3b below shows what happens when the form is wrong.

## Run it

```bash
python powers_of_ten_ai.py
```

Needs only `torch` and runs on CPU (works in Google Colab). Runtime was about 10 seconds in the sandbox where it was tested; other machines will differ.

## What the script does

| Section | What it does | Practical relevance |
|---|---|---|
| 1. Film frames | Builds frames from 10^-16 m to 10^24 m and integrates a variable zoom rate r(t) into L(t) = 10^s(t) | Metaphor only: params, tokens, and FLOPs span many decades, so reason in log space |
| 2. Scaling-law fit | Fits `L(N) = E + A / N^alpha` with LBFGS in log space | The real technique: predict a large model's loss from small runs |
| 3. Extrapolation | Predicts loss one decade (10x) past the data, repeated over 50 noise seeds | Tests sensitivity to noise |
| 3b. Distance and misspecification | Measures extrapolation error out to 4 decades, for correct and misspecified data | Tests the two things that actually limit extrapolation |
| 4. Compute projection | Projects FLOPs over 5 years at 4.5x/year | Capacity and budget planning |

## Findings

### Zoom film
- Zoom rate ranges from 0.05 to 0.15 decades/s, slowest in the middle of the leg (t = 120 s).
- Total travel is exactly 24 decades, ending at L = 1.000e24 m.
- The "edge speed" at the end is 3.45e23 m/s, vastly faster than light. That is fine for a visualization but not physical.

### Scaling-law fit (data matches the model form)
Single fit (seed 0), 12 points from 1M to about 3B parameters:

| Parameter | Fitted | True |
|---|---|---|
| E (irreducible loss) | 1.703 | 1.69 |
| alpha | 0.346 | 0.34 |
| Loss at 3e10 params | 1.808 | 1.801 |

Across 50 noise seeds:

| Quantity | Mean +/- std | True |
|---|---|---|
| E | 1.691 +/- 0.012 | 1.69 |
| alpha | 0.340 +/- 0.004 | 0.34 |
| Loss at 3e10 params | 1.802 +/- 0.008 | 1.801 |

- **No detectable bias.** The mean errors in E and alpha are within one standard error of zero (z = 0.74 and 0.63).
- **Accurate when the form is right.** Mean absolute error at 3e10 params is 0.36%.
- **E and alpha are strongly correlated (0.93).** They drift together, which is consistent with the data constraining their *combination* far better than either alone. The extrapolated loss is stable (+/- 0.008) even though E and alpha individually wobble.
- **Diminishing returns, within this model.** Each extra decade of parameters multiplies the reducible loss (loss minus E) by about 0.45. E is the floor *of the fitted form*; whether real models have such a floor is an assumption.

### Extrapolation error vs distance (section 3b)
Mean absolute error over 50 noise seeds. "Decades out" is measured from the last observed size (about 3.2e9 params).

| Params | Decades out | Correctly specified | Misspecified |
|---|---|---|---|
| 3e10 | 1.0 | 0.36% | 2.47% |
| 3e11 | 2.0 | 0.47% | 6.07% |
| 3e12 | 3.0 | 0.54% | 10.65% |
| 3e13 | 4.0 | 0.58% | 15.63% |

"Misspecified" means the data has an extra term, `60 / N^0.15`, that the 3-parameter model cannot represent. That term is an arbitrary illustration, not a measurement of real models, so treat the exact percentages as illustrative.

What this shows:
- When the form is right, error barely grows with distance. Having only 12 points is *not* what limits extrapolation here.
- When the form is wrong, error grows steadily with distance, because the fit absorbs the missing term into the wrong parameters. Real loss curves are never exactly a pure power law, so this is the realistic risk.

### Compute growth
At an assumed 4.5x per year (0.65 decades/year, doubling time about 5.5 months):

| Year | FLOPs |
|---|---|
| 0 | 1.00e24 |
| 1 | 4.50e24 |
| 2 | 2.03e25 |
| 3 | 9.11e25 |
| 4 | 4.10e26 |
| 5 | 1.85e27 |

The 4.5x/year rate is an **assumption**, chosen inside the 4-5x/year range that Epoch AI recommends for summarizing the historical trend in frontier training compute ([source](https://epoch.ai/blog/training-compute-of-frontier-ai-models-grows-by-4-to-5x-per-year)). Their separate decomposition of post-2018 frontier models gives a central estimate of about 4.2x/year ([source](https://epoch.ai/data-insights/training-compute-decomposition)). Those are historical trends, not forecasts, so the projection is illustrative.

## Changes from the original script

- **Zoom shape:** the comment said "slower in the middle" but the sine made the zoom slowest at t = 180 s. Switched to a cosine so it is slowest at t = 120 s.
- **Integration:** replaced `cumsum * dt` with `cumulative_trapezoid`, validated against the closed-form integral. The original double-counted an endpoint and ended at 1.023e24 m instead of 1.000e24 m.
- **Fit robustness:** LBFGS learning rate raised from 0.5 to 1.0, tolerances tightened, and the final fit error is now reported.
- **Noise generation:** uses a per-seed `torch.Generator` instead of the global seed, so each seed is independent and reproducible.
- **New analysis:** the 50-seed sweep (section 3) and the distance/misspecification sweep (section 3b).
- **Compute comment:** the 4.5x/year comment now names its source and says it is an assumption.

## Limitations

- Synthetic data only. Swap in real training-run losses to use this for anything real.
- The main experiment fits the exact formula that generated the data, which is the best case. Section 3b shows how much worse it gets otherwise.
- The model covers parameters only. Real planning usually also varies data: `L(N, D) = E + A / N^alpha + B / D^beta`.
- Scaling laws can break in new regimes (data limits, architecture changes), which no fit on past runs can see.

## Next steps

- Replace the synthetic data with losses from your own runs at several model sizes.
- Add the data/tokens dimension to the fit.
- Plot the fit and extrapolation with uncertainty bands.
- Convert the compute projection into hardware and cost estimates.
