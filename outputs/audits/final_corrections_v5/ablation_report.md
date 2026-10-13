# Five-seed one-feature ablation report

All 50 prespecified feature/model/seed runs completed. Each row adds exactly one engineered feature to the six grouped predictors, and is paired against the same-seed, same-architecture baseline. Validation macro-F1 selects epoch; test keys are identical to the locked v4 cohort (671 rows).

Positive paired differences favor the one-feature input relative to its paired six-feature baseline. Mean and SD describe seed variability on this fixed cohort; they are not inferential intervals or causal effects.

| Feature added | Model | Mean paired macro-F1 | SD across seeds | Higher / equal / lower seeds |
|---|---|---:|---:|---:|
| ACTIVE_CRIME_GROUPS | GRU | +0.00152 | 0.00350 | 3/0/2 |
| ACTIVE_CRIME_GROUPS | LSTM | +0.00446 | 0.00571 | 4/0/1 |
| CRIME_DIVERSITY_INDEX | GRU | -0.00178 | 0.00848 | 2/0/3 |
| CRIME_DIVERSITY_INDEX | LSTM | +0.00580 | 0.00577 | 5/0/0 |
| PREV_YEAR_TOTAL_CRIME | GRU | -0.00106 | 0.00580 | 3/0/2 |
| PREV_YEAR_TOTAL_CRIME | LSTM | +0.00487 | 0.00339 | 5/0/0 |
| TOTAL_GROUPED_CRIME | GRU | -0.00425 | 0.00454 | 1/0/4 |
| TOTAL_GROUPED_CRIME | LSTM | +0.00186 | 0.00322 | 3/0/2 |
| YOY_TOTAL_CRIME_CHANGE | GRU | -0.00511 | 0.01408 | 2/0/3 |
| YOY_TOTAL_CRIME_CHANGE | LSTM | -0.00156 | 0.00978 | 2/0/3 |

Saved exact per-seed metrics, validation selections, test predictions, model files, integrity checks, run config, run status and SHA-256 manifest are in this directory.
No p-values or confidence intervals are provided. These are one-feature-at-a-time descriptive comparisons, not causal feature importance.
