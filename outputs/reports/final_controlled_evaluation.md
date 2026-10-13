# Controlled corrected forecast evaluation

Generated from saved corrected-v4 experiment outputs. Target thresholds, classes, and temporal partitions are frozen from the prior target correction.

## Protocol

Shared strict sequence cohort: 5,842 targets (train 3,847; validation 1,324; test 671). Test years are 2013–2014; supports are class 1: 113, class 2: 76, class 3: 105, class 4: 145, class 5: 232. The baseline and enhanced variants use the exact same target keys and labels.

GRU and LSTM pairs have identical architecture, optimizer, loss, batch size, stopping/epoch-selection procedure, and seed within each pair. Epochs and logistic C were selected on validation only. Preprocessing was fit on training sequences for validation selection and refit on train+validation for final test prediction. Test data was not used for selection.

Multiclass ROC-AUC is macro one-vs-rest. The reported PR metric is macro unweighted mean one-vs-rest average precision (AP), not trapezoidal PR-curve area.

## Full feature set and baselines

| Variant | Model | Accuracy | Macro F1 | Weighted F1 | Macro OvR ROC-AUC | Macro OvR AP | n |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | GRU | 0.7168 | 0.7020 | 0.7228 | 0.9375 | 0.7246 | 671 |
| enhanced | GRU | 0.7198 | 0.7054 | 0.7250 | 0.9345 | 0.7133 | 671 |
| baseline | LSTM | 0.7228 | 0.7041 | 0.7275 | 0.9374 | 0.7209 | 671 |
| enhanced | LSTM | 0.7198 | 0.6996 | 0.7240 | 0.9354 | 0.7244 | 671 |
| baseline | LogisticRegression | 0.6960 | 0.6761 | 0.7019 | 0.9327 | 0.7095 | 671 |
| enhanced | LogisticRegression | 0.6975 | 0.6762 | 0.7026 | 0.9312 | 0.7059 | 671 |
| baseline | Majority baseline | 0.3458 | 0.1028 | 0.1777 | 0.5000 | 0.2000 | 671 |
| enhanced | Majority baseline | 0.3458 | 0.1028 | 0.1777 | 0.5000 | 0.2000 | 671 |

## Per-class metrics: same-architecture neural comparisons

Each cell reports precision / recall / F1. The test support is identical for every model.

| Class (support) | Baseline GRU | Enhanced GRU | Baseline LSTM | Enhanced LSTM |
|---|---:|---:|---:|---:|
| 1 — Lowest violent-crime burden (113) | 0.906 / 0.938 / 0.922 | 0.870 / 0.947 / 0.907 | 0.883 / 0.938 / 0.910 | 0.863 / 0.947 / 0.903 |
| 2 — Low-to-moderate violent-crime burden (76) | 0.538 / 0.750 / 0.626 | 0.565 / 0.684 / 0.619 | 0.540 / 0.711 / 0.614 | 0.535 / 0.697 / 0.606 |
| 3 — Moderate violent-crime burden (105) | 0.541 / 0.562 / 0.551 | 0.563 / 0.638 / 0.598 | 0.556 / 0.571 / 0.563 | 0.556 / 0.571 / 0.563 |
| 4 — Moderate-to-high violent-crime burden (145) | 0.582 / 0.614 / 0.597 | 0.575 / 0.607 / 0.591 | 0.600 / 0.621 / 0.610 | 0.595 / 0.607 / 0.601 |
| 5 — Highest violent-crime burden (232) | 0.914 / 0.733 / 0.813 | 0.918 / 0.728 / 0.812 | 0.907 / 0.754 / 0.824 | 0.911 / 0.754 / 0.825 |

## One-at-a-time engineered-feature ablations

Values are test macro-F1 change versus the same-architecture six-group baseline; all use the same 671 test keys.

| Added predictor | GRU delta | LSTM delta |
|---|---:|---:|
| TOTAL_GROUPED_CRIME | +0.0043 | -0.0006 |
| ACTIVE_CRIME_GROUPS | -0.0047 | +0.0059 |
| CRIME_DIVERSITY_INDEX | +0.0018 | +0.0062 |
| PREV_YEAR_TOTAL_CRIME | +0.0053 | +0.0031 |
| YOY_TOTAL_CRIME_CHANGE | -0.0232 | +0.0019 |

The full enhanced set raises GRU macro F1 by about 0.0034 but lowers its ROC-AUC and AP. For LSTM it lowers macro F1 and ROC-AUC while slightly raising AP. One-feature ablation effects are mixed and small; they are single-seed descriptive results, not evidence of reliable gains.

## Saved artifacts and limits

Detailed per-class reports, confusion matrices, test keys, predictions, probabilities, AUC files, and configurations are in `ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/` and `.../corrected_v4_feature_ablation/`. Figures in the controlled directory were generated from those saved CSV result tables.

Prior corrected-v2 and corrected-v3 model outputs were preserved. Corrected-v3 supplemental AUC results are in `corrected_v3/supplemental_metrics_auc/`; no legacy same-year score is treated as forecasting performance.

Annual source-release dates, original upstream raw provenance, IPC parent/component relationships, and causes of NaN source observations remain unresolved. No conference-paper manuscript was found, so no paper sections were changed.
