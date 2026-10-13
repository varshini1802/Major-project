# NCRB project final handoff

This handoff reflects the v5 audits plus the completed five-seed one-feature ablation. The project is technically documented and reproducible for the available processed files and saved experiments, but several source-provenance and operational-availability dependencies remain unresolved.

## Frozen experiment

- Ordered project target boundaries: 107, 237, 362, 563, fit on 2001–2010 training observations; no revised target definition.
- Chronological split: train 2001–2010, validation 2011–2012, locked test 2013–2014, 671 test district-years.
- Tabular forecasting uses district-keyed t−1 predictors; sequences require consecutive same-district t−3, t−2, t−1 records. Validation chooses models/epochs; preprocessing is refit using train plus validation before locked test inference.
- The observed-year random-split notebook is a separate classification experiment. Its same-year crime counts and legacy 90.76% figure are not corrected forecasting results.

## Experiments

- Saved corrected-v4 metrics were independently recomputed; all eight variant/model rows match. The v4 model suite was not retrained.
- Five full paired baseline/enhanced model runs (seeds 11, 23, 37, 53, 71) completed. Mean enhanced-minus-baseline macro-F1: GRU +0.00785 (seed SD 0.00772), LSTM +0.01248 (0.00767), logistic regression +0.00016 (0). Metric direction is mixed; no universal enhanced-feature superiority or statistical significance is claimed.
- Five-seed one-feature ablation completed 50/50 GRU/LSTM × five features × five seeds. Every prediction artifact matches the locked 671 test keys and same-seed baseline pairing. See `ablation_report.md`, `ML_Implementation/corrected_temporal_results/corrected_v5_feature_ablation_multiseed/paired_difference_summary.csv`, and `integrity_checks.csv`. Results are fixed-test descriptive differences, not causal feature importance or inferential uncertainty.

## Source and data findings

- Nine direct IPC overlaps between grouped predictors and target construction are documented. IPC legal definitions and empirical parent/component relationships do not reveal NCRB's statistical case-counting rules; indirect overlap semantics remain partial.
- The normalized `COUNTERFIETING` heading in 2001–2013 is not silently mapped to canonical `IPC_COUNTERFEITING`. Alternative 2014 attempt-to-murder and burglary headings also remain unmapped pending an official year crosswalk.
- Seven duplicate normalized district-year key groups across five cleaned files (14 rows): five with identical offence payloads and two conflicting Railway groups. Integration marks all for duplicate review and excludes them from core. Railway identities/count provenance are unavailable; no records were changed.
- Group formulas and `min_count=1` outputs reconcile with zero value mismatches. Partial sums are incomplete totals; all-missing components stay missing; observed zero is separately counted.
- Original upstream raw files, administrative-boundary/version provenance and semantic missingness causes are unavailable. Available normalized-to-integrated reconciliations are recorded and do not establish pre-normalization provenance.

## Audit and test state

This closure reran target/source audit, forecast-availability audit, raw-stage reconciliation, duplicate-key audit, missingness reconciliation, saved-v4 metric audit, the multi-seed ablation integrity audit, 26 unit tests and Python compilation. Structural lag, sequence and test-lock checks pass. Annual source release timestamps and a forecast issuance date are absent, so actual operational availability remains unverified.

## Paper and closure package

No original manuscript or conference template was found. A new draft exists at `outputs/reports/conference_paper_draft.md`; it includes verified v4 and multi-seed results, exact five-seed ablation summary, limitations and sourced references. It still requires author review, venue formatting, and resolution or explicit acceptance of source limitations. The project README is at `README.md`; inventory and exact-duplicate listing are at `outputs/reports/project_inventory.md` and `outputs/reports/exact_duplicate_files.csv`; correction statuses and evidence paths are in `correction_status.csv`.

## Remaining dependencies

1. Official NCRB year-specific schemas/crosswalks and statistical category-counting documentation.
2. Original raw files and Railway unit identifiers for upstream record provenance.
3. Administrative boundary provenance and documentation explaining missing values.
4. Publication/release dates and a defined issuance date for operational forecast availability.
5. Author approval of the new paper draft and venue/template selection.
