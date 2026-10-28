# Computational Analysis of Crime and Social System Using Machine Learning

This repository studies district-year violent-crime burden using integrated NCRB crime tables. The primary experiment forecasts the burden class for year *t* from data available through *t−1*. The notebook for same-year inputs is a separate observed-year classification experiment; its scores are not forecasting results.

## Data and target

The project target is `TARGET_VIOLENT_LEVEL`, derived from `VIOLENT_CRIME_BURDEN` using frozen boundaries 107, 237, 362, and 563. The boundaries were fitted on 2001–2010 training observations. The temporal split is train 2001–2010, validation 2011–2012, and locked test 2013–2014. Tabular forecasting inputs are district-keyed one-year lags. Sequence forecasting requires that district's consecutive *t−3*, *t−2*, and *t−1* records.

Original upstream source files and publication dates are not available in this repository. Several source-heading equivalences and the 2014 raw-stage discrepancy remain unresolved. See `outputs/audits/final_corrections_v5/` and `outputs/audits/final_release_readiness/` for the evidence and limitations.

## Project structure

| Folder | Purpose |
|---|---|
| `data/raw/` | Source-stage files retained for the project. |
| `data/processed/` | Integrated and feature-engineered district-year datasets. |
| `src/` | Reusable preprocessing, target, feature, and audit modules. |
| `ML_Implementation/` | Forecasting input builders, model experiments, and evaluation utilities. The directory name is retained for compatibility with archived paths. |
| `ML_Implementation/corrected_temporal_results/` | Saved chronological experiment versions, predictions, metrics, models, and run configurations. These are historical result identifiers and must not be renamed casually. |
| `notebooks/` | Source integration, exploratory analysis, and observed-year classification notebooks. |
| `tests/` | Automated checks for target construction, feature availability, leakage contracts, and sequence generation. |
| `outputs/audits/` | Data, leakage, reproducibility, and release-readiness evidence. |
| `outputs/reports/` | Project reports and manuscript drafts. |
| `outputs/tables/` | Generated audit tables and summaries. |
| `outputs/figures/` | Figures generated from saved results. |

## Project File Guide

See [PROJECT_FILE_GUIDE.md](PROJECT_FILE_GUIDE.md) for a plain-language explanation of the main files, how they fit together, and which scripts are safe to run without training models.

## Current experiment and evidence

- **Research Audit Report**: [docs/research/experiment_audit_report.md](docs/research/experiment_audit_report.md) provides the complete repository audit, base paper comparison, leakage audit, and test status.
- **Reproducibility Guide**: [docs/research/experiment_reproducibility.md](docs/research/experiment_reproducibility.md) documents exact environment dependencies, execution commands, and random seeds.
- **Revised Conference Paper**: [docs/paper/revised_paper_draft.md](docs/paper/revised_paper_draft.md) (and [outputs/reports/revised_conference_paper_draft.md](outputs/reports/revised_conference_paper_draft.md)) updates only Methodology and Results (Sections 3 and 4.3), strictly preserving the existing document structure.
- **Original Paper Draft**: [docs/paper/original_paper_draft.md](docs/paper/original_paper_draft.md) preserves the original submitted student draft (`draft paper.pdf`) for provenance.
- **Claims & Inconsistencies Register**: [docs/paper/claims_and_inconsistencies_register.md](docs/paper/claims_and_inconsistencies_register.md) documents unedited abstract/intro claims vs reproduced empirical findings.
- **Final Scientific Validation Report**: [docs/research/final_scientific_validation_report.md](docs/research/final_scientific_validation_report.md) provides the 7-section independent certification of deliverables, test-set independence, metrics, and readiness.
- **Before-and-After Comparison Notebook**: [notebooks/08_model_performance_before_after_corrections.ipynb](notebooks/08_model_performance_before_after_corrections.ipynb) provides executable, reproducible comparison tables and publication plots contrasting the historical same-year classification setup with the corrected chronological forecasting setup.
- **Feature Evaluation Summary Table**: [outputs/tables/feature_evaluation_study_summary.csv](outputs/tables/feature_evaluation_study_summary.csv) consolidates multi-seed model comparisons and single-feature ablations.
- The current controlled forecasting evaluation uses the saved results under `ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/`.
- Five-seed paired robustness results are under `ML_Implementation/corrected_temporal_results/corrected_v5_multiseed/`.
- One-feature ablations are under `corrected_v4_feature_ablation/` and `corrected_v5_feature_ablation_multiseed/` within the result directory.
- The observed-year classification notebook is `notebooks/observed_year_classification_models.ipynb`.
- The release-readiness report is `outputs/audits/final_release_readiness/release_readiness_report.md`.

The direct-overlap audit found nine IPC source columns shared with target construction. Their observed counts sum to 76.592% of target burden summed over rows with valid targets, but this ratio is not a deterministic reconstruction. Evidence does not establish that enhanced predictors are universally superior or that the historical information was operationally available at forecast time. Do not report the legacy 90.76% same-year score as forecasting performance.

## Reproduction and checks

Use Python 3.11.9 and the package versions listed in `outputs/audits/final_corrections_v5/reproducibility_manifest.json`. Install `requirements.txt`; TensorFlow 2.18.1 is additionally required for neural model runs.

To regenerate paper figures from existing saved outputs without fitting models:

```powershell
python ML_Implementation/plot_saved_paper_results.py
```

The following commands train models and should only be run when a new experiment is intended:

```powershell
python ML_Implementation/run_multiseed_forecasting_experiment.py
python ML_Implementation/run_multiseed_feature_ablation_experiment.py
```

Useful read-only checks and the automated test suite:

```powershell
python src/saved_experiment_release_readiness_audit.py
python src/multiseed_forecast_results_audit.py
python src/multiseed_feature_ablation_audit.py
python src/duplicate_normalized_key_audit.py
python src/grouped_feature_missingness_audit.py
python src/grouped_predictor_source_audit.py
python src/forecast_feature_availability_audit.py
python -m unittest discover -s tests
python -m compileall -q src ML_Implementation tests
```

Experiment runners use versioned result folders and refuse to overwrite existing outputs. Preserve the target backup, source datasets, locked test keys, and saved results. See `FILE_RENAMING_RECORD.md` for the source/notebook path changes and the historical paths intentionally retained.
