# Project File Guide

This guide explains the files that are most useful when presenting the project. In simple terms, the pipeline integrates the crime tables, constructs a documented burden class, builds predictors available before each forecast year, and evaluates saved models on a locked chronological test period.

## Main folders

| Folder | What it contains |
|---|---|
| `data/raw/` | Source-stage tables. Original upstream downloads and their retrieval dates are not all available. |
| `data/processed/` | Integrated district-year tables, grouped predictors, and enhanced features. |
| `src/` | Reusable data, target, feature-engineering, and audit code. |
| `ML_Implementation/` | Forecast input construction, model experiments, saved predictions, and metric calculations. |
| `notebooks/` | The integration and exploration workflow and the separate observed-year classification experiment. |
| `tests/` | Automated checks of target rules, keyed lag/sequence construction, feature availability, and reporting. |
| `outputs/audits/` | Machine-readable and written evidence about data integrity, target construction, leakage, reproducibility, and release readiness. |
| `outputs/reports/` | Research and project reports, including the paper draft. |
| `outputs/tables/` | Generated audit and summary tables. |
| `outputs/figures/` | Figures generated from the saved experiment outputs. |

## Project File Guide

| File | What it does | Input and output | How to run |
|---|---|---|---|
| `src/data_loading.py` | Loads the integrated crime data and checks required columns. | Reads processed CSV data; returns data and schema summaries. | Imported by project scripts. |
| `src/data_cleaning.py` | Provides reusable safe cleaning and data-quality report functions. | Takes data frames; returns cleaned data or report tables. | Imported by pipeline code. |
| `src/target_creation.py` | Defines the frozen project target and training-fitted boundaries. | Uses `VIOLENT_CRIME_BURDEN`; produces `TARGET_VIOLENT_LEVEL`. | Imported by target/data scripts. |
| `src/target_creation_backup.py` | Preserved target-generation backup for provenance. | Historical backup; do not edit as part of ordinary runs. | Do not run to refresh current data. |
| `src/target_definition.py` | Audits candidate target definitions and target summaries. | Reads integrated crime data; writes target analysis tables. | `python src/target_definition.py` when target analysis is needed. |
| `src/target_audit.py` | Checks whether the integrated source already contains target-like columns. | Reads integrated data; reports possible target columns. | `python src/target_audit.py`. |
| `src/feature_engineering.py` | Creates the extra predictors from grouped crime data. | Reads `district_grouped_reduced.csv`; writes `district_enhanced_features.csv`. | `python src/feature_engineering.py`. |
| `src/refresh_downstream_targets.py` | Refreshes labels by district-year key using the frozen canonical target. | Reads processed target tables and updates only downstream label fields. | Run only when a deliberate dataset refresh is required. |
| `src/leakage_audit.py` | Checks target and predictor overlap evidence. | Reads source mapping and target artifacts; writes audit tables. | `python src/leakage_audit.py`. |
| `src/grouped_predictor_source_audit.py` | Validates source formulas for the six grouped crime predictors. | Reads provenance mappings and integrated/grouped data; writes a formula audit. | `python src/grouped_predictor_source_audit.py`. |
| `src/forecast_feature_availability_audit.py` | Checks which historical source years appear in saved forecasting features. | Reads forecast matrices and source records; writes availability results. | `python src/forecast_feature_availability_audit.py`. |
| `src/source_stage_reconciliation_audit.py` | Compares IPC values across available processing stages. | Reads source-stage and integrated tables; writes a discrepancy audit. | `python src/source_stage_reconciliation_audit.py`. |
| `src/duplicate_normalized_key_audit.py` | Explains normalized district-year duplicate groups. | Reads duplicate-key evidence; writes a duplicate findings report. | `python src/duplicate_normalized_key_audit.py`. |
| `src/grouped_feature_missingness_audit.py` | Checks missing-value behavior in grouped formulas. | Reads grouped and integrated data; writes reconciliation tables. | `python src/grouped_feature_missingness_audit.py`. |
| `src/grouped_dataset_integrity_audit.py` | Checks the reduced grouped dataset's shape, columns, duplicate keys, missingness, and target distribution. | Reads `district_grouped_reduced.csv`; reports validation results. | `python src/grouped_dataset_integrity_audit.py`. |
| `src/multiseed_forecast_results_audit.py` | Verifies saved multi-seed predictions against the locked cohort and configuration. | Reads saved runs; writes integrity and paired metric summaries. | `python src/multiseed_forecast_results_audit.py`. |
| `src/multiseed_feature_ablation_audit.py` | Checks the completed multi-seed feature ablation outputs. | Reads saved ablation predictions and configurations; writes summary checks. | `python src/multiseed_feature_ablation_audit.py`. |
| `src/saved_experiment_release_readiness_audit.py` | Recomputes metrics and checks keys, labels, cohorts, and protected hashes for saved forecasts. | Reads predictions, probabilities, reports, and canonical target; writes `outputs/audits/final_release_readiness/`. | `python src/saved_experiment_release_readiness_audit.py`. |
| `src/technical_completion_audit.py` | Builds a read-only technical evidence package from project artifacts. | Reads saved evidence; writes `outputs/audits/final_technical_completion/`. | `python src/technical_completion_audit.py`. |
| `src/evidence_closure_audit.py` | Builds evidence ledgers and checks protected-file hashes. | Reads source/evidence files; writes `outputs/audits/final_evidence_closure/`. | `python src/evidence_closure_audit.py`. |
| `ML_Implementation/forecast_feature_contract.py` | Defines permitted forecast predictors and keyed lag construction rules. | Reads district-year records; returns validated forecast matrices. | Imported by experiment builders. |
| `ML_Implementation/sequence_generation.py` | Makes strict district-keyed three-year histories for prediction year *t*. | Uses consecutive years *t−3* through *t−1*; returns sequences and metadata. | Imported by sequence builder. |
| `ML_Implementation/sequence_cohort.py` | Selects the same eligible district-year keys for baseline and enhanced sequence comparison. | Reads baseline/enhanced eligibility; returns the common cohort. | Imported by experiment scripts. |
| `ML_Implementation/prepare_forecasting_inputs.py` | Builds saved tabular forecast matrices and shared sequence inputs. | Reads processed district-year features; writes versioned inputs. | `python ML_Implementation/prepare_forecasting_inputs.py`. |
| `ML_Implementation/generate_forecasting_sequences.py` | Creates sequence arrays and metadata without training models. | Reads keyed forecast history; writes sequence inputs under the result directory. | `python ML_Implementation/generate_forecasting_sequences.py`. |
| `ML_Implementation/tabular_forecast_evaluation.py` | Evaluates models on the saved tabular forecast matrices. | Reads versioned matrices and training configuration; writes predictions and metrics. | Run only for an intended evaluation rerun. |
| `ML_Implementation/common_cohort_sequence_evaluation.py` | Compares sequence models on identical eligible district-year keys. | Reads shared sequences; writes predictions and metrics. | Run only for an intended evaluation rerun. |
| `ML_Implementation/controlled_sequence_forecast_evaluation.py` | Runs paired GRU/LSTM forecasts with validation selection and a locked test cohort. | Reads strict sequence inputs; writes model outputs and evaluation tables. | Run only for an intended evaluation rerun. |
| `ML_Implementation/run_multiseed_forecasting_experiment.py` | Runs the declared multi-seed robustness experiment. | Uses frozen inputs and seeds; writes a new versioned result set. | Trains models; do not run just to regenerate reports. |
| `ML_Implementation/run_multiseed_feature_ablation_experiment.py` | Runs the multi-seed one-feature-at-a-time ablations. | Uses saved forecast inputs and fixed seeds; writes ablation results. | Trains models; do not run just to regenerate reports. |
| `ML_Implementation/plot_saved_paper_results.py` | Builds manuscript figures from saved experiment result tables. | Reads saved CSV metrics and writes figures; does not fit models. | `python ML_Implementation/plot_saved_paper_results.py`. |
| `notebooks/01_source_data_integration.ipynb` | Documents integration of source files into common district-year tables. | Reads raw/source-stage tables; produces integrated data. | Open and run cells in order when reproducing integration. |
| `notebooks/02_crime_data_exploration.ipynb` | Explores the integrated crime data and its distributions. | Reads integrated data; displays analysis and figures. | Open in Jupyter. |
| `notebooks/observed_year_classification_models.ipynb` | Runs the same-year observed-burden classification analysis. | Uses crime data from the target year; this is not a future-year forecast. | Open in Jupyter; interpret separately from forecast scores. |
| `tests/test_target_creation.py` | Checks target boundaries, classes, and row-order behavior. | Uses controlled test cases. | `python -m unittest discover -s tests`. |
| `tests/test_forecast_feature_contract.py` | Checks keyed feature construction and forecast-time eligibility. | Uses controlled data and feature contracts. | Included in the test suite command above. |
| `tests/test_sequence_generation.py` | Checks sequence continuity, district boundaries, and target alignment. | Uses controlled district-year examples. | Included in the test suite command above. |

## Similar but distinct scripts

The closure, technical-completion, evidence-closure, and release-readiness audits overlap in reading hashes and evidence, but answer different questions: source/code integrity, assembly of technical evidence, provenance ledgers, and saved forecast metric/cohort reconciliation. They write separate evidence packages and are retained separately. The multi-seed model audit and the feature-ablation audit inspect different saved experiments and are also complementary.

Three old temporal runners remain explicitly retired: `ML_Implementation/combined_temporal_model_comparison.py`, `ML_Implementation/temporal_deep_learning_experiment.py`, and `ML_Implementation/temporal_leakage_free_experiment.py`. Their saved historical outputs are not current forecast results. The empty `src/analysis.py` and `src/visualization.py` modules are retained as existing placeholders; they currently contain no implementation.

Experiment result folders include version identifiers to preserve run provenance. They should not be renamed as part of source-file naming cleanup.
