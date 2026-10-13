# File Renaming Plan and Record

This plan targets source and notebook names that obscure their responsibility. It preserves experiment outputs, datasets, the target backup, and historical run folders because those names and paths are part of the saved research record.

## Proposed renaming plan

| Current path | Proposed path | Actual purpose | Reason for renaming | Required reference updates |
|---|---|---|---|---|
| `src/audit_final_remaining_corrections.py` | `src/data_quality_and_claims_audit.py` | Checks data-quality and paper-claim evidence | Removes vague chronology labels | Tests, README, report commands |
| `src/duplicate_key_v5_audit.py` | `src/duplicate_normalized_key_audit.py` | Explains duplicate normalized district-year keys | Names the checked condition; removes run version | README, audit commands |
| `src/final_closure_audit.py` | `src/project_closure_audit.py` | Builds the code/data closure package | Removes generic “final” label | README, audit commands |
| `src/final_corrections_v5_audit.py` | `src/corrected_results_integrity_audit.py` | Recomputes correction and saved-result checks | States the responsibility | README, reports and commands |
| `src/final_evidence_closure.py` | `src/evidence_closure_audit.py` | Builds evidence and provenance ledgers | Clarifies that it verifies and packages evidence | Tests, README, commands |
| `src/final_release_readiness_audit.py` | `src/saved_experiment_release_readiness_audit.py` | Reconciles saved forecast results for sign-off | Makes its scope explicit | README and release audit command |
| `src/final_remaining_corrections_audit.py` | `src/data_quality_pipeline_audit.py` | Checks missingness, pipeline counts and claims | Removes temporary chronology wording | Tests, README, commands |
| `src/final_technical_completion_audit.py` | `src/technical_completion_audit.py` | Builds a technical evidence package | Removes redundant “final” wording | README, commands |
| `src/forecast_availability_audit.py` | `src/forecast_feature_availability_audit.py` | Checks saved forecast inputs against their source years | Names the checked object | README, commands |
| `src/missingness_v5_reconciliation.py` | `src/grouped_feature_missingness_audit.py` | Reconciles formula missingness with grouped values | Removes the historical run label | README, commands |
| `src/multiseed_v5_ablation_audit.py` | `src/multiseed_feature_ablation_audit.py` | Validates saved feature-ablation runs | States the check and subject | README, commands |
| `src/multiseed_v5_audit.py` | `src/multiseed_forecast_results_audit.py` | Verifies multi-seed forecasts against the locked cohort | Removes unexplained version label | README, commands |
| `src/patch_notebook_forecasting_cells.py` | `src/update_forecasting_notebook_sequences.py` | Applies keyed-history fixes to notebook sequence cells | Describes its action and scope | README and notebook-maintenance instructions |
| `src/predictor_source_audit.py` | `src/grouped_predictor_source_audit.py` | Recovers and validates grouped predictor formulas | Names the audited predictors | README, commands |
| `src/project_folder_inventory.py` | `src/project_file_inventory.py` | Creates file inventory and duplicate-content report | Uses a concise, direct name | README and command references |
| `src/raw_stage_reconciliation_audit.py` | `src/source_stage_reconciliation_audit.py` | Traces differences across integration stages | Clarifies the stages being compared | README, commands, reports |
| `src/crime_grouping.py` | `src/grouped_dataset_integrity_audit.py` | Checks reduced grouped-data shape, columns, duplicate keys, missingness and target distribution | Its former name implied grouping construction, while it is a dataset validation script | Python evidence references and the project file guide |
| `src/update_notebook_research_claims.py` | `src/update_observed_year_notebook_labels.py` | Corrects stale target descriptions in the observed-year notebook | Identifies the notebook and change | README and notebook-maintenance instructions |
| `ML_Implementation/audit_corrected_v3_inputs.py` | `ML_Implementation/forecast_input_integrity_audit.py` | Performs fail-closed structural checks on forecast matrices | Removes run version and states purpose | Python imports, README, commands |
| `ML_Implementation/audit_legacy_sequences.py` | `ML_Implementation/legacy_sequence_integrity_audit.py` | Explains incomplete legacy histories and labels | Identifies the audited artifact | README and commands |
| `ML_Implementation/corrected_sequence_experiment.py` | `ML_Implementation/three_year_sequence_forecasting_experiment.py` | Trains/evaluates the earlier corrected three-year sequence run | Names the horizon and task | README and Python paths |
| `ML_Implementation/corrected_temporal_experiment.py` | `ML_Implementation/tabular_forecasting_experiment.py` | Runs keyed one-year-lag tabular forecasting | Names the method and task | README, commands, metadata-generating code |
| `ML_Implementation/corrected_v3_common_sequence_experiment.py` | `ML_Implementation/common_cohort_sequence_evaluation.py` | Evaluates baseline/enhanced sequences on common keys | Removes run version; identifies cohort design | Imports, README, commands |
| `ML_Implementation/corrected_v3_tabular_experiment.py` | `ML_Implementation/tabular_forecast_evaluation.py` | Evaluates saved tabular forecast matrices | Removes run version | Imports, README, commands |
| `ML_Implementation/corrected_v4_controlled_sequence_experiment.py` | `ML_Implementation/controlled_sequence_forecast_evaluation.py` | Runs paired GRU/LSTM sequence evaluation with locked test | States design without run version | Tests, imports, README, commands |
| `ML_Implementation/corrected_v4_engineered_feature_ablation.py` | `ML_Implementation/engineered_feature_ablation_experiment.py` | Runs one-at-a-time engineered-feature ablations | Removes run version | Imports, README, commands |
| `ML_Implementation/generate_corrected_sequences.py` | `ML_Implementation/generate_forecasting_sequences.py` | Builds strict three-year sequence inputs and metadata | Names output and task | Metadata string, README, commands |
| `ML_Implementation/prepare_corrected_v3_inputs.py` | `ML_Implementation/prepare_forecasting_inputs.py` | Builds forecast matrices and common sequence cohort | Removes run version | Imports, README, commands |
| `ML_Implementation/run_corrected_v5_multiseed.py` | `ML_Implementation/run_multiseed_forecasting_experiment.py` | Runs predeclared multi-seed model comparisons | Removes version from source filename | README and run instructions |
| `ML_Implementation/run_corrected_v5_multiseed_ablations.py` | `ML_Implementation/run_multiseed_feature_ablation_experiment.py` | Runs multi-seed one-feature ablations | Removes version and clarifies purpose | README and run instructions |
| `ML_Implementation/supplemental_v3_auc_metrics.py` | `ML_Implementation/recompute_saved_probability_metrics.py` | Recomputes AUC/AP from saved probabilities | Names operation and input | README and commands |
| `ML_Implementation/plot_final_correction_results.py` | `ML_Implementation/plot_controlled_forecast_results.py` | Plots saved controlled forecast results | Removes temporary “final” label | README and commands |
| `ML_Implementation/plot_final_paper_figures.py` | `ML_Implementation/plot_saved_paper_results.py` | Makes manuscript figures from saved outputs | Identifies output source and purpose | README, reports and commands |
| `notebooks/01_Common_Schema_Integration.ipynb` | `notebooks/01_source_data_integration.ipynb` | Integrates source datasets into the common schema | Lowercase descriptive name; sequence number marks workflow order | README and notebook path references |
| `notebooks/02_India_Crime_Final_EDA.ipynb` | `notebooks/02_crime_data_exploration.ipynb` | Explores integrated crime data | Removes “Final” and title-case naming | README and notebook path references |
| `notebooks/basepaper_prediction_models.ipynb` | `notebooks/observed_year_classification_models.ipynb` | Performs same-year observed-burden classification | Makes task distinction visible | README and notebook path references |

## Names and paths to retain

| Current name/path | Reason to retain |
|---|---|
| `ML_Implementation/` | Established directory referenced throughout code, notebooks, test import setup, and saved run metadata. Renaming it would alter many archived paths without improving the organization enough to justify that churn. |
| `ML_Implementation/corrected_temporal_results/corrected_v1` through `corrected_v5_*` | These are immutable historical experiment identifiers. Renaming them would alter saved-result paths and run provenance. |
| `src/target_creation.py`, `src/target_creation_backup.py`, `src/target_definition.py` | These names map to distinct target-construction and target-audit responsibilities; the backup is explicitly preserved. |
| `src/feature_engineering.py`, `src/data_cleaning.py`, `src/data_loading.py`, `src/leakage_audit.py`, `src/target_audit.py` | Already clear, reusable module names. |
| `tests/test_*.py` | Follows the standard automated-test naming convention. |
| Data files, model files, generated figures and reports | Their names and locations are part of the saved analysis and are not necessary to rename to clarify source-code responsibilities. |
| `ML_Implementation/combined_temporal_model_comparison.py`, `ML_Implementation/temporal_deep_learning_experiment.py`, `ML_Implementation/temporal_leakage_free_experiment.py` | These are explicitly retired historical runners. Keep their names and contents for provenance; the README will label them retired. |

## Implemented record

The renames above were applied after capturing the pre-change Git status and project inventory. Active references were updated in `README.md`, `src/corrected_results_integrity_audit.py`, `src/project_closure_audit.py`, `src/technical_completion_audit.py`, `src/evidence_closure_audit.py`, `src/saved_experiment_release_readiness_audit.py`, `ML_Implementation/temporal_leakage_free_experiment.py`, `ML_Implementation/temporal_deep_learning_experiment.py`, `ML_Implementation/combined_temporal_model_comparison.py`, `ML_Implementation/generate_forecasting_sequences.py`, `ML_Implementation/prepare_forecasting_inputs.py`, `ML_Implementation/engineered_feature_ablation_experiment.py`, `ML_Implementation/run_multiseed_forecasting_experiment.py`, `tests/test_final_evidence_closure.py`, and `tests/test_final_correction_reporting.py`. README commands were checked against the renamed files. Notebook JSON was validated after the notebook renames.

The new `outputs/audits/final_release_readiness/` package was regenerated through the renamed release-readiness audit script. Existing experiment result directories, model/data files, paper draft, and prior audit snapshots were retained. As historical provenance, older saved run configuration files and audit/report snapshots still contain the source paths that were current when those runs were produced; these records were not rewritten. Examples include `ML_Implementation/corrected_temporal_results/corrected_v5_multiseed/runner_config.json`, `outputs/audits/final_evidence_closure/reproducibility_manifest.json`, and `outputs/reports/conference_paper_audit.md`. This record is the canonical crosswalk from those archived names to current source names.

Pre-rename state is recorded in `outputs/audits/project_renaming_initial_git_status.txt`, the path inventory is in `outputs/audits/project_file_inventory_before.txt`, and hashes for the 198 previously protected files are in `outputs/audits/project_renaming_protected_hashes_before.json`.
