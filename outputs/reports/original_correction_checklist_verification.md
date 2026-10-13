# Original correction checklist verification

## Source identity and handling

The available source is `C:\Users\Dell\Downloads\correction(2).md` (24,514 Unicode characters, SHA-256 recorded below). Its heading is **“NCRB Crime Classification Research Paper: Required Corrections”** and it contains requirements 1–13, paper requirements 14–20, and a 17-item final checklist. It is the only `*correction*.md` file found in Downloads. The earlier-mentioned unsuffixed `correction.md` was not present, so identity with that earlier file cannot be proven. No repository or Git-history copy was found. Source wording and numbering were preserved in the Downloads file and are summarized here by original section number; this report does not silently reconstruct missing versions.

The source is preserved outside the repository at its original path and copied byte-for-byte to [`source_correction_checklist_correction2_verbatim.md`](source_correction_checklist_correction2_verbatim.md). SHA-256: `4929d4110e8cea5cb32adb9cf64b7b8aa60a25e42fde521716f354512851d9ce` (the source and copy match byte-for-byte).

Status meanings: **COMPLETE** means evidence supports the specified correction for the saved experiment; **PARTIAL** means implementation/evidence exists but a material provenance or availability limit remains; **UNRESOLVED** means the needed source evidence is unavailable. Status describes the project evidence, not paper-submission readiness.

## Requirement-by-requirement disposition

| Original no. and requirement | Status | Disposition and evidence |
|---|---|---|
| 1. Critical: Resolve Target–Predictor Overlap | PARTIAL | The experiment is explicitly future-year burden classification with t−1 inputs; all target sources were traced; nine direct IPC overlaps were found. The audit sums observed direct-source counts over valid rows (3,228,843) and divides by target burden summed over all 9,631 valid-target rows (4,215,620), a descriptive ratio of 76.592%. 9,552 rows have at least one observed overlap value; 79 have none. This is not proof of deterministic reconstruction. Potential indirect semantic/counting overlap cannot be established without NCRB crosswalk and accounting documentation. See `direct_indirect_source_overlap.csv`, `direct_overlap_burden_summary.csv`, and `ipc_semantic_relationship_review.md`. |
| 2. Correct the Quantile-Based Target Construction | COMPLETE | Ties remain together under frozen right-closed boundaries 107, 237, 362, 563, estimated from 2001–2010 only. Saved target checks and class distributions are in `corrected_target_saved_data_verification.csv`, `target_thresholds.csv`, and `final_target_distribution.csv`. Downstream labels match the canonical target on district-year keys. No retraining was performed for this closure. |
| 3. Clarify What the Five Classes Represent | COMPLETE | The manuscript calls them ordered project-defined violent-crime burden levels, defines all five, and states that the target is an adaptation rather than a verified reproduction of the original paper. See `src/target_creation.py` and the draft's target section. |
| 4. Reconsider the Five Engineered Features | COMPLETE | Formulas, source years and availability are documented in the feature-source/availability audits. Five-seed paired model comparisons and all 50 one-feature ablation fits are saved and audited. Effects are descriptive, not causal. See `ablation_report.md`, `corrected_v5_multiseed/`, and `corrected_v5_feature_ablation_multiseed/`. |
| 5. Replace Random Evaluation with Chronological Evaluation When Forecasting | COMPLETE, operational availability PARTIAL | Chronological train/validation/test periods are 2001–2010 / 2011–2012 / 2013–2014. The random-split notebook is classified as observed-year classification. Source publication dates and a real forecast issue date are unavailable; therefore actual operational availability remains unverified. |
| 6. Correct the MLP Validation Preprocessing | COMPLETE | The observed-year notebook's validation split precedes scaler fitting; forecasting audits verify training-only fitting/selection and test lock for the corrected pipelines. Evidence: `forecast_feature_availability_audit.csv`, `final_forecast_availability_audit.csv`, and leakage negative tests. |
| 7. Verify the Correctness of the Previous-Year Features | COMPLETE structurally; availability PARTIAL | Keyed joins verify same-district exact t−1 values; sequence tests verify same-district consecutive t−3…t−1 history and target-year alignment. Missing district-years are rejected. Release timing still lacks evidence. See `forecast_feature_availability_audit.csv`, `sequence_stage_counts.csv`, and tests. |
| 8. Validate Dataset Integration and District-Year Keys | PARTIAL | Available-stage reconciliation and key checks are documented. Seven duplicate key groups (14 rows) in five cleaned files are excluded; five are identical payloads with different provenance identifiers and two Railway groups conflict. Original upstream files, Railway subunit identities and boundary provenance are unavailable. See `cleaned_duplicate_key_findings.md`, `district_year_key_validation.csv`, and `raw_stage_reconciliation_audit.csv`. |
| 9. Review Missing-Value Handling | PARTIAL | Observed zero is distinguished from missing; group sums use `min_count=1`, preserving all-missing as missing and summing observed components for partial groups. There are 225 missing target/group rows; component missingness is often partial and its semantic cause is not encoded. Target missingness, feature missingness, partial aggregate and missing historical district-year are separately counted. See group missingness and sequence-stage reports. |
| 10. Validate the Meaning of the Classification Metrics | COMPLETE for saved experiments | All eight v4 rows were recomputed from saved predictions/probabilities and matched. Accuracy, macro/weighted F1, per-class metrics, confusion matrices, multiclass macro OvR ROC-AUC/AP, majority baseline and partition supports are documented. The v4 test cohort has 671 sequences; supports are 113/76/105/145/232. Results from observed-year classification are not compared as forecasting results. |
| 11. Reassess the Need for LSTM and GRU Models | COMPLETE structurally | Three-year same-district sequences are strict t−3, t−2, t−1 histories for a t target; incomplete histories/features are rejected, not fabricated. Baseline has 7,161 valid sequences before common-cohort restriction; enhanced has 5,842; model comparisons use the 5,842 common cohort, including 3,847 train, 1,324 validation, 671 test. See sequence counts and test-key integrity reports. |
| 12. Improve the Leakage Audit | PARTIAL | Source-level overlap, feature lineage, structural temporal eligibility, joins, preprocessing, negative tests and leakage limits are documented. Nine direct shared inputs are verified. NCRB event/category accounting and real release timing remain unknown, so complete semantic/operational leakage elimination is not claimed. |
| 13. Improve Reproducibility | PARTIAL | Python/package versions, seeds, configs, hashes, feature lists, split periods, checkpoints, saved predictions, counts and audit commands are recorded. Reproduction from original NCRB files is not possible from this repository because original upstream source files and some provenance are absent. |
| 14. Abstract | COMPLETE as a project draft | Updated with task, target, horizon, data period, split, saved corrected results and limitations. Manuscript still needs author and venue review. |
| 15. Introduction and Research Objectives | COMPLETE as a project draft | Distinguishes future-year burden forecasting from contemporaneous observed-year classification; states the question and limits. |
| 16. Dataset and Preprocessing | COMPLETE with explicit source limits | Includes unit, periods, target classes, missingness/zero distinction, grouping, duplicates and unavailable upstream provenance. The exact original-source transformations and year-specific unresolved mappings remain open. |
| 17. Methodology | COMPLETE for documented saved experiment | Describes integration evidence, groups, target, features, chronological partitions, preprocessing, selection, and test evaluation. It explicitly limits overlap and availability claims. |
| 18. Experimental Results | COMPLETE for saved results | Tables and generated figures are tied to saved corrected runs; selected v4 enhanced LSTM is chosen by validation macro-F1, with its confusion matrix and per-class metrics. The five-seed results are reported separately. No new training occurred. |
| 19. Discussion | COMPLETE as a project draft | Interprets next-year class performance and modest/mixed changes; notes class errors, target/provenance limits and dependence; makes no causal or significance claim. |
| 20. Limitations and Future Work | COMPLETE as a project draft | Lists reported-crime, source-schema, missingness, duplicate Railway, boundary, threshold, dependence and release-date limits that apply to the saved data/design. |

## Original final verification checklist

The following lines retain the source checklist wording. “Status” and evidence links are appended for each original item.

| Source checklist item | Status | Evidence / remaining limit |
|---|---|---|
| The target is clearly defined and reproducible. | COMPLETE | Frozen thresholds and target-function audit. |
| Identical burden values are not arbitrarily assigned different classes. | COMPLETE | Tie-safe target tests and saved-label verification. |
| The target-generation variables have been mapped against all model predictors. | PARTIAL | Full source inventory and nine direct overlaps; indirect event/category accounting unavailable. |
| The prediction task and prediction horizon are explicit. | COMPLETE | Future-year burden in t, inputs through t−1; three-year sequence lookback. |
| Every predictor is available at the intended prediction time. | PARTIAL | Structural source-year eligibility passes; actual annual release/issue timing is unverified. |
| Historical features are aligned to the correct district and year. | COMPLETE structurally | Exact keyed t−1 and t−3…t−1 audits/tests pass. |
| Dataset joins and district-year keys have been validated. | PARTIAL | Processed key checks pass; seven excluded duplicate groups and upstream provenance remain relevant. |
| Missing values and genuine zero counts are distinguished. | PARTIAL | Stored zeros remain observed; missing causes and some source-schema gaps cannot be determined. |
| Preprocessing is fitted on training data only. | COMPLETE | Saved-code/config audits and negative checks. |
| The final evaluation set is not used for model selection. | COMPLETE | Validation selection precedes locked 2013–2014 test scoring. |
| A chronological evaluation is used if forecasting is claimed. | COMPLETE | Corrected experiments use the specified temporal split. |
| LSTM and GRU sequences are constructed correctly if these models are retained. | COMPLETE structurally | Strict keyed consecutive histories; incomplete sequences rejected; key/target tests pass. |
| All models are evaluated using the same appropriate protocol. | COMPLETE within each comparison | Same locked 671 test keys in paired comparisons; observed-year notebook remains separate. |
| The results have been regenerated after correcting the methodology. | COMPLETE for saved corrected experiments | Corrected v4, five-seed v5, and 50 ablation fits are saved; this closure did not retrain them. |
| All tables and figures match the regenerated results. | COMPLETE after figure generation | Paper audit figure inputs are saved metric and confusion-matrix artifacts; see `conference_paper_audit.md`. |
| The abstract, methodology, discussion, and conclusions agree with the final experiments. | COMPLETE as a draft | Updated manuscript; author/venue review remains. |
| The code, configuration, and data-processing steps are documented sufficiently for independent reproduction. | PARTIAL | Saved pipeline and experiment artifacts exist; original upstream raw sources and provenance are unavailable. |

## Explicit unresolved items

1. The earlier unsuffixed `correction.md` cannot be matched byte-for-byte to the available `correction(2).md`.
2. Nine direct source overlaps do not quantify all possible indirect overlap; legal/statistical category semantics and double-counting rules require NCRB documentation.
3. `COUNTERFIETING` versus `IPC_COUNTERFEITING` (2001–2013), 2014 attempt-to-murder/burglary headings, and several year-varying schema absences have no authoritative crosswalk.
4. Original upstream raw files, Railway subunit identities, administrative boundary provenance, and semantic causes of missing values are unavailable.
5. Source-year eligibility is verified, but source publication/release dates and a forecast issue date are unavailable; operational availability is unverified.
6. This is a project-defined target adaptation. No inference is made that it reproduces the cited study's target.
7. Corrected model outputs are available and audited, but the suite has not been rerun during this paper/closure pass. Statistical uncertainty over districts/years was not estimated.

## Evidence locations

- Target and leakage audits: `outputs/audits/final_corrections_v5/`.
- Direct overlap burden detail: `outputs/tables/final_remaining_corrections/direct_overlap_burden_summary.csv`.
- Sequence cohort and rejection counts: `outputs/audits/final_corrections_v5/sequence_stage_counts.csv`.
- Saved corrected experiments: `ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/`, `corrected_v5_multiseed/`, and `corrected_v5_feature_ablation_multiseed/`.
- Manuscript claim-by-claim check: `outputs/reports/conference_paper_audit.md`.
- Verbatim available source checklist: `outputs/reports/source_correction_checklist_correction2_verbatim.md`.
