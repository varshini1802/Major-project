# Final project closure and handoff

## Verified project state

- Frozen target boundaries are 107, 237, 362 and 563, estimated from 2001–2010 training observations. The target is an ordered project-defined adaptation, not a verified reproduction of another paper's label procedure.
- Temporal periods are train 2001–2010, validation 2011–2012, and locked test 2013–2014. The same-year random-split notebook is a separate observed-year classification experiment.
- Nine exact IPC source columns overlap between grouped predictors and target construction. On 9,552 valid-target rows with at least one observed overlap source, these columns sum to 3,228,843 counts versus 4,215,620 total target-burden counts (76.592%). This is descriptive coverage only, not deterministic reconstruction. Potential indirect category/accounting overlap remains unverified.
- Strict sequence construction accepts 7,161 baseline and 5,842 enhanced sequences; paired model comparisons use the 5,842 shared keys (3,847 train, 1,324 validation, 671 test). Histories use exact district-keyed t−3, t−2, t−1 records. Missing histories and incomplete features are rejected, never fabricated.
- Seven duplicate district-year key groups (14 rows) in five cleaned files are excluded from core: five have identical offense payloads with differing pipeline/provenance IDs; two Railway groups conflict. Original Railway subunit identities are unavailable.
- Raw-stage-to-final IPC values reconstruct across the available normalized stages with zero mismatches after documented alias/canonicalization and row-filtering rules. Original upstream raw files are absent, so end-to-end source provenance is not established. The 2014 source-heading crosswalk and `COUNTERFIETING`/`IPC_COUNTERFEITING` mapping remain unresolved.
- Saved v4 metrics were recomputed and all eight rows matched. The five-seed paired comparisons and all 50 one-feature ablation fits are complete and integrity-checked. No model was retrained during this final manuscript and closure pass.
- Structural temporal and preprocessing checks pass. Annual source release/publication dates and an operational forecast issue date are absent, so real-world forecast-time availability is unverified.

## Target, features, and missingness

The six grouped formulas match stored values with zero mismatches. They use row-wise sums with `min_count=1`, preserving all-missing rows as missing while summing observed components in partial rows. Observed zero is distinct from missing. The data do not encode why sources are missing, so partial sums are not interpreted as complete counts. There are 225 missing target rows; 2,469 target observations lack at least one required historical year; enhanced histories reject another 1,320 incomplete/nonfinite feature histories. These are distinct data-quality conditions.

The source checklist and complete requirement mapping are in `original_correction_checklist_verification.md`; its available source `correction(2).md` is copied byte-for-byte to `source_correction_checklist_correction2_verbatim.md`. The earlier unsuffixed `correction.md` was not found, so exact version identity cannot be confirmed.

## Paper status

The new draft at `conference_paper_draft.md` now includes keywords, class distributions, target-source overlap, missingness distinctions, per-class results, a validation-selected v4 confusion matrix, and figures generated from saved results. Its exact numerical and reference evidence is inventoried in `conference_paper_audit.md`. It remains a draft requiring author, venue and final literature review; no original manuscript or conference template was located.

## Remaining limitations

1. Obtain NCRB category crosswalk/accounting rules for indirect overlap and the unresolved year-varying fields.
2. Obtain original source files, provenance and Railway subunit identifiers to establish upstream lineage and resolve conflicts.
3. Obtain administrative-boundary/version metadata and source-specific explanations for missing values.
4. Establish source release timing and a concrete forecast origin before claiming operational availability.
5. Assess dependence-aware uncertainty on more future periods; seed variability alone is not sampling uncertainty.

## Verification commands

This closure pass ran the manuscript-figure generator, project unit tests, compilation checks and relevant saved-output integrity audits. It did not run any training runner. Exact command outcomes are appended to `conference_paper_audit.md` after final verification.

## Files added or updated in this closure pass

- Project guidance: `README.md`.
- Audit/report generation: `src/final_corrections_v5_audit.py`, `src/multiseed_v5_audit.py`, and `ML_Implementation/plot_final_paper_figures.py`.
- Manuscript and review: `outputs/reports/conference_paper_draft.md`, `conference_paper_audit.md`, `original_correction_checklist_verification.md`, and byte-preserved `source_correction_checklist_correction2_verbatim.md`.
- Closure/data notes: `outputs/reports/final_project_closure.md`, `data_dictionary_and_schema_crosswalk.md`, `correction_checklist_v5.csv`, and `final_correction_checklist_status.md`.
- Figures: `outputs/figures/final_paper/five_seed_macro_f1.png`, `five_seed_ablation_macro_f1.png`, and `selected_v4_enhanced_lstm_confusion_matrix.png`.
- Saved-output audits were regenerated under `outputs/audits/final_corrections_v5/`, `ML_Implementation/corrected_temporal_results/corrected_v5_feature_ablation_multiseed/`, and `outputs/audits/final_corrections_v5/` (duplicate-key findings). These are audit/summary artifacts only; no model or data-generation output was retrained or rewritten.
