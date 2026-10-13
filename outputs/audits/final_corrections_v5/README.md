# Final corrections v5 audit

This is a non-mutating audit of current inputs and saved corrected-v4 results.
No model suite was retrained and no protected v2/v3/v4 result, target backup, or raw input was overwritten.

## Recomputed metric check

Saved-v4 result rows checked: 8; metric rows agreeing within 1e-8: 8.
Probability columns are class ordered 1..5 and validated for finite values and row sums. Macro mean one-vs-rest AP is an average of class-wise average precision, not trapezoidal PR-AUC.

## Evidence limits

The available correction source correction(2).md is preserved at outputs/reports/source_correction_checklist_correction2_verbatim.md and mapped requirement-by-requirement in outputs/reports/original_correction_checklist_verification.md. The earlier unsuffixed correction.md was not found, so exact version identity is unverified.
Completed multi-seed runs: 11, 23, 37, 53, 71; see ML_Implementation/corrected_temporal_results/corrected_v5_multiseed/run_status.csv. The original v4 seed-42 results remain unchanged.
The five-seed one-feature/model ablation runner records 50/50 expected fits; see its run_status.csv and paired_difference_summary.csv. Seed spread is descriptive and not sampling uncertainty.
The local repository lacks original upstream raw copies, verified administrative boundary provenance, annual publication/release dates, and documented missingness causes.
Component schema coverage differs by year. In particular, the normalized 2001–2013 IPC files contain the source spelling COUNTERFIETING while the final canonical IPC_COUNTERFEITING field is empty for those years; alternative 2014 headings for attempt-to-murder/burglary were not equated without an official crosswalk. See ipc_group_source_schema_discrepancies.csv.
Indirect IPC hierarchy/overlapping offence semantics remain unresolved unless explicitly supported by local documentation.
No original conference manuscript or template was found. A new project draft was created at outputs/reports/conference_paper_draft.md; it requires author and venue review.

## Outputs

See the CSV/JSON files in this directory: processed dataset manifest, key audit, source lineage, overlap mapping, component missingness, v4 metric recomputation, per-class metrics, confusion matrices, v4 ablation review, reproducibility manifest, and correction status.
