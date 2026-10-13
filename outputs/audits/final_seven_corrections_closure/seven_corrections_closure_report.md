# Final closure of seven partially resolved corrections

## Outcome

**Overall status: PARTIALLY_RESOLVED; external evidence remains required.** Repository-level formulas, temporal mechanics, saved prediction integrity, and experiment configuration were verified from retained code and audit reports. No new material defect was identified in the audited corrected forecasting artifacts. This is not a finding of “no leakage”: direct component overlap remains in historical predictors, indirect NCRB counting semantics are unresolved, and historical source release/access at a defined forecast origin is unverified.

No model was retrained. No dataset, target threshold/label, locked test key, saved prediction, official result, or paper draft was modified. Existing working-tree changes were preserved.

## Correction 1 — target/predictor overlap

Nine IPC columns are directly shared across five grouped predictors: homicide (`IPC_ATTEMPT_TO_MURDER`, `IPC_CULPABLE_HOMICIDE_NOT_AMOUNTING_TO_MURDER`, `IPC_MURDER`); sexual violence (`IPC_RAPE`); kidnapping (`IPC_KIDNAPPING_ABDUCTION`); robbery/dacoity (`IPC_DACOITY`, `IPC_ROBBERY`); and public violence (`IPC_ARSON`, `IPC_RIOTS`). No direct shared input was mapped for property crime; that does not rule out indirect overlap. The **76.592364%** is `3,228,843 / 4,215,620`, the sum of the nine shared source counts divided by total violent-crime burden. The positive-burden mask uses 9,552 of 9,631 valid rows; 79 omitted zero-burden rows have observed zero shared contribution, so the all-valid-row ratio is unchanged. It is a count-contribution ratio, not event duplication or exact label reconstruction.

Corrected forecast matrices use shared-source groups historically at t−1 or t−3…t−1. The relationship is not eliminated. The same-year random-split experiment is observed-year classification, not future forecasting.

## Correction 4 — engineered features

The five exact formulas, missing rules, forecast-matrix presence, and availability limits are in `engineered_feature_availability_final.csv`. `TOTAL_GROUPED_CRIME` sums six groups with `min_count=1`; `ACTIVE_CRIME_GROUPS` counts positive groups; `CRIME_DIVERSITY_INDEX` uses Shannon entropy (individual missing group contributes zero, zero total gives 0, all-missing remains NA); `PREV_YEAR_TOTAL_CRIME` is an exact same-state/unit year-minus-one lookup; `YOY_TOTAL_CRIME_CHANGE` is current minus prior absolute count, not a percentage. Current-year TOTAL, ACTIVE, DIVERSITY, and YOY values are unsuitable for advance prediction of that same year. Lagged values are structurally historical, but publication/access dates are unknown.

## Correction 7 — temporal alignment

The prior-year join uses `STATE`, `UNIT_NAME`, and exact `YEAR-1`; no row-order shift or gap bridging occurs. Missing prior records remain missing. Sequence code requires same normalized state/unit records for t−3, t−2, t−1 and rejects missing keys or incomplete/nonfinite features without padding or imputation. Mechanics are verified within available name keys; official geographic continuity is not.

## Correction 8 — integration/geography

Existing audits report 2014 IPC cleaned rows 838 and core rows 729, with 73 special-unit and 36 aggregate rows excluded. Seven duplicate normalized district-year groups (14 rows) were found; five have identical offence payloads and two 2010 Jammu & Kashmir `RAILWAYS` groups conflict. Railway groups remain excluded under the existing rule. Pipeline row identifiers are not verified NCRB identifiers. Local processed-stage reconciliation does not establish original download provenance, and matching names do not establish boundary continuity.

## Correction 9 — missingness

Observed zeros are distinct from missing. Each grouped column has 225 missing of 9,856 rows (2.282873%). Grouping uses `min_count=1`: partial sums remain partial sums and all-missing groups stay NA. Target audits report 9,631 valid and 225 excluded rows. A missing prior year remains NA. YOY is a difference, so it has no zero denominator. Existing audits report source/year missingness, but causes cannot be inferred.

## Correction 12 — leakage audit

The 12-part machine-readable audit is `consolidated_leakage_findings.csv`. Findings include `NO_DIRECT_TARGET_COLUMN_LEAKAGE_FOUND` within approved corrected forecast matrices, `DIRECT_COMPONENT_OVERLAP_PRESENT`, `TEMPORAL_FEATURE_ALIGNMENT_VERIFIED_WITHIN_AVAILABLE_KEYS`, no future feature years found in audited inputs, `INDIRECT_SEMANTIC_OVERLAP_UNRESOLVED`, and `HISTORICAL_OPERATIONAL_AVAILABILITY_UNVERIFIED`. Existing release audit reconciles 78 prediction artifacts with zero canonical key/label failures. The later controlled sequence cohort has 671 rows; legacy v1/v2 sequence baseline/enhanced cohorts (1,276 vs 671) are unequal and superseded.

## Correction 13 — reproducibility

`reproducibility_inventory.json` records environment, thresholds, split, feature lists, retained run configs, seeds, artifact paths and pipeline row counts. Saved predictions and metrics are reconciled. Full raw-to-result reproduction is not established because original downloads and acquisition provenance are incomplete.

## Safe submission claims

State the frozen thresholds (107, 237, 362, 563), train/validation/test periods (2001–2010, 2011–2012, 2013–2014), and code-level input timing (tabular t−1; sequence t−3 through t−1). State that nine source columns overlap and describe 76.592364% only as a burden-count contribution ratio. Label same-year classification separately. Qualify operational availability, indirect semantics, geographic continuity, source provenance, missingness causes, and end-to-end reproducibility. Do not claim leakage is fully eliminated or all corrections externally resolved.

## Validation and preservation

Exact test/compile outcomes and preservation checks are recorded in `validation_report.txt`. New outputs are confined to this directory.
