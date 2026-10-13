# Data dictionary and verified source crosswalk

## Canonical tables

The detailed processed-column inventory is `outputs/tables/final_data_dictionary.csv`. It includes column type, missing count/percentage and distinct-value count for the integrated project table. The forecast target is `TARGET_VIOLENT_LEVEL`; its source burden is `VIOLENT_CRIME_BURDEN`. Forecast input identity is district plus year (`STATE`, `UNIT_NAME`, `YEAR`), with model artifacts additionally retaining `INPUT_YEARS`.

## Grouped predictor formulas

Source columns, aggregation and missing behavior are recorded in `outputs/audits/final_corrections_v5/group_predictor_formula_recovery.csv` and were checked against `district_grouped_reduced.csv` with zero value mismatches:

- `GROUP_HOMICIDE`: attempt to murder + culpable homicide not amounting to murder + murder.
- `GROUP_SEXUAL_VIOLENCE`: assault on women with intent to outrage modesty + rape + sexual harassment + stalking + voyeurism.
- `GROUP_KIDNAPPING_ABDUCTION`: the primary IPC kidnapping/abduction aggregate.
- `GROUP_ROBBERY_DACOITY`: dacoity + robbery.
- `GROUP_PUBLIC_VIOLENCE`: arson + riots + unlawful assembly.
- `GROUP_PROPERTY_CRIME`: burglary + cheating + counterfeiting + criminal breach of trust + theft.

Each group is a row-wise sum with `min_count=1`: all-missing components yield missing; partial sums sum only observed components. This reproduces stored group values but does not make partial values complete totals.

## Target-source overlap and burden coverage

Nine exact source variables are shared by the grouped predictors and `VIOLENT_CRIME_BURDEN`: `IPC_ATTEMPT_TO_MURDER`, `IPC_CULPABLE_HOMICIDE_NOT_AMOUNTING_TO_MURDER`, `IPC_MURDER`, `IPC_KIDNAPPING_ABDUCTION`, `IPC_ARSON`, `IPC_RIOTS`, `IPC_DACOITY`, `IPC_ROBBERY`, and `IPC_RAPE`. They occur in five groups: homicide (three), kidnapping/abduction (one), public violence (two), robbery/dacoity (two), and sexual violence (one). The group-level mapping and target-side variables are listed in `outputs/audits/final_corrections_v5/direct_indirect_source_overlap.csv`.

The nine source counts sum to 3,228,843 over valid-target rows where those values are observed; target burden sums to 4,215,620 over all 9,631 valid-target rows. Their ratio is 76.592% by the saved descriptive definition. There are 9,552 valid-target rows with at least one observed overlap value; 79 have none. This ratio is not a row-wise decomposition or a deterministic reconstruction test. It must not be interpreted as explaining the remaining target burden, because the source formulas overlap and category/accounting relationships are incompletely documented. `direct_overlap_burden_summary.csv` contains the calculation and definition.

The other grouped source variables (`IPC_ASSAULT_ON_WOMEN_WITH_INTENT_TO_OUTRAGE_HER_MODESTY`, `IPC_SEXUAL_HARASSMENT`, `IPC_STALKING`, `IPC_VOYEURISM`, `IPC_UNLAWFUL_ASSEMBLY`, `IPC_BURGLARY`, `IPC_CHEATING`, `IPC_COUNTERFEITING`, `IPC_CRIMINAL_BREACH_OF_TRUST`, and `IPC_THEFT`) have no direct column match in the target input list. The available audit does not establish absence of indirect overlap for all these variables; NCRB classification and case-counting semantics are needed to decide that.

## Missingness distinctions

- **Missing target:** 225 of 9,856 integrated rows lack a valid burden target. This is a data state; the repository does not establish a single semantic reason for each row.
- **Missing stored grouped value:** each of the six groups is missing on the same 225 rows. The recovered sum rule keeps an all-missing component row missing.
- **Partial component missingness:** the component missingness audit reports a missing component in 729 homicide-group rows, 9,631 property-group rows, and 8,902 rows each for sexual-violence/public-violence components. Such a grouped number may be a partial sum; it is not made complete by the observed values.
- **Observed zero:** zero remains a recorded value, distinct from missing. No global missing-to-zero replacement is used.
- **Missing historical district-year:** 2,469 target observations do not have all three exact prior district-year records required by the sequence builder.
- **Incomplete historical feature values:** on otherwise buildable histories, the enhanced stream rejects 1,320 incomplete/nonfinite feature histories (baseline rejects one). No history is padded or imputed to retain it.

The reason for source missingness cannot be inferred from numeric values alone. See `group_missingness_overall.csv`, `group_component_missingness.csv`, `group_total_completeness.csv`, `sequence_stage_counts.csv`, and `pipeline_stage_counts.csv`.

## Crosswalk decision rule

Same-named canonical source columns and normalized aliases already documented in the integration notebook are verified aliases. The complete cross-stage alias/value audit is `outputs/audits/final_corrections_v5/raw_stage_reconciliation_audit.csv`. Do not infer that similarly worded offence headings are equivalent categories.

| Candidate mapping | Status | Evidence and decision |
|---|---|---|
| Normalized `COUNTERFIETING` → canonical `IPC_COUNTERFEITING` (2001–2013) | UNRESOLVED | Spelling is suggestive and normalized rows contain values, but the canonical integrated field is missing for these years. No crosswalk or authoritative source record validates equivalence; do not rename or fill. |
| 2014 `Attempt to commit Murder` → `IPC_ATTEMPT_TO_MURDER` | UNRESOLVED | Similar heading, but no verified NCRB year-to-year category mapping. |
| 2014 `Criminal Trespass/Burglary`, `Criminal Trespass or Burglary`, `House Trespass & House Breaking` → `IPC_BURGLARY` | UNRESOLVED | Multiple candidate headings and no verified category crosswalk or combination rule. |
| 2014 sexual harassment, stalking, unlawful assembly and voyeurism absent from earlier normalized IPC schemas | UNRESOLVED | Schema availability differs; absence of a field does not prove the offence was zero or not collected in another table. |

See `outputs/audits/final_corrections_v5/ipc_group_source_schema_discrepancies.csv` and `ipc_group_component_availability_by_year.csv` for the full audit. Required evidence: original year-specific NCRB codebooks/source tables and an authoritative crosswalk describing definitions, aggregation and breaks in series.
