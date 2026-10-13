# Targeted evidence-gap investigation

## Overall status: `BLOCKED_PENDING_EXTERNAL_EVIDENCE`

Existing repository evidence narrows all six issues without changing research inputs. Code and saved-stage checks verify implementation facts, but the remaining claims depend on original NCRB documentation, source records, release timestamps, or geographic identifiers not present locally. No official counting rules, source provenance, historical release dates, or geographic crosswalks are inferred.

This investigation did not modify the paper draft, datasets, target thresholds, labels, locked test keys, saved predictions, models, or official results. No models were trained. Machine-readable evidence is in `evidence_gap_register.csv`, `target_overlap_source_mapping.csv`, and `source_stage_reconciliation.csv`.

## 1. Shared target and predictor IPC sources

The exact source mapping identifies nine directly shared IPC columns across five grouped predictors: `GROUP_HOMICIDE` ? `IPC_ATTEMPT_TO_MURDER`, `IPC_CULPABLE_HOMICIDE_NOT_AMOUNTING_TO_MURDER`, `IPC_MURDER`; `GROUP_SEXUAL_VIOLENCE` ? `IPC_RAPE`; `GROUP_KIDNAPPING_ABDUCTION` ? `IPC_KIDNAPPING_ABDUCTION`; `GROUP_ROBBERY_DACOITY` ? `IPC_DACOITY`, `IPC_ROBBERY`; and `GROUP_PUBLIC_VIOLENCE` ? `IPC_ARSON`, `IPC_RIOTS`. `GROUP_PROPERTY_CRIME` has no direct shared source among its recovered inputs; this does not establish absence of indirect counting overlap. Formula recovery records all six groups as `sum(min_count=1)`: all-missing inputs remain missing, partial missing inputs are skipped, no other stored-column transformation is applied, and row-wise formula validation has zero mismatches. The original group-generation implementation was not found; formulas were recovered from committed provenance artifacts and validated against the saved grouped dataset.

The ratio is **3,228,843 / 4,215,620 = 76.592364%**: summed shared-source counts divided by total target burden. It is not a per-row average, an event-level duplicate count, a percentage of labels exactly reconstructed, or proof that categories are mutually exclusive. The earlier 9,552-row mask was restricted to positive-burden rows. All 9,631 valid-target rows have at least one shared source observed; the 79 omitted rows have zero burden and observed zero shared contribution, so the all-valid-row ratio is the same.

Existing legal and empirical reviews flag possible relationships among theft/robbery/dacoity, murder/dacoity-with-murder, rape subheads, kidnapping subheads, and riot subheads. Such relationships do not prove NCRB counted a case in multiple variables. **Missing evidence:** official year-specific IPC definitions, table notes/codebooks, and event-counting or mutually-exclusive aggregation rules for 2001?2014.

## 2. Historical availability at forecast origin

Code-level alignment is documented: tabular features use keyed *t?1* records and sequences use same-district *t?3?t?1* records. The prior feature-availability ledger has 552 rows marked `SCENARIO_UNDEFINED`; the project does not establish a calendar forecast issue date. Prior audits record catalog publication metadata of 2013-09-17 for the 2001?2012 OGD resource, 2014-10-16 for the 2013 resource, and 2016-08-18 for the 2014 resource. These dates are catalog metadata, not verified NCRB release dates, retrieval timestamps, or proof of a forecaster's access at a specified origin.

**Missing evidence:** a declared forecast origin plus original NCRB release/issue records and contemporaneous retrieval/access timestamps for each used source year. Structural lag alignment passes; operational availability remains unverified. Evidence: `outputs/audits/final_technical_completion/forecast_origin_and_availability.csv` and `outputs/audits/final_evidence_closure/forecast_availability_ledger.csv`.

## 3. 2014 stage reconciliation and provenance

The available processed-stage evidence narrows the numerical issue. One prior audit compared 100 integrated IPC aliases between `data/processed/district_integrated_raw.csv` and `district_integrated_final.csv` across 9,856 rows and found zero value mismatches. For the audited 2014 columns, normalized and cleaned stages have the same 838 keys and values; cleaned-to-core retains 729 of 838 keys with zero value mismatches on retained keys. The 109 excluded rows comprise 73 special units and 36 aggregates. Eight available 2014 case-variant IPC fields match final values on the 729 retained keys, with differences described as case/feature normalization.

This explains the observed row reduction and local alias normalization. The file named `district_integrated_raw.csv` is a processed staging file, not proof that the original NCRB raw download is available. No original 2014 download, resource ID tied to that exact file, retrieval timestamp/hash, or complete extraction record was found. Thus available local stages reconcile with zero value mismatches on compared values, while original-source provenance remains unresolved. Full prior rows and interpretations are in `source_stage_reconciliation.csv`.

## 4. Duplicate district-year records

The reviewed cleaned stages contain seven duplicate normalized key groups (14 rows) in five files: five have identical offence payloads with different pipeline row identifiers, and two 2010 Jammu & Kashmir / `RAILWAYS` groups have conflicting payloads (IPC and women records). The documented quality rule excludes these groups from core, and the canonical model keys are unique. Pipeline `SOURCE_ROW_ID` and `SOURCE_RECORD_ID` values are not verified NCRB identifiers. Exclusion avoids an invented selection or aggregation rule but does not identify whether the Railway rows are distinct units, duplicates, or a mapping collision.

**Missing evidence:** original 2010 NCRB records, stable Railway reporting-unit identifiers or official crosswalk, and extraction provenance.

## 5. Geographic continuity

The pipeline normalizes `YEAR`, `STATE`, and `UNIT_NAME` and checks within-year key uniqueness. These are name-based key checks, not geographic identity checks. For example, the cleaned 2014 IPC source has 36 states and 783 normalized names; the core has 709 names after unit/quality filtering. These counts alone do not identify or prove a boundary change.

**Missing evidence:** official dated district/state codes and 2001?2014 split, merger, renaming, police-district, and special-unit crosswalks. Geographic continuity remains unresolved.

## 6. Missingness, zeros, and partial sums

Audits distinguish observed zero from missing. Each of the six grouped columns has 225 missing values among 9,856 rows (2.282873%); formula evidence classifies these as rows where all sources for that group are missing. Observed zero counts range from 28 (`GROUP_PROPERTY_CRIME`) to 781 (`GROUP_KIDNAPPING_ABDUCTION`). Group formulas use `sum(min_count=1)`: available components are summed when at least one is observed, missing components are skipped, and all-missing groups remain missing. Formula-to-saved-value comparisons report zero mismatches. The component-level audit contains 532 source-column/year summary records.

A partial sum is not a complete group total when one or more source components are missing. `CRIME_DIVERSITY_INDEX` has a separately documented rule that treats an individual missing group as a zero contribution while preserving an all-groups-missing row as missing; this is not a blanket fill. Numeric values do not reveal whether a source cell was unreported, structurally inapplicable, unavailable, or lost during extraction. **Missing evidence:** original source forms, reporting notes/data dictionary, and extraction logs.

## Validation and preservation

- `python -m unittest discover -s tests`: **PASS**, 41 tests.
- `python -m compileall -q src ML_Implementation tests`: **PASS**, exit code 0.
- Existing reports were inspected before creating these files; prior audit artifacts were not overwritten.
- Protected-file hash comparison: 198 files checked against the existing baseline; zero mismatches.
- The paper draft, datasets, thresholds, labels, locked test keys, saved predictions, models, and official results are unchanged. No retraining occurred.
- The working tree had existing modifications and untracked files before this task; those changes were preserved. Exactly the five requested audit files were created under this directory.

## Evidence needed to close the remaining gaps

1. Official NCRB year-specific IPC codebooks/table notes and counting rules for parent/subcategory and cross-domain categories.
2. A declared operational forecast origin and actual source release/access records.
3. Original NCRB downloads and a verifiable URL/resource ID, retrieval date, checksum, and extraction/transformation manifest.
4. Original 2010 Railway identifiers and official mapping for the conflicting rows.
5. Stable official district/state codes and dated geographic crosswalks for 2001?2014.
6. Source reporting notes/forms and extraction logs explaining missing cells.
