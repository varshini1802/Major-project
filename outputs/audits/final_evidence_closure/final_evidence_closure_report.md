# Final evidence-closure report

## Scope and baseline

This audit began with the repository status captured in `initial_git_status.txt`; substantial modified and untracked project files pre-existed this task. Existing closure outputs and the target, feature, integration, missingness, sequence, and split code were inspected before adding the evidence-specific audit. The frozen target thresholds remain 107, 237, 362, 563 (fit 2001–2010); partitions remain train 2001–2010, validation 2011–2012, locked test 2013–2014. No data, labels, target backup, locked keys, or saved results were written by this audit.

## Issue findings

### 1. Target–predictor overlap — direct confirmed; indirect blocked

The target sums the 11 `BASE_PAPER_GROUPS` with `min_count=1`, then sums group totals with `min_count=1`; any observed negative component invalidates the row. The six grouped predictor formulas are row sums with `min_count=1`. The matrix accounts for all 19 predictor source columns and confirms **9 direct shared IPC components** across five grouped predictors. Confirmed indirect NCRB counting overlaps: **0**. Potential semantic relations: **9** mapped predictor-source rows; additional unresolved source rows: **10**. No `NO_IDENTIFIED_OVERLAP` is used as a claim of independence. The direct sources feed the tabular baseline/enhanced feature families and their LSTM/GRU historical sequences; same-year notebook classification is separate.

On the 9,631 valid rows, shared components contribute 76.592364% of summed burden. The prior code's 9,552-row mask excludes 79 valid rows with zero burden and observed zero direct-source contribution. The direct components therefore reconstruct a portion of target burden on all valid rows, not the remaining burden or exact class labels. Confirmed indirect event-count duplication is not established.

The repository relationship review records legal and empirical category relationships, but explicitly says those do not prove NCRB’s event-counting behavior. Official year-specific definitions/crosswalks and aggregation rules for 2001–2014 were not located. This issue remains externally blocked.

### 2. Historical feature availability — structural checks pass; operational origin undefined

The project defines keyed t−1 tabular inputs and strict same-district t−3..t−1 sequence inputs, but does not define a forecast issue date or operational decision point. The official OGD catalog gives catalog publication dates: the 2001–2012 resource was published 17 September 2013; the 2013 resource, which supplies t−1 inputs for the locked 2014 forecasts, was published 16 October 2014; the separate 2014 resource was published 18 August 2016 and is not used as a t−1 predictor for a 2014 target. The catalog describes figures as counts and notes police districts/special units may differ from revenue districts. These are catalog dates, not actual NCRB release dates, retrieval dates, or a particular forecaster’s access date. Each ledger row therefore reports `SCENARIO_UNDEFINED`; a strict date check with a known origin and unknown actual release date returns `UNKNOWN`, never available. Sources: [OGD IPC catalog](https://www.data.gov.in/catalog/district-wise-crimes-under-various-sections-indian-penal-code-ipc-crimes) and [2013 resource](https://www.data.gov.in/resource/district-wise-crime-under-various-sections-indian-penal-code-ipc-crimes-during-2013).

### 3. Provenance, duplicates, and boundaries — partial or externally blocked

The repository contains normalized, cleaned, core, and integrated processed files, but no `data/raw` directory or verified original source download. Processed-file hashes and known transformation code are listed without treating processed files as raw downloads. Read-only re-canonicalization rechecked **100 integrated IPC aliases** with **0 value mismatches**. Seven duplicate normalized key groups (14 rows) across five cleaned files were recomputed and compared to the earlier audit. Five have identical offence payloads; two 2010 Railway groups conflict in IPC and women records. Pipeline row IDs are not proven NCRB identifiers; both conflicting groups remain excluded by the existing quality rule, with no aggregation or record selection. The OGD catalog’s police-district/special-unit caveat confirms a distinction from revenue districts but supplies no historical boundary crosswalk. Original data, Railway unit identifiers, district codes, and merger/split histories are required to close these issues.

### 4. Missingness — counts/formulas verified; causes unknown

The audit recomputes **532 IPC source-column/year rows** from the canonical integrated file, plus group and engineered-feature counts. It keeps observed zero, positive, negative, missing, prior-year gap, excluded duplicate, and schema mismatch records separate. Aggregation code preserves `min_count=1`; all-missing group inputs remain missing, partial aggregates are not complete totals, and missing year-over-year inputs remain missing. `CRIME_DIVERSITY_INDEX` treats an individual missing group as zero contribution within its entropy calculation, while an all-groups-missing row remains missing; this is feature-specific behavior, not a blanket dataset fill. No documented reason is assigned to an individual missing source cell. The distinction between structural gap and unknown cause is marked where keyed prior-year evidence permits; missingness cause remains `UNKNOWN_CAUSE` absent original forms or source reporting notes.

### 5. Denominator — corrected reproducibly

The all-valid-row ratio is 3,228,843/4,215,620 = 76.592364%, with 9,631/9,631 rows having observed overlap components. The original positive-burden mask gives the same ratio on 9,552 rows because the 79 omitted rows have zero numerator and denominator contribution. The earlier 79-row missing-source interpretation was incorrect. See `denominator_correction_note.md`; the draft remains unchanged.

## Validation and preservation

The evidence-specific regression tests cover all nine direct inputs, availability statuses, denominator masks and sums, missingness/aggregation semantics, duplicate-key preservation, frozen thresholds, chronological partitions, and protected input hashes. Test and audit outcomes are recorded in `test_report.txt`. Models were not retrained. Protected hashes are compared with the initial snapshot and recorded in `reproducibility_manifest.json`; any mismatch fails the generator. The paper and saved experiment outputs are not modified.

## Exact evidence still needed

1. NCRB year-specific district IPC tables/crosswalks and documentation stating whether aggregate and subcategory counts overlap or are mutually exclusive, including applicable years.
2. A declared forecast origin plus actual release/issue records for each annual NCRB source resource; catalog publication and retrieval records must remain distinct.
3. The original downloaded files, stable source identifiers, URL/catalog IDs, retrieval dates and hashes linking them to normalized files.
4. Original Railway unit names/codes and NCRB records for the two conflicting 2010 keys.
5. Official year-specific district/state codes and merger/split boundary tables for 2001–2014.
6. Original source reporting notes or extraction logs identifying why missing IPC component cells occur, including cells in invalid-target rows.
