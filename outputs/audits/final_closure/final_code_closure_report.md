# NCRB code, data quality, and forecasting closure

## Scope and outcome

The frozen target thresholds remain 107, 237, 362, and 563, fitted on 2001–2010. The chronological split remains train 2001–2010, validation 2011–2012, and locked test 2013–2014. This closure package audits code, data, forecasting validity, saved results, and reproducibility. It does not alter the research paper.

Structural audits support the target implementation, t−1 tabular feature alignment, strict same-district t−3..t−1 sequences, train/validation-only model selection, and locked-test artifact integrity. They do not establish that historical NCRB annual values were published by a specific operational forecast date. Consequently, operational forecast availability and semantic independence from overlapping source counts are unresolved.

## Verified corrections

- Nine grouped-predictor source columns directly overlap target inputs. Row-wise source formulas use `min_count=1`; missing source values are skipped while all-missing rows stay missing. Formula checks report zero mismatches against stored grouped values.
- Tabular baseline and enhanced lag fields match the keyed same-district t−1 sources on 8,682 rows, with zero value mismatches. The predictors are structurally historical, but actual release dates remain unverified.
- Saved sequences require consecutive same-district years t−3, t−2, t−1. The strict paired cohort has 5,842 sequences; incomplete histories are rejected rather than fabricated. Baseline strict sequences: 7,161. Locked test cohort: 671 district-year keys.
- Target verification artifacts report the frozen boundaries, row-order invariance, holdout perturbation behavior, and downstream label agreement.
- Saved v4 test metrics were recomputed and all eight model/variant rows validate. Five-seed comparison and 50/50 feature-ablation integrity checks pass. No model was retrained for this closure.
- Processed-stage reconciliation includes normalized aliases and final integrated values. Seven duplicate-key groups (14 rows) across five cleaned files were reviewed; two Railway groups conflict and lack source identifiers.
- Missing values and observed zeros are separately counted. There are 9,856 canonical rows, 9,631 valid targets, and 225 missing targets. Missingness causes are not identifiable from stored columns.

## Discrepancies and external evidence still needed

- The source-overlap audit denominator is 9,552 rows with observed overlap values. Its burden ratio is 3,228,843 / 4,215,620 = 76.592%. The paper draft contains denominator wording referring to all 9,631 valid-target rows; that sentence is inconsistent with the calculation. The draft was left untouched as instructed.
- A t−1 source is only structurally prior. The portal catalog lists the 2013 resource as published 16 October 2014 and the 2014 resource as published 18 August 2016. Catalog publication timestamps do not establish the earliest NCRB release or the forecast decision date. Operational feature availability therefore remains unresolved. See [OGD India IPC district-crime catalog](https://www.data.gov.in/catalog/district-wise-crimes-under-various-sections-indian-penal-code-ipc-crimes).
- Original upstream downloads, year-specific official schema/crosswalks, Railway unit identifiers, administrative boundary histories, and official definitions for unavailable/missing entries are absent. The `COUNTERFIETING`/counterfeiting and 2014 heading differences are not silently mapped.
- Direct shared columns are confirmed. Possible parent/component or duplicate case-counting relationships cannot be settled from legal definitions or correlations; official NCRB statistical tabulation rules are required.

## Commands and artifacts

See `test_report.txt` for exact test and compilation results. `reproducibility_manifest.json` records versions, hashes, seeds, thresholds, split, sample counts, and commands. The CSV audits in this directory preserve machine-readable evidence and unresolved statuses. Model suites were not rerun because the saved frozen artifacts passed integrity checks and no implementation defect affecting those artifacts was found.

## File preservation

No paper draft, abstract, manuscript section, original source file, processed dataset, frozen prediction, or target backup was modified by this closure generator. This task adds the focused feature-engineering availability test and this read-only audit generator, and corrects the V5 runner's stale ablation description in its code and saved configuration.
