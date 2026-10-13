# Final data-quality and claim audit

Generated from saved processed tables and prior reconciliation artifacts; raw inputs were not changed.

## Target and protocol

Frozen boundaries are 107, 237, 362, and 563, fitted on 2001–2010. The target is a project-defined adaptation of ordered violent-crime burden levels: class 1 lowest, class 2 low-to-moderate, class 3 moderate, class 4 moderate-to-high, class 5 highest. Split: train 2001–2010, validation 2011–2012, test 2013–2014.

Canonical target table has 9856 rows; valid targets=9631; missing targets=225.

## Missingness interpretation

A numeric zero is counted as an observed zero. NaN is counted as missing/unavailable; the processed group table does not encode whether a missing source means non-reporting, unavailable, or another cause, so those causes cannot be distinguished from these files. Partial grouped totals sum observed groups with `min_count=1`; they are not complete totals. No global zero-fill is applied. At the six stored group-column level, 9,631 rows have all six groups observed and 225 rows have all six missing; there are no rows with only some stored group values observed. At the underlying IPC-component aggregation level, partial source coverage does occur; see `group_predictor_formula_recovery.csv` for per-group counts (these group counts can overlap and are not unique-row totals).

Detailed missingness is in `group_missingness_overall.csv`, `group_missingness_by_year.csv`, `group_missingness_by_state.csv`, and `group_missingness_by_district.csv`; grouped total completeness is in `group_total_completeness.csv`.

## Exclusions and integration evidence

Stage counts for target labels, tabular matrices, and strict sequence histories are in `pipeline_stage_counts.csv` and `sequence_stage_counts.csv`. Matrix and sequence counts use canonical district-year keys and the frozen labels. The raw reconciliation audit explains available alias normalization and 2014 row filtering; original upstream raw copies are absent, so provenance before the available normalized stage remains unresolved.

Available-stage reconciliation found 100 final IPC measures matching their normalized raw-stage aliases across 9,856 district-year rows with zero value mismatches. In the 2014 branch, normalization-to-cleaned retained 838 rows and core filtering retained 729 (73 special units and 36 aggregate rows removed); the compared values matched with zero mismatches. This verifies only the available normalized/cleaned/core artifacts, not provenance back to absent upstream copies.

The nine unique IPC sources shared directly by predictor groups and target contribute 76.6% of summed burden on 9552 target-valid rows with observed overlap-source values; 79 other valid-target rows have no observed value among these sources. This is a descriptive reconstruction fraction; it does not resolve all component relationships or imply deterministic class recovery. See `direct_overlap_burden_summary.csv`.

## Claims and limitations

The same-year random-split notebook is observed-year classification only. Its cached scores, including the legacy 90.76% figure, are not corrected forecasting results. The forecasting evaluations use historical keyed inputs and chronological partitions. Annual source-release dates are unavailable, and possible IPC parent/component relationships remain unresolved; operational forecast-time availability and complete indirect-overlap resolution are not claimed.

No conference-paper manuscript was found in the repository file inventory during this audit. No manuscript text or claims were edited.
