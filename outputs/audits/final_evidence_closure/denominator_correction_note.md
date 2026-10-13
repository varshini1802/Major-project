# Denominator correction note

This note is an audit artifact for the author. The paper draft was not changed.

## What the reproducible ratio measures

The prior code's positive-burden-row ratio is **3,228,843 / 4,215,620 = 76.592364%**. The code uses **9,552** rows: valid target and finite nonnegative burden, observed direct-source sum, and burden greater than zero. These numerator and denominator sums use the same row mask.

This is a descriptive shared-component burden share over positive-burden rows. The all-valid-row calculation is also estimable and has the same ratio because the 79 excluded valid rows have zero burden and observed zero shared-source contribution. It does not demonstrate that every label can be reconstructed from the predictor subset.

## Why 9,552 differs from 9,631

There are 9,631 valid target rows. All 9,631 have at least one observed value among the nine shared source columns. The legacy 9,552 count is the number of positive-burden rows because the original code also requires `burden > 0`; the remaining 79 rows have zero burden and observed zero direct-source contribution. Thus overlap-source coverage is **9,631/9,631 = 100.000000%**. Missing values are not set to zero.

The all-valid-row burden share is estimable because the direct-source sum is observed for every valid target row. Its numerator and denominator equal the positive-burden-row sums: the excluded 79 rows add zero to both. The previous audit's phrase “79 other valid-target rows have no observed value among these sources” is contradicted by the current inputs and is not supported by the prior code's mask.

## Suggested author action

The audit recalculation supports the ratio on all 9,631 valid-target rows and shows it equals the positive-burden subset ratio (9,552 rows). The draft's ratio values are numerically reproducible, but its accompanying 9,552/79 coverage sentence is wrong: 9,552 is the positive-burden mask count, and all 9,631 valid rows have observed direct-source values. The author should amend that interpretation, not change the paper during this audit. No claim about indirect category overlap or deterministic full-label reconstruction is supported by this ratio.

Recalculation code: `src/final_evidence_closure.py::calculate_overlap_reconciliation` and `denominator_audit`. Input file hash is recorded in `reproducibility_manifest.json`.
