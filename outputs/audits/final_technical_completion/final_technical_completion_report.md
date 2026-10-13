# Final technical completion report

## Decision and scope

The corrected temporal matrices do not contain same-year target-source components: tabular inputs are exact district-keyed t-1 records and recurrent inputs are strict t-3..t-1 records. The random-split notebook remains a separate observed-year classification and does contain contemporaneous crime-count overlap. No corrected experiment was retrained in this audit.

The completion matrix records corrections 1-13 individually. Corrections 2, 3, 5, 6, 10, and 11 are supported as complete by code and saved evidence. Corrections 1, 4, 7-9, 12, and 13 are partial because semantic counting, forecast-time availability, source provenance, missingness causes, or administrative identifiers remain unresolved.

## Empirical checks

Frozen boundaries remain 107, 237, 362, and 563 from the 2001-2010 training period. Official split remains training 2001-2010, validation 2011-2012, locked test 2013-2014. Sequence inputs require consecutive t-3,t-2,t-1 observations for the same normalized district-year key; missing history is rejected, not fabricated. Corrected pipelines fit learned preprocessing on training data during validation selection; final refits use train+validation after selection.

The direct shared-component ratio is independently recalculated in `denominator_reconciliation.csv`; positive-burden and all-valid masks are both retained.

The 198-file protected-hash set from the prior closure matches the current source/processed data, target backup, frozen thresholds, locked test keys, and saved test predictions/results with zero mismatches.

## Remaining blockers

External evidence is required for NCRB category counting/parent-child relationships; original downloads and retrieval dates; annual publication/access timestamps and a defined operational origin; stable unit codes and district boundary histories; causes of missing source values; and source documentation for the unresolved 2014 raw-stage/final-integrated discrepancy. Catalog publication dates are not treated as actual release dates.

The paper draft and sections 14-20 were not edited. Source and processed inputs, target backup, frozen thresholds, locked test keys, and existing result outputs are preserved. See `final_test_report.txt` and `reproducibility_manifest.json` for commands and hashes.
