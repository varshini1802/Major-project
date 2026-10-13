# NCRB final evaluation and release-readiness report

## Decision

**Overall status: `READY_FOR_TECHNICAL_SIGN_OFF_WITH_DOCUMENTED_LIMITATIONS`.** The corrected saved forecasting artifacts pass the verified target/key, feature-year, sequence-year, and recomputed metric checks performed here. This status does not verify operational historical data availability, official NCRB counting semantics, source provenance, boundary continuity, or missingness causes.

No models were retrained. Metrics were recomputed from saved predictions and probabilities; no official result files were overwritten. The existing corrected temporal experiment outputs remain the evaluated artifacts.

The audit reconciled 78 saved prediction files; saved summary metric mismatches: 0; prediction-to-canonical-label/key failures: 0. See the CSVs for per-artifact details.

## Experiment integrity and performance interpretation

Tabular inputs are keyed same-district t-1 values. Strict sequence inputs use consecutive t-3, t-2, t-1 records. All evaluated target years are 2013-2014; sequence cohorts and prediction labels are checked by key. The locked sequence test cohort contains 671 rows. Tabular test cohorts contain their feature-eligible district-year records and are not conflated with the sequence cohort. The observed-year random-split notebook remains classification of already-observed burden and is excluded from forecast comparisons.

The five-seed v5 paired comparisons and one-feature ablations are summarized descriptively in `model_comparison_and_stability.csv`. The five-seed enhanced-minus-baseline macro-F1 mean deltas are +0.00785 (GRU), +0.01248 (LSTM), and +0.00016 (logistic regression); gains occurred in 4/5, 5/5, and 3/5 seeds respectively. These are fixed-test descriptive results, not significance tests. Direct component overlap is historical in these forecast inputs, while indirect NCRB category-count relationships remain unresolved.

The v1/v2 sequential outputs include six saved baseline/enhanced prediction files with unequal test-key sets (1,276 baseline versus 671 enhanced records); they are superseded historical runs and are not valid paired comparisons. Later v3 common-cohort and v4/v5 controlled runs use the 671-key cohort. The cohort report marks the v1/v2 mismatches explicitly.

Model complexity is not presumed useful: LSTM and GRU should be retained as exploratory comparisons only where their paired results provide consistent improvement over logistic regression and simpler baselines. Test-set results are descriptive and were not used for model selection.

## Remaining partial corrections

Corrections 1, 4, 7, 8, 9, 12, and 13 remain PARTIAL. No further supportable code defect was identified in these areas. See `remaining_correction_actions.csv` and `external_evidence_requirements.csv` for specific actions and source materials needed.

## Preservation and limitations

The 198 protected files from the prior closure hash set are unchanged. Target thresholds remain 107, 237, 362, 563 (fit on 2001-2010); split remains train 2001-2010, validation 2011-2012, locked test 2013-2014. The paper draft was not edited. Historical availability, original downloads, NCRB category rules, missingness explanations, Railway IDs, and administrative boundary history remain unverified. The 2014 raw-stage/final-integrated discrepancy remains unresolved.

Technical sign-off is therefore limited to implementation and saved-experiment reproducibility, not operational deployment or full data provenance.
