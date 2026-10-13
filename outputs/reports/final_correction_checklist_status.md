# Final correction checklist status

Statuses are based on executed audits and saved artifacts. `PARTIAL` and `UNRESOLVED` items remain open.

| # | Requirement | Status | Evidence / remaining limit |
|---:|---|---|---|
| 1 | Target-source overlap | PARTIAL | Nine direct shared IPC sources; direct counts are 76.6% of summed burden on 9,552 observed-overlap rows; indirect parent/component relationships remain unresolved. |
| 2 | Tie-safe quantile target | COMPLETE | Frozen 107/237/362/563 boundaries; row-order and target saved-data audits pass. |
| 3 | Five class meanings | COMPLETE | Canonical code and notebook describe ordered violent-burden levels; notebook stale class names corrected. |
| 4 | Engineered features | COMPLETE | All five formulas and all 50 five-seed one-feature ablation runs are saved and integrity-audited. Source availability limits are tracked separately under items 1, 7 and 12. |
| 5 | Chronological evaluation | COMPLETE | Train 2001-2010, validation 2011-2012, test 2013-2014; random notebook sections observed-year only. |
| 6 | Preprocessing and selection scope | COMPLETE | Corrected pipelines fit learned transforms on train, select using validation, then refit train+validation before locked test. |
| 7 | Previous-year alignment | PARTIAL | District/year keyed t-1 alignment passes; publication dates are unavailable. |
| 8 | Integration and keys | PARTIAL | Available-stage aliases and key checks reconcile; original upstream copies and administrative-boundary provenance are unavailable. |
| 9 | Missing values | PARTIAL | Zero/missing counts and missingness strata are reported; causes of NaN are not encoded, so zero versus non-reporting provenance cannot be fully established. |
| 10 | Metrics and baselines | COMPLETE | Accuracy, macro/weighted F1, class reports/supports, confusion matrices, class distributions, majority and logistic baselines, macro OvR ROC-AUC and macro mean OvR AP saved. |
| 11 | LSTM/GRU sequence method | COMPLETE | Strict t-3..t-1 district histories; paired GRU and LSTM feature comparison plus logistic baseline; common test keys verified. |
| 12 | Leakage audit | PARTIAL | Automated structural/source audits pass; semantic indirect overlap and real issue-date availability remain unverified. |
| 13 | Reproducibility | PARTIAL | Seeds, configurations, versions, keys, metrics, models, and completed ablation audits saved; full upstream source provenance is missing. |
| 14 | Paper abstract | PARTIAL | A new draft exists at `outputs/reports/conference_paper_draft.md`; requires author and venue review. |
| 15 | Paper introduction/objectives | PARTIAL | New project draft covers objectives and related work; literature review needs author review. |
| 16 | Paper dataset/preprocessing | PARTIAL | Draft records verified project data and open schema/provenance limits. |
| 17 | Paper methodology | PARTIAL | Draft records target, chronological split, keyed lags, sequences and fit scope. |
| 18 | Paper results | COMPLETE for saved evidence | Draft includes saved v4, five-seed and ablation results, supports, per-class metrics, and generated figures; independent author review remains. |
| 19 | Paper discussion | PARTIAL | Draft discussion distinguishes model-family and metric differences; author review required. |
| 20 | Paper limitations/future work | PARTIAL | Draft lists unresolved source, boundary and release-date dependencies. |

The corrected v4 controlled comparison is stored separately from v2/v3. No original conference-paper manuscript or template was located; the separate new draft is not submission-ready.
