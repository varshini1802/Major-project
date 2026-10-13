# Integration, duplicate-key, and missingness evidence

Read-only rechecks of available processed stages are preserved in `outputs/audits/final_evidence_closure/`. The prior closure recomputed 100 integrated IPC aliases with zero value mismatches. Seven duplicate normalized key groups (14 rows) were found in the cleaned stages; five duplicate groups have identical offense payloads and two 2010 Railway groups conflict. Existing processing excludes both conflicting groups; this audit does not select, aggregate, or modify them. Pipeline row identifiers are not verified NCRB stable identifiers.

Aggregation uses `min_count=1`: all-missing inputs remain missing, partial sums remain partial, and no global missing-to-zero conversion is used. The diversity index specifically treats an individual missing group as a zero contribution while preserving all-groups-missing as NaN. Missingness counts and strata are available, but causes cannot be attributed without original source forms, reporting notes, or extraction logs.

No `data/raw` source download, source retrieval manifest, stable Railway unit code, or year-specific district boundary crosswalk is present. Therefore integration provenance, conflicting Railway identity, historical unit continuity, and the 2014 raw-stage/final-integrated discrepancy remain externally unresolved. Original and processed inputs were not changed.
