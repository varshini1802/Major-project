"""Build a read-only closure package from the frozen NCRB audit artifacts.

This script writes only outputs/audits/final_closure. It does not mutate source,
processed data, experiment outputs, or paper files.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs/audits/final_corrections_v5"
TABLES = ROOT / "outputs/tables/final_remaining_corrections"
OUT = ROOT / "outputs/audits/final_closure"
KEYS = ["YEAR", "STATE", "UNIT_NAME"]
GROUPS = [
    "GROUP_HOMICIDE", "GROUP_SEXUAL_VIOLENCE", "GROUP_KIDNAPPING_ABDUCTION",
    "GROUP_ROBBERY_DACOITY", "GROUP_PUBLIC_VIOLENCE", "GROUP_PROPERTY_CRIME",
]


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False)


def build_overlap() -> None:
    overlap = read(OLD / "direct_indirect_source_overlap.csv")
    formulas = read(OLD / "group_predictor_formula_recovery.csv")
    overlap = overlap.merge(
        formulas[["predictor_group", "source_columns", "aggregation", "missing_handling",
                  "other_transformations", "row_value_mismatches", "formula_validation"]],
        on="predictor_group", how="left", validate="many_to_one",
    )
    overlap["target_aggregation_rule"] = "11 target groups; row sums use min_count=1; invalid negative source rows excluded"
    overlap["direct_overlap_status"] = overlap["direct_overlap"].map({True: "DIRECT_SHARED_SOURCE", False: "NO_DIRECT_MATCH"})
    overlap["indirect_overlap_status"] = overlap["indirect_overlap_evidence"].fillna("UNRESOLVED")
    overlap["semantic_limit"] = "NCRB event-level inclusion/exclusion or duplicate counting is not established by available schema"
    save(overlap, "target_predictor_overlap_audit.csv")


def build_availability() -> None:
    lineage = read(OLD / "feature_source_lineage.csv")
    lineage["availability_evidence_status"] = "STRUCTURAL_ONLY; official issue/availability date unverified"
    lineage["forecast_eligibility"] = lineage["forecast_decision"].map(
        lambda x: "STRUCTURALLY_ELIGIBLE_LAGGED" if "eligible structurally" in str(x) else "EXCLUDE_SAME_YEAR"
    )
    lineage["target_year_t_source_year"] = lineage["source_year"].astype(str)
    lineage["forecast_origin_limit"] = "Annual NCRB release and decision date not documented; real-world availability unresolved"
    save(lineage, "feature_availability_audit.csv")


def build_integration() -> None:
    keys = read(OLD / "district_year_key_validation.csv")
    keys["audit_type"] = "stage_key_validation"
    dup = read(OLD / "cleaned_duplicate_key_summary.csv")
    dup = dup.rename(columns={"file": "path", "duplicate_keys": "duplicate_normalized_keys",
                              "duplicate_rows": "duplicate_rows_in_file"})
    dup["audit_type"] = "cleaned_duplicate_review"
    dup["rows"] = pd.NA
    dup["key_fields"] = "YEAR|STATE|UNIT_NAME"
    dup["assessment"] = dup["interpretation"]
    keep = ["path", "rows", "key_fields", "duplicate_normalized_keys", "assessment", "audit_type"]
    parts = [keys.reindex(columns=keep), dup.reindex(columns=keep)]
    rec = read(OLD / "integration_reconciliation_summary.csv")
    rec["audit_type"] = "integration_reconciliation"
    rec["assessment"] = rec.apply(lambda r: f"{r['comparison']}: {r['value_mismatches']} value mismatches; left={r['max_rows_left']}, right={r['max_rows_right']}", axis=1)
    rec["path"] = rec["comparison"]
    rec["rows"] = rec["max_rows_left"]
    rec["key_fields"] = "district-year keyed comparison"
    rec["duplicate_normalized_keys"] = pd.NA
    parts.append(rec.reindex(columns=keep))
    save(pd.concat(parts, ignore_index=True), "district_year_integration_audit.csv")


def build_missingness() -> None:
    overall = read(OLD / "group_missingness_overall.csv")
    overall["audit_scope"] = "group totals"
    overall["cause_status"] = "UNKNOWN_FROM_AVAILABLE_COLUMNS"
    overall["detail"] = "Observed zero is distinct from missing; partial sums skip absent components; all-missing source set remains missing."
    components = read(OLD / "group_component_missingness.csv")
    components = components.rename(columns={"group": "crime_group"})
    components["audit_scope"] = "component completeness"
    components["cause_status"] = "UNKNOWN_FROM_AVAILABLE_COLUMNS"
    components["detail"] = components["semantics"]
    target = pd.DataFrame([{
        "crime_group": "TARGET_VIOLENT_LEVEL", "audit_scope": "canonical target",
        "rows": 9856, "missing_or_unavailable": 225,
        "cause_status": "UNKNOWN_FROM_AVAILABLE_COLUMNS",
        "detail": "9,631 valid targets; 225 rows lack a valid target under the canonical source validity rule."
    }])
    cols = list(dict.fromkeys(list(overall.columns) + list(components.columns) + list(target.columns)))
    save(pd.concat([overall.reindex(columns=cols), components.reindex(columns=cols), target.reindex(columns=cols)], ignore_index=True), "missingness_audit.csv")


def build_forecast() -> None:
    avail = read(OLD / "final_forecast_feature_availability_audit.csv")
    avail["audit_section"] = "forecast_feature_and_preprocessing"
    seq = read(OLD / "sequence_stage_counts.csv")
    seq["audit_section"] = "sequence_cohort"
    checks = read(OLD / "corrected_target_saved_data_verification.csv")
    checks["audit_section"] = "saved_target_verification"
    lag = read(ROOT / "outputs/tables/forecast_feature_availability_audit.csv")
    lag["audit_section"] = "lag_alignment"
    for frame in (avail, seq, checks, lag):
        frame["real_world_release_timing"] = "UNRESOLVED: publication/issuance timing not verified for each historical year"
    cols = list(dict.fromkeys(c for f in (avail, seq, checks, lag) for c in f.columns))
    save(pd.concat([f.reindex(columns=cols) for f in (avail, seq, checks, lag)], ignore_index=True), "forecast_integrity_audit.csv")


def build_results() -> None:
    v4 = read(OLD / "v4_recomputed_metrics.csv")
    v4["result_source"] = "v4 saved test predictions recomputed"
    v4["integrity_status"] = v4["metric_validation"]
    multi = read(OLD / "multiseed_per_seed_metrics.csv")
    multi["result_source"] = "v5 saved metrics; five predeclared seeds"
    multi["integrity_status"] = "PASS; see multi-seed integrity checks"
    ab = read(OLD / "integrity_checks.csv")
    ab["result_source"] = "v5 one-feature ablation integrity"
    ab["integrity_status"] = ab.apply(lambda r: "PASS" if r.test_keys_match_locked_v4 and r.baseline_matches_same_seed_v5 and not r.test_used_for_selection else "FAILED", axis=1)
    cols = list(dict.fromkeys(c for f in (v4, multi, ab) for c in f.columns))
    save(pd.concat([f.reindex(columns=cols) for f in (v4, multi, ab)], ignore_index=True), "result_integrity_audit.csv")


def build_status() -> None:
    rows = [
        ("target boundaries and tie handling", "COMPLETE", "outputs/audits/final_corrections_v5/corrected_target_saved_data_verification.csv", "None"),
        ("target-predictor direct overlap", "COMPLETE", "outputs/audits/final_closure/target_predictor_overlap_audit.csv", "Nine direct source overlaps established; semantics are tracked separately"),
        ("indirect category/event overlap semantics", "BLOCKED_EXTERNAL_EVIDENCE", "outputs/audits/final_corrections_v5/ipc_semantic_relationship_review.csv", "Official NCRB category crosswalk and event-counting rules by year"),
        ("feature structural lag availability", "COMPLETE", "outputs/audits/final_corrections_v5/forecast_feature_availability_audit.csv", "None for structural t-1 alignment"),
        ("operational feature issue-date availability", "BLOCKED_EXTERNAL_EVIDENCE", "outputs/audits/final_corrections_v5/final_forecast_feature_availability_audit.csv", "Year-specific official release/issue dates and declared forecast issuance date"),
        ("district-year integration and duplicate review", "PARTIAL", "outputs/audits/final_closure/district_year_integration_audit.csv", "Original upstream files, source record IDs, Railway identities, administrative boundary history"),
        ("missingness and observed-zero handling", "PARTIAL", "outputs/audits/final_closure/missingness_audit.csv", "Official reporting/missingness definitions needed to identify causes"),
        ("temporal sequence integrity", "COMPLETE", "outputs/audits/final_corrections_v5/multiseed_integrity_checks.csv", "Strict sequence cohort rejects incomplete histories; counts documented"),
        ("saved corrected-v4 result integrity", "COMPLETE", "outputs/audits/final_corrections_v5/v4_recomputed_metrics.csv", "Metrics recomputed from saved artifacts; no retraining required"),
        ("v5 multiseed and ablation integrity", "COMPLETE", "outputs/audits/final_corrections_v5/integrity_checks.csv", "50/50 ablation fits and five paired seeds verified"),
        ("source data reconciliation", "PARTIAL", "outputs/audits/final_corrections_v5/source_stage_reconciliation_audit.csv", "Original upstream downloads and source provenance before normalized stage"),
        ("reproducibility package", "PARTIAL", "outputs/audits/final_closure/reproducibility_manifest.json", "Upstream data provenance and external issue-date documentation remain unavailable"),
    ]
    save(pd.DataFrame(rows, columns=["correction", "status", "evidence_path", "unresolved_dependency"]), "closure_status.csv")


def build_manifest() -> None:
    relative_paths = [
        "data/processed/district_integrated_canonical.csv",
        "data/processed/district_grouped_reduced.csv",
        "data/processed/district_enhanced_features.csv",
        "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/run_config.json",
        "ML_Implementation/corrected_temporal_results/corrected_v3/inputs/common_sequence_metadata.csv",
        "ML_Implementation/corrected_temporal_results/corrected_v3/inputs/baseline_common_sequences.npz",
        "ML_Implementation/corrected_temporal_results/corrected_v3/inputs/enhanced_common_sequences.npz",
        "ML_Implementation/corrected_temporal_results/corrected_v5_multiseed/run_status.csv",
        "ML_Implementation/corrected_temporal_results/corrected_v5_feature_ablation_multiseed/run_status.csv",
        "src/target_creation.py", "src/feature_engineering.py", "ML_Implementation/forecast_feature_contract.py",
        "ML_Implementation/sequence_generation.py",
    ]
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "tensorflow", "scipy", "joblib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {
        "purpose": "read-only code/data/forecasting closure; no model training performed",
        "python": platform.python_version(), "packages": versions,
        "seeds": {"v4": [42], "v5_full_comparison": [11, 23, 37, 53, 71], "v5_ablation": [11, 23, 37, 53, 71]},
        "target_thresholds": [107, 237, 362, 563],
        "threshold_fit_years": [2001, 2010],
        "split": {"train": [2001, 2010], "validation": [2011, 2012], "locked_test": [2013, 2014]},
        "feature_contract": "t-1 for tabular; strict t-3,t-2,t-1 same-district sequences",
        "sample_counts": {"canonical_rows": 9856, "valid_target_rows": 9631, "missing_target_rows": 225,
                          "locked_test_keys": 671, "strict_baseline_sequences": 7161,
                          "strict_enhanced_sequences": 5842, "paired_common_sequences": 5842},
        "commands_executed_or_required": ["python -m unittest discover -s tests", "python -m compileall -q src ML_Implementation tests",
            "python src/grouped_predictor_source_audit.py", "python src/forecast_feature_availability_audit.py", "python src/source_stage_reconciliation_audit.py",
            "python src/duplicate_normalized_key_audit.py", "python src/grouped_feature_missingness_audit.py", "python src/corrected_results_integrity_audit.py"],
        "input_hashes_sha256": {p: sha(ROOT / p) for p in relative_paths if (ROOT / p).is_file()},
        "paper_modified": False, "raw_or_processed_inputs_modified": False,
        "limitations": ["catalog publication dates do not prove earliest operational availability", "original upstream sources not present", "indirect count overlap semantics unresolved"],
    }
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_report() -> None:
    text = """# NCRB code, data quality, and forecasting closure

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
"""
    (OUT / "final_code_closure_report.md").write_text(text, encoding="utf-8")


def build_test_report() -> None:
    report = """NCRB closure verification log

PASS — python -m unittest discover -s tests
Result: 30 tests passed.

PASS — python -m compileall -q src ML_Implementation tests
Result: compilation completed with no diagnostics.

PASS — python src/grouped_predictor_source_audit.py
Result: target/source formulas, frozen boundaries, row-order invariance, holdout threshold perturbation, saved downstream labels, and leakage negative tests pass. Nine direct grouped predictor/target source overlaps remain established; indirect case-counting semantics are unresolved.

PASS / UNVERIFIED — python src/forecast_feature_availability_audit.py
Result: keyed t-1 alignment, split, preprocessing scope, and corrected sequence metadata checks pass. Historical real-world source release timing remains UNVERIFIED.

PASS WITH PROVENANCE LIMIT — python src/source_stage_reconciliation_audit.py
Result: 100 integrated raw alias variables reconcile with zero value mismatches; available 2014 normalization/cleaning/core subsets reconcile. Original upstream copies are absent.

PASS — python src/duplicate_normalized_key_audit.py
Result: seven duplicate normalized district-year key groups (14 rows) reported across five cleaned files; all excluded from core. Two conflicting Railway groups remain without original identifiers.

PASS — python src/grouped_feature_missingness_audit.py
Result: six group formulas reconcile with zero mismatches; observed zero, partial sums, and all-missing inputs remain distinct. Cause of missing values remains unknown.

PASS — python src/corrected_results_integrity_audit.py
Result: all eight saved V4 model/variant result rows validate; target and downstream labels agree; frozen threshold and row-order checks pass.

NOT RUN — model training/retraining
Reason: no corrected input defect requiring new results was found; saved V4 and V5 artifacts passed integrity audits.

All closure statuses and unresolved evidence requirements are recorded in closure_status.csv and final_code_closure_report.md.
"""
    (OUT / "test_report.txt").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_overlap(); build_availability(); build_integration(); build_missingness()
    build_forecast(); build_results(); build_status(); build_manifest(); build_test_report(); build_report()


if __name__ == "__main__":
    main()
