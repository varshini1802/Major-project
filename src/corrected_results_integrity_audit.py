"""Recompute final correction audits without mutating protected data/results.

Writes only to outputs/audits/final_corrections_v5. Metrics are recalculated
from the saved test labels, predictions and probabilities in corrected v4.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audits/final_corrections_v5"
V4 = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence"
KEYS = ["YEAR", "STATE", "UNIT_NAME"]
CLASSES = np.arange(1, 6)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalized_keys(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["YEAR"] = pd.to_numeric(x.YEAR, errors="raise").astype(int)
    for c in KEYS[1:]:
        x[c] = x[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    return x


def dataset_audits() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [p for p in (ROOT / "data/processed").rglob("*.csv")]
    manifest, key_rows = [], []
    for path in sorted(paths):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:  # inventory still records unreadable artifacts
            manifest.append({"path": str(path.relative_to(ROOT)), "read_status": f"ERROR: {exc}"})
            continue
        has_keys = all(k in frame for k in KEYS)
        key_dup = "not_applicable"
        unique_years = ""
        unique_states = unique_units = ""
        if has_keys:
            norm = normalized_keys(frame)
            key_dup = int(norm.duplicated(KEYS).sum())
            unique_years = "|".join(map(str, sorted(norm.YEAR.dropna().unique())))
            unique_states = int(norm.STATE.nunique(dropna=True))
            unique_units = int(norm.UNIT_NAME.nunique(dropna=True))
            key_rows.append({"path": str(path.relative_to(ROOT)), "rows": len(frame),
                             "key_fields": "|".join(KEYS), "duplicate_normalized_keys": key_dup,
                             "unique_years": unique_years, "unique_states": unique_states,
                             "unique_units": unique_units,
                             "assessment": "PASS" if key_dup == 0 else "FAIL"})
        manifest.append({"path": str(path.relative_to(ROOT)), "file_name": path.name,
                         "available_stage": path.parent.name, "rows": len(frame), "columns": len(frame.columns),
                         "key_fields": "|".join(KEYS) if has_keys else "not present",
                         "duplicate_normalized_keys": key_dup, "unique_years": unique_years,
                         "unique_states": unique_states, "unique_units": unique_units,
                         "sha256": sha256(path), "file_size_bytes": path.stat().st_size,
                         "transformation_script": "See repository notebooks/src pipeline; original upstream files absent",
                         "provenance_level": "available processed file only; upstream provenance not established"})
    return pd.DataFrame(manifest), pd.DataFrame(key_rows)


def component_missingness() -> pd.DataFrame:
    formulas = pd.read_csv(ROOT / "outputs/tables/group_predictor_formula_recovery.csv")
    source_cols = sorted({c.strip() for raw in formulas.source_columns for c in raw.split("|")})
    stage_paths = [ROOT / "data/processed/district_integrated_raw.csv",
                   ROOT / "data/processed/district_integrated_canonical.csv",
                   ROOT / "data/processed/district_integrated_final.csv",
                   ROOT / "data/processed/district_grouped_reduced.csv"]
    rows = []
    for path in stage_paths:
        frame = pd.read_csv(path, low_memory=False)
        for col in source_cols:
            if col not in frame:
                rows.append({"stage": str(path.relative_to(ROOT)), "source_column": col,
                             "present": False, "rows": len(frame), "missing": "", "observed_zero": "",
                             "positive": "", "missing_reason": "not inferable from available stage"})
                continue
            values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            rows.append({"stage": str(path.relative_to(ROOT)), "source_column": col,
                         "present": True, "rows": len(frame), "missing": int(values.isna().sum()),
                         "observed_zero": int(values.eq(0).sum()), "positive": int(values.gt(0).sum()),
                         "missing_reason": "not encoded; this count does not distinguish non-reporting from unavailable data"})
    mapping = pd.read_csv(ROOT / "outputs/tables/canonical_feature_mapping.csv")
    grouped_sources = sorted({c.strip() for raw in formulas.source_columns for c in raw.split("|")})
    def token(value: str) -> str:
        return re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")
    for stage_name in ("district_normalized", "district_cleaned", "district_core"):
        folder = ROOT / "data/processed" / stage_name
        ipc_files = sorted(folder.glob("01_District_wise_crimes_committed_IPC_*.csv"))
        for path in ipc_files:
            frame = pd.read_csv(path, low_memory=False)
            by_token = {token(c): c for c in frame.columns}
            for final_col in grouped_sources:
                suffix = final_col.removeprefix("IPC_")
                rows_for_source = mapping[(mapping.crime_group == "IPC") &
                                          (mapping.canonical_feature == suffix)]
                aliases = [str(x).removeprefix("IPC_") for x in rows_for_source.source_feature]
                aliases.extend([suffix])
                source_col = next((by_token[token(x)] for x in aliases if token(x) in by_token), None)
                if source_col is None:
                    rows.append({"stage": str(path.relative_to(ROOT)), "source_column": final_col,
                                 "present": False, "rows": len(frame), "missing": "", "observed_zero": "",
                                 "positive": "", "missing_reason": "No mechanically normalized alias found; provenance unresolved"})
                    continue
                values = pd.to_numeric(frame[source_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                rows.append({"stage": str(path.relative_to(ROOT)), "source_column": final_col,
                             "source_alias": source_col, "present": True, "rows": len(frame),
                             "missing": int(values.isna().sum()), "observed_zero": int(values.eq(0).sum()),
                             "positive": int(values.gt(0).sum()),
                             "missing_reason": "Observed at this processed stage only; cause not inferred"})
    return pd.DataFrame(rows)


def recompute_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results, per_class_rows, confusion_rows = [], [], []
    for variant in ("baseline", "enhanced"):
        for model in ("gru", "lstm", "logistic_regression", "majority_baseline"):
            folder = V4 / variant / model
            pred_df = pd.read_csv(folder / "test_predictions.csv")
            prob_df = pd.read_csv(folder / "test_probabilities.csv")
            join_keys = KEYS + ["TARGET_VIOLENT_LEVEL"]
            a = normalized_keys(pred_df[join_keys + ["PREDICTED_TARGET"]])
            b = normalized_keys(prob_df[join_keys + [f"PROB_CLASS_{c}" for c in CLASSES]])
            if a.duplicated(KEYS).any() or b.duplicated(KEYS).any():
                raise AssertionError(f"Duplicate prediction keys: {variant}/{model}")
            merged = a.merge(b, on=join_keys, how="outer", validate="one_to_one", indicator=True)
            if not merged._merge.eq("both").all():
                raise AssertionError(f"Prediction/probability key disagreement: {variant}/{model}")
            y = merged.TARGET_VIOLENT_LEVEL.astype(int).to_numpy()
            pred = merged.PREDICTED_TARGET.astype(int).to_numpy()
            prob = merged[[f"PROB_CLASS_{c}" for c in CLASSES]].to_numpy(dtype=float)
            if not np.isfinite(prob).all() or (prob < -1e-9).any() or (prob > 1 + 1e-9).any():
                raise AssertionError(f"Invalid probabilities: {variant}/{model}")
            row_sum_error = float(np.max(np.abs(prob.sum(axis=1) - 1)))
            if row_sum_error > 1e-5:
                raise AssertionError(f"Probabilities do not sum to one: {variant}/{model}: {row_sum_error}")
            if not np.array_equal(np.unique(y), CLASSES):
                raise AssertionError(f"Test lacks a required class: {variant}/{model}")
            if model != "majority_baseline" and not np.array_equal(pred, CLASSES[np.argmax(prob, axis=1)]):
                raise AssertionError(f"Hard predictions do not match probability argmax: {variant}/{model}")
            per_class = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
            cm = confusion_matrix(y, pred, labels=CLASSES)
            ovr_auc = roc_auc_score(y, prob, labels=CLASSES, multi_class="ovr", average="macro")
            ovr_ap = float(np.mean([average_precision_score(y == c, prob[:, c - 1]) for c in CLASSES]))
            record = {"variant": variant, "model": model, "rows": len(y),
                      "accuracy": accuracy_score(y, pred),
                      "macro_precision": precision_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
                      "macro_recall": recall_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
                      "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
                      "weighted_precision": precision_score(y, pred, labels=CLASSES, average="weighted", zero_division=0),
                      "weighted_recall": recall_score(y, pred, labels=CLASSES, average="weighted", zero_division=0),
                      "weighted_f1": f1_score(y, pred, labels=CLASSES, average="weighted", zero_division=0),
                      "macro_ovr_roc_auc": ovr_auc,
                      "macro_mean_ovr_average_precision": ovr_ap,
                      "max_probability_row_sum_error": row_sum_error,
                      "probability_columns_class_order": "1|2|3|4|5",
                      "metric_validation": "PASS"}
            saved_report = pd.read_csv(folder / "test_classification_report.csv", index_col=0)
            for c in CLASSES:
                for metric_name in ("precision", "recall", "f1-score", "support"):
                    if abs(float(saved_report.loc[str(c), metric_name]) - float(per_class[str(c)][metric_name])) > 1e-10:
                        record["metric_validation"] = "FAIL"
                        record["metric_validation_detail"] = f"saved class report differs at class {c}/{metric_name}"
            saved_cm = pd.read_csv(folder / "test_confusion_matrix.csv", index_col=0)
            if not np.array_equal(saved_cm.to_numpy(dtype=int), cm):
                record["metric_validation"] = "FAIL"
                record["metric_validation_detail"] = "saved confusion matrix differs"
            saved_auc = json.loads((folder / "test_auc_metrics.json").read_text(encoding="utf-8"))
            if (abs(float(saved_auc["macro_ovr_roc_auc"]) - ovr_auc) > 1e-10 or
                    abs(float(saved_auc["macro_ovr_average_precision"]) - ovr_ap) > 1e-10):
                record["metric_validation"] = "FAIL"
                record["metric_validation_detail"] = "saved probability-based AUC/AP differs"
            prior = pd.read_csv(V4 / "test_results.csv")
            saved = prior[(prior.variant == variant) & (prior.model.str.lower().replace({"logisticregression":"logistic_regression", "majority baseline":"majority_baseline"}) == model)]
            # Explicit model aliasing avoids relying on display labels.
            aliases = {"gru":"GRU", "lstm":"LSTM", "logistic_regression":"LogisticRegression", "majority_baseline":"Majority baseline"}
            saved = prior[(prior.variant == variant) & (prior.model == aliases[model])]
            if len(saved) != 1:
                raise AssertionError(f"Expected one saved metrics row for {variant}/{model}")
            for actual, column in ((record["accuracy"], "test_accuracy"), (record["macro_f1"], "test_macro_f1"),
                                   (record["weighted_f1"], "test_weighted_f1"), (record["macro_ovr_roc_auc"], "macro_ovr_roc_auc"),
                                   (record["macro_mean_ovr_average_precision"], "macro_ovr_average_precision")):
                if abs(float(actual) - float(saved.iloc[0][column])) > 1e-8:
                    record["metric_validation"] = "FAIL"
                    record[f"difference_{column}"] = float(actual) - float(saved.iloc[0][column])
            results.append(record)
            for c in CLASSES:
                row = per_class[str(c)]
                per_class_rows.append({"variant": variant, "model": model, "class": c,
                                       "precision": row["precision"], "recall": row["recall"],
                                       "f1": row["f1-score"], "support": int(row["support"])})
            for i, c1 in enumerate(CLASSES):
                for j, c2 in enumerate(CLASSES):
                    confusion_rows.append({"variant": variant, "model": model,
                                           "actual_class": c1, "predicted_class": c2,
                                           "count": int(cm[i, j])})
    locked = normalized_keys(pd.read_csv(V4 / "locked_test_keys.csv"))
    expected = locked[KEYS + ["TARGET_VIOLENT_LEVEL"]].sort_values(KEYS).reset_index(drop=True)
    for variant in ("baseline", "enhanced"):
        for model in ("gru", "lstm", "logistic_regression", "majority_baseline"):
            predictions = normalized_keys(pd.read_csv(V4 / variant / model / "test_predictions.csv"))
            actual = predictions[KEYS + ["TARGET_VIOLENT_LEVEL"]].sort_values(KEYS).reset_index(drop=True)
            if not actual.equals(expected):
                raise AssertionError(f"{variant}/{model} differs from the locked test keys or labels")
    return pd.DataFrame(results), pd.DataFrame(per_class_rows), pd.DataFrame(confusion_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, key_report = dataset_audits()
    manifest.to_csv(OUT / "processed_dataset_manifest.csv", index=False)
    key_report.to_csv(OUT / "district_year_key_validation.csv", index=False)

    formulas = pd.read_csv(ROOT / "outputs/tables/group_predictor_formula_recovery.csv")
    target_sources = pd.read_csv(ROOT / "outputs/tables/target_source_column_mapping.csv")
    target_col = next((c for c in target_sources if "source" in c.lower() or "column" in c.lower()), None)
    target_set = set(target_sources[target_col].dropna().astype(str)) if target_col else set()
    lineage = []
    for row in formulas.itertuples(index=False):
        cols = [x.strip() for x in row.source_columns.split("|")]
        for col in cols:
            lineage.append({"feature": row.predictor_group, "source_column": col,
                            "source_file_or_module": "src/grouped_dataset_integrity_audit.py; outputs/tables/crime_group_mapping.csv; validated against data/processed/district_grouped_reduced.csv",
                            "formula": "row sum across listed source columns; pandas sum(min_count=1); no further transformation",
                            "source_year": "same-year t in observed-year analysis; shifted t-1 or t-3..t-1 for forecasting",
                            "direct_target_source_overlap": col in target_set,
                            "indirect_parent_component_relationship": "UNRESOLVED unless explicitly documented in local relationship audit",
                            "evidence": str(row.formula_evidence),
                            "forecast_decision": "same-year excluded; historical lag eligible structurally",
                            "availability_limit": "actual release date unavailable"})
    for feature, source, formula in [
        ("TOTAL_GROUPED_CRIME", "six GROUP_* columns", "sum(min_count=1)"),
        ("ACTIVE_CRIME_GROUPS", "six GROUP_* columns", "count values > 0; all-missing remains missing"),
        ("CRIME_DIVERSITY_INDEX", "six GROUP_* columns", "Shannon entropy; partial missing treated as zero; all-missing remains missing; zero total => 0"),
        ("PREV_YEAR_TOTAL_CRIME", "TOTAL_GROUPED_CRIME", "exact district-keyed prior calendar-year lookup; missing prior row => missing"),
        ("YOY_TOTAL_CRIME_CHANGE", "TOTAL_GROUPED_CRIME and PREV_YEAR_TOTAL_CRIME", "absolute current minus previous count")]:
        lineage.append({"feature": feature, "source_column": source,
                        "source_file_or_module": "src/feature_engineering.py and ML_Implementation/forecast_feature_contract.py",
                        "formula": formula, "source_year": "t in engineered row; t-1 row used for forecast target t",
                        "direct_target_source_overlap": "inherited through grouped IPC sources",
                        "indirect_parent_component_relationship": "UNRESOLVED",
                        "evidence": "implemented feature-engineering functions and keyed lag contract",
                        "forecast_decision": "retain only when transformed row is sourced from <=t-1",
                        "availability_limit": "actual publication/release date unavailable"})
    pd.DataFrame(lineage).to_csv(OUT / "feature_source_lineage.csv", index=False)

    overlap = pd.read_csv(ROOT / "outputs/tables/predictor_source_overlap_mapping.csv")
    overlap.to_csv(OUT / "direct_indirect_source_overlap.csv", index=False)
    formula_data = pd.read_csv(ROOT / "data/processed/district_integrated_final.csv", low_memory=False)
    component_rows = []
    for row in formulas.itertuples(index=False):
        cols = [c.strip() for c in row.source_columns.split("|")]
        values = formula_data[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        observed = values.notna().sum(axis=1)
        total = values.sum(axis=1, min_count=1)
        component_rows.append({"group": row.predictor_group, "components": len(cols),
                               "all_components_observed_rows": int(observed.eq(len(cols)).sum()),
                               "some_components_missing_rows": int(observed.gt(0).sum() & observed.lt(len(cols)).sum()) if False else int((observed.gt(0) & observed.lt(len(cols))).sum()),
                               "all_components_missing_rows": int(observed.eq(0).sum()),
                               "observed_zero_total_rows": int(total.eq(0).sum()),
                               "positive_total_rows": int(total.gt(0).sum()),
                               "partial_sum_rows": int((observed.gt(0) & observed.lt(len(cols))).sum()),
                               "semantics": "partial sums are not complete totals; no all-missing-to-zero conversion"})
    pd.DataFrame(component_rows).to_csv(OUT / "group_component_missingness.csv", index=False)
    component_missingness().to_csv(OUT / "ipc_component_missingness_by_stage.csv", index=False)
    canonical_ipc = pd.read_csv(ROOT / "data/processed/district_integrated_final.csv", low_memory=False)
    year_component_rows = []
    for component in sorted({c.strip() for raw in formulas.source_columns for c in raw.split("|")}):
        for year, year_frame in canonical_ipc.groupby("YEAR", sort=True):
            if component not in year_frame:
                year_component_rows.append({"source_column": component, "year": int(year), "rows": len(year_frame),
                                            "observed": "", "observed_zero": "", "missing": len(year_frame),
                                            "column_present": False})
                continue
            vals = pd.to_numeric(year_frame[component], errors="coerce").replace([np.inf, -np.inf], np.nan)
            year_component_rows.append({"source_column": component, "year": int(year), "rows": len(year_frame),
                                        "observed": int(vals.notna().sum()), "observed_zero": int(vals.eq(0).sum()),
                                        "missing": int(vals.isna().sum()), "column_present": True})
    pd.DataFrame(year_component_rows).to_csv(OUT / "ipc_group_component_availability_by_year.csv", index=False)

    schema_notes = [
        ("IPC_ATTEMPT_TO_MURDER", "2014", "Attempt to commit Murder", "Alternative heading present; not coalesced because category crosswalk is absent."),
        ("IPC_BURGLARY", "2014", "Criminal Trespass/Burglary; Criminal Trespass or Burglary; House Trespass & House Breaking", "Several alternative headings present; they were not combined without an official crosswalk."),
        ("IPC_COUNTERFEITING", "2001-2013", "COUNTERFIETING in normalized 2001-2012 and 2013 IPC files", "Observed spelling variant has values, but canonical IPC_COUNTERFEITING is missing in final integration for these years; no automatic remapping performed."),
        ("IPC_SEXUAL_HARASSMENT", "2001-2013", "No matching IPC field in the available IPC normalized file schemas", "Schema coverage differs; possible corresponding fields in other NCRB tables are not assumed equivalent."),
        ("IPC_STALKING", "2001-2013", "No matching IPC field in the available IPC normalized file schemas", "Schema coverage differs; possible corresponding fields in other NCRB tables are not assumed equivalent."),
        ("IPC_UNLAWFUL_ASSEMBLY", "2001-2013", "No matching IPC field in the available IPC normalized file schemas", "Schema coverage differs; no equivalent field is inferred."),
        ("IPC_VOYEURISM", "2001-2013", "No matching IPC field in the available IPC normalized file schemas", "Schema coverage differs; possible corresponding fields in other NCRB tables are not assumed equivalent."),
    ]
    pd.DataFrame(schema_notes, columns=["canonical_group_source", "period", "available_source_schema_evidence",
                                        "audit_conclusion"]).to_csv(OUT / "ipc_group_source_schema_discrepancies.csv", index=False)

    for name in ("source_stage_reconciliation_audit.csv", "forecast_feature_availability_audit.csv",
                 "corrected_target_saved_data_verification.csv", "group_predictor_formula_recovery.csv",
                 "target_predictor_source_overlap.csv", "direct_overlap_burden_summary.csv",
                 "target_thresholds.csv", "final_target_distribution.csv",
                 "integration_reconciliation_summary.csv", "group_missingness_overall.csv",
                 "group_total_completeness.csv", "ipc_component_missingness_by_group.csv",
                 "pipeline_stage_counts.csv", "sequence_stage_counts.csv"):
        source = ROOT / "outputs/tables" / name
        if source.exists():
            (OUT / name).write_bytes(source.read_bytes())
    for name in ("integration_reconciliation_summary.csv", "group_missingness_overall.csv",
                 "group_total_completeness.csv", "ipc_component_missingness_by_group.csv",
                 "pipeline_stage_counts.csv", "sequence_stage_counts.csv"):
        source = ROOT / "outputs/tables/final_remaining_corrections" / name
        if source.exists():
            (OUT / name).write_bytes(source.read_bytes())
    availability_path = ROOT / "outputs/tables/forecast_feature_availability_audit.csv"
    if availability_path.exists():
        availability = pd.read_csv(availability_path)
        audit = pd.DataFrame({
            "audit_id": [f"AVAIL-{i+1:02d}" for i in range(len(availability))],
            "requirement": availability.check,
            "test_performed": availability.availability_rule,
            "evidence_file": "outputs/tables/forecast_feature_availability_audit.csv; see corrected artifacts named in each row",
            "result": availability.status,
            "explanation": availability.availability_rule,
            "remediation_or_limit": availability.status.map({
                "UNVERIFIED": "Obtain forecast-origin date and original release/publication records.",
                "PASS_WITH_PROVENANCE_LIMIT": "Obtain original upstream source files for end-to-end provenance.",
                "PASS": "No further structural remediation indicated by this check.",
                "FAIL": "Repair corresponding implementation and rerun audit."
            }).fillna("Review the underlying evidence and status."),
        })
        audit.to_csv(OUT / "final_forecast_feature_availability_audit.csv", index=False)

    metrics, per_class, confusions = recompute_metrics()
    metrics.to_csv(OUT / "v4_recomputed_metrics.csv", index=False)
    per_class.to_csv(OUT / "v4_recomputed_per_class_metrics.csv", index=False)
    confusions.to_csv(OUT / "v4_recomputed_confusion_matrices.csv", index=False)
    ablation = pd.read_csv(ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_feature_ablation/test_results.csv")
    ablation.to_csv(OUT / "v4_ablation_cohort_recheck.csv", index=False)
    new_ablation = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v5_feature_ablation_multiseed"
    for name in ("ablation_report.md", "paired_difference_summary.csv", "integrity_checks.csv",
                 "run_config.json", "input_output_hashes.json"):
        source = new_ablation / name
        if source.exists():
            (OUT / name).write_bytes(source.read_bytes())

    env_file = V4 / "environment_versions.json"
    env = json.loads(env_file.read_text(encoding="utf-8"))
    hash_paths = [ROOT / "data/processed/district_integrated_canonical.csv",
                  ROOT / "data/processed/district_grouped_reduced.csv",
                  ROOT / "data/processed/district_enhanced_features.csv",
                  ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/run_config.json",
                  ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3/inputs/common_sequence_metadata.csv",
                  ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3/inputs/baseline_common_sequences.npz",
                  ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3/inputs/enhanced_common_sequences.npz",
                  ROOT / "ML_Implementation/run_multiseed_forecasting_experiment.py",
                  ROOT / "ML_Implementation/run_multiseed_feature_ablation_experiment.py",
                  ROOT / "ML_Implementation/corrected_temporal_results/corrected_v5_feature_ablation_multiseed/run_config.json",
                  ROOT / "ML_Implementation/corrected_temporal_results/corrected_v5_feature_ablation_multiseed/paired_difference_summary.csv"]
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "tensorflow", "scipy", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed in current interpreter"
    multiseed_status_path = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v5_multiseed/run_status.csv"
    multiseed_status = pd.read_csv(multiseed_status_path) if multiseed_status_path.exists() else pd.DataFrame()
    complete_seeds = (multiseed_status.loc[multiseed_status.status.eq("COMPLETE"), "seed"].astype(int).tolist()
                      if not multiseed_status.empty else [])
    ablation_root = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v5_feature_ablation_multiseed"
    ablation_status_path = ablation_root / "run_status.csv"
    ablation_status = pd.read_csv(ablation_status_path) if ablation_status_path.exists() else pd.DataFrame()
    ablation_complete = int(ablation_status.status.eq("COMPLETE").sum()) if not ablation_status.empty else 0
    ablation_expected = 50
    reproducibility = {"python_current": platform.python_version(), "packages_current": packages,
                       "saved_v4_environment": env,
                       "sha256_inputs": {str(p.relative_to(ROOT)): sha256(p) for p in hash_paths if p.exists()},
                       "target_thresholds": [107, 237, 362, 563],
                       "training_years": list(range(2001, 2011)), "validation_years": [2011, 2012],
                       "test_years": [2013, 2014], "saved_v4_seed": 42,
                       "recomputed_from_saved_predictions": True, "model_suite_retrained": False,
                       "multi_seed_experiment": {"predeclared_seeds": [11, 23, 37, 53, 71],
                                                   "completed_seeds": complete_seeds,
                                                   "status_file": "ML_Implementation/corrected_temporal_results/corrected_v5_multiseed/run_status.csv",
                                                   "runner": "ML_Implementation/run_multiseed_forecasting_experiment.py"},
                       "multi_seed_feature_ablations": {"completed_fits": ablation_complete,
                                                        "expected_fits": ablation_expected,
                                                        "status_file": "ML_Implementation/corrected_temporal_results/corrected_v5_feature_ablation_multiseed/run_status.csv",
                                                        "runner": "ML_Implementation/run_multiseed_feature_ablation_experiment.py"},
                       "commands": ["python src/grouped_predictor_source_audit.py", "python src/forecast_feature_availability_audit.py",
                                   "python ML_Implementation/run_multiseed_feature_ablation_experiment.py",
                                   "python src/duplicate_normalized_key_audit.py", "python src/grouped_feature_missingness_audit.py",
                                   "python src/corrected_results_integrity_audit.py", "python -m unittest discover -s tests",
                                   "python -m compileall -q src ML_Implementation tests"]}
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(reproducibility, indent=2) + "\n", encoding="utf-8")

    statuses = [
        (1, "Target-source overlap", "PARTIAL", "Nine direct overlaps are established and some statutory relationships are documented; NCRB category crosswalks and event-level duplicate-counting semantics remain unresolved.", "outputs/audits/final_corrections_v5/direct_indirect_source_overlap.csv; ipc_semantic_relationship_review.csv; ipc_semantic_relationship_review.md; outputs/tables/target_predictor_source_overlap.csv"),
        (4, "Engineered features and robustness", "COMPLETE" if ablation_complete == ablation_expected else "PARTIAL", f"Full-feature multi-seed paired models and {ablation_complete}/{ablation_expected} one-feature/model/seed ablation fits are recorded; effects remain descriptive, not causal.", "ML_Implementation/corrected_temporal_results/corrected_v5_multiseed/run_status.csv; outputs/audits/final_corrections_v5/multiseed_robustness_report.md; ML_Implementation/corrected_temporal_results/corrected_v5_feature_ablation_multiseed/run_status.csv; ML_Implementation/corrected_temporal_results/corrected_v5_feature_ablation_multiseed/paired_difference_summary.csv; ML_Implementation/corrected_temporal_results/corrected_v4_feature_ablation/test_results.csv"),
        (8, "Integration and provenance", "PARTIAL", "Available-stage reconciliation and duplicate disposition are reported; original upstream files and administrative-boundary provenance unavailable.", "outputs/audits/final_corrections_v5/processed_dataset_manifest.csv; district_year_key_validation.csv; cleaned_duplicate_key_findings.md; outputs/audits/final_corrections_v5/source_stage_reconciliation_audit.csv"),
        (9, "Missing values and observed zeros", "PARTIAL", "Source-to-stored component counts reconcile; semantic causes of missingness are not encoded and schema mapping gaps remain.", "outputs/audits/final_corrections_v5/group_missingness_source_to_stored_reconciliation.csv; ipc_group_component_availability_by_year.csv; ipc_group_source_schema_discrepancies.csv; ipc_component_missingness_by_stage.csv"),
        (10, "Metrics and baselines", "COMPLETE" if metrics.metric_validation.eq("PASS").all() else "PARTIAL", "Recomputed from saved labels/predictions/probabilities; all eight v4 rows agree; no v4 retraining.", "outputs/audits/final_corrections_v5/v4_recomputed_metrics.csv; v4_recomputed_per_class_metrics.csv; v4_recomputed_confusion_matrices.csv; ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/test_results.csv"),
        (12, "Leakage and forecast availability", "PARTIAL", "Structural keyed lag and sequence checks pass; real issue-date availability and indirect target semantics remain unverified.", "outputs/audits/final_corrections_v5/final_forecast_feature_availability_audit.csv; outputs/audits/final_corrections_v5/leakage_negative_test_results.csv; outputs/audits/final_corrections_v5/multiseed_integrity_checks.csv; tests/test_forecast_feature_contract.py; tests/test_sequence_generation.py"),
        (13, "Reproducibility and consistency", "PARTIAL", "Five predeclared comparison seeds, package versions, input hashes, configurations, logs and integrity checks are recorded; upstream provenance and administrative-boundary evidence remain open.", "outputs/audits/final_corrections_v5/reproducibility_manifest.json; outputs/audits/final_corrections_v5/multiseed_reproducibility_manifest.json; ML_Implementation/corrected_temporal_results/corrected_v5_multiseed/run_status.csv; ML_Implementation/corrected_temporal_results/corrected_v5_multiseed/logs/; ML_Implementation/corrected_temporal_results/corrected_v5_feature_ablation_multiseed/run_config.json"),
    ]
    pd.DataFrame(statuses, columns=["correction", "requirement", "status", "evidence_or_limit", "evidence_paths"]).to_csv(OUT / "correction_status.csv", index=False)

    summary = ["# Final corrections v5 audit", "",
               "This is a non-mutating audit of current inputs and saved corrected-v4 results.",
               "No model suite was retrained and no protected v2/v3/v4 result, target backup, or raw input was overwritten.", "",
               "## Recomputed metric check", "",
               f"Saved-v4 result rows checked: {len(metrics)}; metric rows agreeing within 1e-8: {int(metrics.metric_validation.eq('PASS').sum())}.",
               "Probability columns are class ordered 1..5 and validated for finite values and row sums. Macro mean one-vs-rest AP is an average of class-wise average precision, not trapezoidal PR-AUC.", "",
               "## Evidence limits", "",
               "The available correction source correction(2).md is preserved at outputs/reports/source_correction_checklist_correction2_verbatim.md and mapped requirement-by-requirement in outputs/reports/original_correction_checklist_verification.md. The earlier unsuffixed correction.md was not found, so exact version identity is unverified.",
               f"Completed multi-seed runs: {', '.join(map(str, complete_seeds)) if complete_seeds else 'none'}; see ML_Implementation/corrected_temporal_results/corrected_v5_multiseed/run_status.csv. The original v4 seed-42 results remain unchanged.",
               f"The five-seed one-feature/model ablation runner records {ablation_complete}/{ablation_expected} expected fits; see its run_status.csv and paired_difference_summary.csv. Seed spread is descriptive and not sampling uncertainty.",
               "The local repository lacks original upstream raw copies, verified administrative boundary provenance, annual publication/release dates, and documented missingness causes.",
               "Component schema coverage differs by year. In particular, the normalized 2001–2013 IPC files contain the source spelling COUNTERFIETING while the final canonical IPC_COUNTERFEITING field is empty for those years; alternative 2014 headings for attempt-to-murder/burglary were not equated without an official crosswalk. See ipc_group_source_schema_discrepancies.csv.",
               "Indirect IPC hierarchy/overlapping offence semantics remain unresolved unless explicitly supported by local documentation.",
               "No original conference manuscript or template was found. A new project draft was created at outputs/reports/conference_paper_draft.md; it requires author and venue review.", "",
               "## Outputs", "",
               "See the CSV/JSON files in this directory: processed dataset manifest, key audit, source lineage, overlap mapping, component missingness, v4 metric recomputation, per-class metrics, confusion matrices, v4 ablation review, reproducibility manifest, and correction status.", ""]
    (OUT / "README.md").write_text("\n".join(summary), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Output: {OUT}")
    if not metrics.metric_validation.eq("PASS").all():
        raise AssertionError("One or more recomputed v4 metrics disagree with saved summary values")


if __name__ == "__main__":
    main()
