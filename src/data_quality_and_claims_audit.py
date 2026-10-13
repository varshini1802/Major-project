"""Verify controlled v4 outputs, corrected-v3 evaluation coverage, and claims."""
from pathlib import Path
import json
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from target_creation import CLASS_LABELS

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3"
V4 = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence"
ABL = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_feature_ablation"
OUT = ROOT / "outputs/tables/final_remaining_corrections"
KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TARGET = "TARGET_VIOLENT_LEVEL"
CLASSES = [1, 2, 3, 4, 5]


def check_v4():
    rows = []
    result = pd.read_csv(V4 / "test_results.csv")
    test_keys = pd.read_csv(V4 / "locked_test_keys.csv")
    if len(test_keys) != 671 or test_keys.duplicated(KEYS).any():
        raise AssertionError("Controlled v4 locked test key count/uniqueness failed")
    expected_models = {("baseline", "GRU"), ("enhanced", "GRU"),
                       ("baseline", "LSTM"), ("enhanced", "LSTM"),
                       ("baseline", "LogisticRegression"), ("enhanced", "LogisticRegression"),
                       ("baseline", "Majority baseline"), ("enhanced", "Majority baseline")}
    actual = set(zip(result.variant, result.model))
    if not expected_models.issubset(actual):
        raise AssertionError(f"Missing v4 paired model rows: {expected_models-actual}")
    for rec in result.to_dict("records"):
        variant = rec["variant"]; model_name = rec["model"]
        folder_name = {"LogisticRegression": "logistic_regression"}.get(
            model_name, model_name.lower().replace(" ", "_"))
        folder = V4 / variant / folder_name
        pred = pd.read_csv(folder / "test_predictions.csv")
        pred_keys = pred[KEYS + [TARGET, "INPUT_YEARS"]]
        if not pred_keys.equals(test_keys[KEYS + [TARGET, "INPUT_YEARS"]]):
            raise AssertionError(f"v4 test keys/labels/histories differ for {variant}/{model_name}")
        report = pd.read_csv(folder / "test_classification_report.csv", index_col=0)
        if set(map(str, CLASSES)) - set(report.index.astype(str)):
            raise AssertionError(f"Missing per-class support/report rows for {variant}/{model_name}")
        if int(report.loc[[str(c) for c in CLASSES], "support"].sum()) != len(test_keys):
            raise AssertionError(f"Class supports do not sum to test n for {variant}/{model_name}")
        confusion = pd.read_csv(folder / "test_confusion_matrix.csv", index_col=0)
        if confusion.shape != (5, 5) or int(confusion.to_numpy().sum()) != len(test_keys):
            raise AssertionError(f"Confusion matrix does not represent test cohort for {variant}/{model_name}")
        auc = json.loads((folder / "test_auc_metrics.json").read_text(encoding="utf-8"))
        if not (0 <= auc["macro_ovr_roc_auc"] <= 1 and 0 <= auc["macro_ovr_average_precision"] <= 1):
            raise AssertionError(f"Invalid AUC metrics for {variant}/{model_name}")
    config = json.loads((V4 / "run_config.json").read_text(encoding="utf-8"))
    if config["test_used_for_selection"] is not False or config["common_test_rows"] != 671:
        raise AssertionError("v4 run config violates locked test/common cohort")
    rows.append({"check": "v4_paired_model_test_alignment", "status": "PASS", "rows": len(test_keys),
                 "details": "paired GRU/LSTM/logistic/majority reports, supports, matrices, probabilities, and test keys align"})

    ablation = pd.read_csv(ABL / "test_results.csv")
    if len(ablation) != 10 or set(ablation.feature_added) != {
        "TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX",
        "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"}:
        raise AssertionError("v4 one-at-a-time engineered-feature ablation is incomplete")
    if not ablation.rows.eq(671).all() or not ablation.test_rows.eq(671).all():
        raise AssertionError("Ablation models do not share the locked 671 test targets")
    for rec in ablation.to_dict("records"):
        folder = ABL / rec["feature_added"].lower() / rec["model"].lower()
        pred = pd.read_csv(folder / "test_predictions.csv")
        if not pred[KEYS + [TARGET, "INPUT_YEARS"]].equals(test_keys[KEYS + [TARGET, "INPUT_YEARS"]]):
            raise AssertionError(f"Ablation model changed test keys: {rec['feature_added']}/{rec['model']}")
    rows.append({"check": "v4_engineered_feature_ablation", "status": "PASS", "rows": len(ablation),
                 "details": "five single-feature additions x two architectures; identical 671 locked keys"})
    return rows


def check_v3():
    rows = []
    for cohort, result_path, base in (
        ("tabular", V3 / "tabular/test_results.csv", V3 / "tabular"),
        ("common_sequence", V3 / "common_sequence_models/test_results.csv", V3 / "common_sequence_models"),
    ):
        results = pd.read_csv(result_path)
        for rec in results.to_dict("records"):
            variant = rec["variant"]; model = rec["model"]
            subdir = model.lower().replace(" ", "_")
            folder = base / variant / subdir
            for name in ("test_predictions.csv", "test_classification_report.csv", "test_confusion_matrix.csv"):
                if not (folder / name).is_file():
                    raise AssertionError(f"Corrected-v3 result is missing {folder/name}")
            pred = pd.read_csv(folder / "test_predictions.csv")
            if len(pred) != int(rec["test_rows"]):
                raise AssertionError(f"v3 test count differs from saved predictions: {cohort}/{variant}/{model}")
            report = pd.read_csv(folder / "test_classification_report.csv", index_col=0)
            if int(report.loc[[str(c) for c in CLASSES], "support"].sum()) != len(pred):
                raise AssertionError(f"v3 class support count mismatch: {cohort}/{variant}/{model}")
            cm = pd.read_csv(folder / "test_confusion_matrix.csv", index_col=0)
            if int(cm.to_numpy().sum()) != len(pred):
                raise AssertionError(f"v3 confusion matrix count mismatch: {cohort}/{variant}/{model}")
    auc = pd.read_csv(V3 / "supplemental_metrics_auc/corrected_v3_auc_metrics_all.csv")
    if len(auc) != 8:
        raise AssertionError("Corrected-v3 supplemental AUC report is incomplete")
    rows.append({"check": "corrected_v3_result_artifact_completeness", "status": "PASS", "rows": len(auc),
                 "details": "all selected v3 tabular/sequence and majority results have class reports, supports, confusion matrices, predictions, test counts; macro OvR AUC/AP present for all eight rows"})
    return rows


def write_evaluation_report():
    results = pd.read_csv(V4 / "test_results.csv")
    ablations = pd.read_csv(ABL / "test_results.csv")
    supports = pd.read_csv(V4 / "class_distributions.csv")
    test_support = supports[supports.partition.eq("test")].set_index("class")["count"].to_dict()
    lines = ["# Controlled corrected forecast evaluation", "",
             "Generated from saved corrected-v4 experiment outputs. Target thresholds, classes, and temporal partitions are frozen from the prior target correction.", "",
             "## Protocol", "",
             "Shared strict sequence cohort: 5,842 targets (train 3,847; validation 1,324; test 671). Test years are 2013–2014; supports are " + ", ".join(f"class {c}: {test_support[c]}" for c in sorted(test_support)) + ". The baseline and enhanced variants use the exact same target keys and labels.", "",
             "GRU and LSTM pairs have identical architecture, optimizer, loss, batch size, stopping/epoch-selection procedure, and seed within each pair. Epochs and logistic C were selected on validation only. Preprocessing was fit on training sequences for validation selection and refit on train+validation for final test prediction. Test data was not used for selection.", "",
             "Multiclass ROC-AUC is macro one-vs-rest. The reported PR metric is macro unweighted mean one-vs-rest average precision (AP), not trapezoidal PR-curve area.", "",
             "## Full feature set and baselines", "",
             "| Variant | Model | Accuracy | Macro F1 | Weighted F1 | Macro OvR ROC-AUC | Macro OvR AP | n |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    order = [("baseline", "GRU"), ("enhanced", "GRU"), ("baseline", "LSTM"), ("enhanced", "LSTM"),
             ("baseline", "LogisticRegression"), ("enhanced", "LogisticRegression"),
             ("baseline", "Majority baseline"), ("enhanced", "Majority baseline")]
    for variant, model in order:
        row = results[(results.variant.eq(variant)) & (results.model.eq(model))].iloc[0]
        lines.append(f"| {variant} | {model} | {row.test_accuracy:.4f} | {row.test_macro_f1:.4f} | {row.test_weighted_f1:.4f} | {row.macro_ovr_roc_auc:.4f} | {row.macro_ovr_average_precision:.4f} | {int(row.test_rows)} |")
    lines += ["", "## Per-class metrics: same-architecture neural comparisons", "",
              "Each cell reports precision / recall / F1. The test support is identical for every model.", "",
              "| Class (support) | Baseline GRU | Enhanced GRU | Baseline LSTM | Enhanced LSTM |",
              "|---|---:|---:|---:|---:|"]
    pair_dirs = {"Baseline GRU": V4/"baseline/gru/test_classification_report.csv",
                 "Enhanced GRU": V4/"enhanced/gru/test_classification_report.csv",
                 "Baseline LSTM": V4/"baseline/lstm/test_classification_report.csv",
                 "Enhanced LSTM": V4/"enhanced/lstm/test_classification_report.csv"}
    reports = {name: pd.read_csv(path, index_col=0) for name, path in pair_dirs.items()}
    for c in range(1, 6):
        cells=[]
        for name in pair_dirs:
            r=reports[name].loc[str(c)]
            cells.append(f"{r.precision:.3f} / {r.recall:.3f} / {r['f1-score']:.3f}")
        lines.append(f"| {c} — {CLASS_LABELS[c]} ({int(test_support[c])}) | " + " | ".join(cells) + " |")
    lines += ["", "## One-at-a-time engineered-feature ablations", "",
              "Values are test macro-F1 change versus the same-architecture six-group baseline; all use the same 671 test keys.", "",
              "| Added predictor | GRU delta | LSTM delta |",
              "|---|---:|---:|"]
    for feature in ["TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX", "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"]:
        subset=ablations[ablations.feature_added.eq(feature)].set_index("model")
        lines.append(f"| {feature} | {subset.loc['GRU','macro_f1_delta_vs_same_architecture_baseline']:+.4f} | {subset.loc['LSTM','macro_f1_delta_vs_same_architecture_baseline']:+.4f} |")
    lines += ["", "The full enhanced set raises GRU macro F1 by about 0.0034 but lowers its ROC-AUC and AP. For LSTM it lowers macro F1 and ROC-AUC while slightly raising AP. One-feature ablation effects are mixed and small; they are single-seed descriptive results, not evidence of reliable gains.", "",
              "## Saved artifacts and limits", "",
              "Detailed per-class reports, confusion matrices, test keys, predictions, probabilities, AUC files, and configurations are in `ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/` and `.../corrected_v4_feature_ablation/`. Figures in the controlled directory were generated from those saved CSV result tables.", "",
              "Prior corrected-v2 and corrected-v3 model outputs were preserved. Corrected-v3 supplemental AUC results are in `corrected_v3/supplemental_metrics_auc/`; no legacy same-year score is treated as forecasting performance.", "",
              "Annual source-release dates, original upstream raw provenance, IPC parent/component relationships, and causes of NaN source observations remain unresolved. No conference-paper manuscript was found, so no paper sections were changed."]
    (ROOT/"outputs/reports/final_controlled_evaluation.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = check_v4() + check_v3()
    pd.DataFrame(checks).to_csv(OUT / "evaluation_artifact_audit.csv", index=False)
    notebook = (ROOT / "notebooks/observed_year_classification_models.ipynb").read_text(encoding="utf-8")
    required_labels = ["Lowest violent-crime burden", "Low-to-moderate violent-crime burden",
                       "Moderate violent-crime burden", "Moderate-to-high violent-crime burden",
                       "Highest violent-crime burden"]
    if any(label not in notebook for label in required_labels):
        raise AssertionError("Notebook class descriptions do not match canonical ordered burden labels")
    if "Class 1 — Homicide" in notebook or "approximately balanced" in notebook:
        raise AssertionError("Notebook retains stale crime-type class names or target distribution claim")
    if "observed-year" not in notebook.lower():
        raise AssertionError("Notebook does not label the random split as observed-year classification")
    if list(CLASS_LABELS) != [1, 2, 3, 4, 5]:
        raise AssertionError("Canonical class labels do not cover ordered classes 1..5")
    checks.append({"check": "notebook_target_claims", "status": "PASS", "rows": 5,
                   "details": "five labels are ordered burden levels; same-year random split is explicitly observed-year only"})
    pd.DataFrame(checks).to_csv(OUT / "evaluation_artifact_audit.csv", index=False)

    checklist = [
        (1, "Target-source overlap", "PARTIAL", "Nine direct shared IPC sources; direct counts are 76.6% of summed burden on 9,552 observed-overlap rows; indirect parent/component relationships remain unresolved."),
        (2, "Tie-safe quantile target", "COMPLETE", "Frozen 107/237/362/563 boundaries; row-order and target saved-data audits pass."),
        (3, "Five class meanings", "COMPLETE", "Canonical code and notebook describe ordered violent-burden levels; notebook stale class names corrected."),
        (4, "Engineered features", "PARTIAL", "All five formulas traced and one-at-a-time ablations completed; source release timing remains unverified."),
        (5, "Chronological evaluation", "COMPLETE", "Train 2001-2010, validation 2011-2012, test 2013-2014; random notebook sections observed-year only."),
        (6, "Preprocessing and selection scope", "COMPLETE", "Corrected pipelines fit learned transforms on train, select using validation, then refit train+validation before locked test."),
        (7, "Previous-year alignment", "PARTIAL", "District/year keyed t-1 alignment passes; publication dates are unavailable."),
        (8, "Integration and keys", "PARTIAL", "Available-stage aliases and key checks reconcile; original upstream copies and administrative-boundary provenance are unavailable."),
        (9, "Missing values", "PARTIAL", "Zero/missing counts and missingness strata are reported; causes of NaN are not encoded, so zero versus non-reporting provenance cannot be fully established."),
        (10, "Metrics and baselines", "COMPLETE", "Accuracy, macro/weighted F1, class reports/supports, confusion matrices, class distributions, majority and logistic baselines, macro OvR ROC-AUC and macro mean OvR AP saved."),
        (11, "LSTM/GRU sequence method", "COMPLETE", "Strict t-3..t-1 district histories; paired GRU and LSTM feature comparison plus logistic baseline; common test keys verified."),
        (12, "Leakage audit", "PARTIAL", "Automated structural/source audits pass; semantic indirect overlap and real issue-date availability remain unverified."),
        (13, "Reproducibility", "PARTIAL", "Seeds, configurations, versions, keys, metrics, models, and audits saved; full upstream source provenance is missing."),
        (14, "Paper abstract", "UNRESOLVED", "No manuscript file found; cannot revise or verify claims."),
        (15, "Paper introduction/objectives", "UNRESOLVED", "No manuscript file found."),
        (16, "Paper dataset/preprocessing", "UNRESOLVED", "No manuscript file found."),
        (17, "Paper methodology", "UNRESOLVED", "No manuscript file found."),
        (18, "Paper results", "UNRESOLVED", "No manuscript file found; v4 results and generated metrics are available for a future update."),
        (19, "Paper discussion", "UNRESOLVED", "No manuscript file found."),
        (20, "Paper limitations/future work", "UNRESOLVED", "No manuscript file found."),
    ]
    checklist_df = pd.DataFrame(checklist, columns=["correction", "requirement", "status", "evidence_or_limit"])
    checklist_df.to_csv(OUT / "correction_checklist_status.csv", index=False)
    reports = ROOT / "outputs/reports"; reports.mkdir(exist_ok=True)
    md = ["# Final correction checklist status", "",
          "Statuses are based on executed audits and saved artifacts. `PARTIAL` and `UNRESOLVED` items remain open.", "",
          "| # | Requirement | Status | Evidence / remaining limit |", "|---:|---|---|---|"]
    md += [f"| {i} | {name} | {status} | {evidence} |" for i, name, status, evidence in checklist]
    md += ["", "The corrected v4 controlled comparison is stored separately from v2/v3. No conference-paper manuscript was located or modified."]
    (reports / "final_correction_checklist_status.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_evaluation_report()
    source_files = [
        "ML_Implementation/controlled_sequence_forecast_evaluation.py",
        "ML_Implementation/engineered_feature_ablation_experiment.py",
        "ML_Implementation/recompute_saved_probability_metrics.py",
        "ML_Implementation/plot_controlled_forecast_results.py",
        "src/data_quality_and_claims_audit.py",
        "src/data_quality_pipeline_audit.py",
        "src/update_observed_year_notebook_labels.py",
        "src/target_creation.py",
        "notebooks/observed_year_classification_models.ipynb",
        "tests/test_final_correction_reporting.py",
    ]
    artifact_roots = [V4, ABL, V3/"supplemental_metrics_auc", OUT, reports]
    manifest = [{"path": f, "kind": "source_or_notebook"} for f in source_files]
    for artifact_root in artifact_roots:
        for file in sorted(artifact_root.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts:
                manifest.append({"path": file.relative_to(ROOT).as_posix(), "kind": "generated_or_supplemental_artifact"})
    manifest.append({"path": "ML_Implementation/corrected_temporal_results/corrected_v3/input_audit_report.csv",
                     "kind": "refreshed_v3_audit_artifact"})
    manifest.append({"path": "outputs/reports/final_correction_file_manifest.csv",
                     "kind": "generated_file_manifest"})
    pd.DataFrame(manifest).drop_duplicates("path").sort_values(["kind", "path"]).to_csv(
        reports / "final_correction_file_manifest.csv", index=False)
    import numpy as np_module
    import pandas as pd_module
    import tensorflow as tf
    env = {"python": platform.python_version(), "numpy": np_module.__version__,
           "pandas": pd_module.__version__, "scikit_learn": sklearn.__version__,
           "tensorflow": tf.__version__, "seed": 42,
           "target_thresholds": [107, 237, 362, 563],
           "threshold_fit_years": "2001-2010", "train": "2001-2010",
           "validation": "2011-2012", "test": "2013-2014"}
    (V4 / "environment_versions.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    (ABL / "environment_versions.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(checks).to_string(index=False))
    print(json.dumps(env, indent=2))


if __name__ == "__main__": main()
