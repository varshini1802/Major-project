"""Recompute probability-based multiclass AUC metrics for corrected-v3 models.

Writes to a new supplemental directory and leaves the original v3 outputs intact.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3"
OUT = BASE / "supplemental_metrics_auc"
KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TARGET = "TARGET_VIOLENT_LEVEL"
TEST = [2013, 2014]
CLASSES = np.arange(1, 6)


def macro_metrics(y, prob):
    if not np.array_equal(np.unique(y), CLASSES):
        raise ValueError("The locked test cohort must contain each of the five classes")
    per_auc = roc_auc_score(y, prob, labels=CLASSES, multi_class="ovr", average=None)
    per_ap = [average_precision_score((y == c).astype(int), prob[:, c-1]) for c in CLASSES]
    return float(np.mean(per_auc)), float(np.mean(per_ap)), per_auc, per_ap


def assert_predictions(pred_path, meta):
    pred = pd.read_csv(pred_path)
    keys = pred[KEYS + [TARGET]].astype(str).reset_index(drop=True)
    expected = meta[KEYS + [TARGET]].astype(str).reset_index(drop=True)
    if not keys.equals(expected):
        raise AssertionError(f"Saved prediction keys/labels differ: {pred_path}")
    return pred


def main():
    if (OUT / "corrected_v3_auc_metrics_all.csv").exists():
        raise FileExistsError(f"Refusing to overwrite completed supplemental metrics in {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # Tabular MLP and majority classifiers: use their saved fitted pipelines.
    for variant in ("baseline", "enhanced"):
        matrix = pd.read_csv(BASE / "inputs" / f"{variant}_forecast_matrix.csv")
        te = matrix[matrix.YEAR.isin(TEST)].reset_index(drop=True)
        result = pd.read_csv(BASE / "tabular/test_results.csv")
        for model_name in ("MLP", "Most frequent baseline"):
            row = result[(result.variant == variant) & (result.model == model_name)].iloc[0]
            model_dir = model_name.lower().replace(" ", "_")
            folder = BASE / "tabular" / variant / model_dir
            saved = assert_predictions(folder / "test_predictions.csv", te)
            estimator = joblib.load(folder / "fitted_pipeline.joblib")
            probs = estimator.predict_proba(te[[c for c in row.feature_columns.split("|")]])
            classes = estimator.classes_.astype(int)
            aligned = np.zeros((len(te), len(CLASSES)))
            for idx, cls in enumerate(classes): aligned[:, cls-1] = probs[:, idx]
            roc, ap, per_roc, per_ap = macro_metrics(te[TARGET].to_numpy(dtype=int), aligned)
            rows.append({"experiment": f"corrected_v3_tabular_{variant}", "model": model_name,
                         "test_rows": len(te), "macro_ovr_roc_auc": roc,
                         "macro_ovr_average_precision": ap,
                         "per_class_roc_auc": "|".join(map(str, per_roc)),
                         "per_class_average_precision": "|".join(map(str, per_ap)),
                         "probability_source": "saved fitted pipeline predict_proba; test predictions key/label alignment verified"})

    # Sequence selected networks use their saved scaler and common-cohort arrays.
    import tensorflow as tf
    meta = pd.read_csv(BASE / "inputs/common_sequence_metadata.csv")
    meta.YEAR = meta.YEAR.astype(int); meta[TARGET] = meta[TARGET].astype(int)
    test_meta = meta[meta.YEAR.isin(TEST)].reset_index(drop=True)
    for variant, model_name, model_dir in (("baseline", "GRU", "gru"), ("enhanced", "LSTM", "lstm")):
        archive = np.load(BASE / f"inputs/{variant}_common_sequences.npz")
        X = archive["X"].astype(np.float32)
        Xte = X[meta.YEAR.isin(TEST).to_numpy()]
        folder = BASE / "common_sequence_models" / variant / model_dir
        saved = assert_predictions(folder / "test_predictions.csv", test_meta)
        estimator = tf.keras.models.load_model(BASE / "common_sequence_models" / variant / f"{model_dir}_model.keras")
        scaler_cfg = json.loads((BASE / "common_sequence_models" / variant / "scaler.json").read_text(encoding="utf-8"))
        mean = np.asarray(scaler_cfg["mean"]); scale = np.asarray(scaler_cfg["scale"])
        Xscaled = ((Xte - mean) / scale).astype(np.float32)
        probs = estimator.predict(Xscaled, verbose=0)
        roc, ap, per_roc, per_ap = macro_metrics(test_meta[TARGET].to_numpy(dtype=int), probs)
        rows.append({"experiment": f"corrected_v3_common_sequence_{variant}", "model": model_name,
                     "test_rows": len(test_meta), "macro_ovr_roc_auc": roc,
                     "macro_ovr_average_precision": ap,
                     "per_class_roc_auc": "|".join(map(str, per_roc)),
                     "per_class_average_precision": "|".join(map(str, per_ap)),
                     "probability_source": "saved Keras model and saved scaler; test predictions key/label alignment verified"})
        # Majority sequence scores are reconstructed as one-hot hard-label
        # probabilities because the legacy majority output saved labels only.
        majority_folder = BASE / "common_sequence_models" / variant / "most_frequent_baseline"
        majority_pred = assert_predictions(majority_folder / "test_predictions.csv", test_meta)
        hard_probs = np.eye(len(CLASSES))[majority_pred.PREDICTED_TARGET.to_numpy(dtype=int) - 1]
        roc, ap, per_roc, per_ap = macro_metrics(test_meta[TARGET].to_numpy(dtype=int), hard_probs)
        rows.append({"experiment": f"corrected_v3_common_sequence_{variant}", "model": "Most frequent baseline",
                     "test_rows": len(test_meta), "macro_ovr_roc_auc": roc,
                     "macro_ovr_average_precision": ap,
                     "per_class_roc_auc": "|".join(map(str, per_roc)),
                     "per_class_average_precision": "|".join(map(str, per_ap)),
                     "probability_source": "one-hot probabilities derived from saved hard predictions; discrimination is tied-score baseline only"})
    pd.DataFrame(rows).to_csv(OUT / "corrected_v3_auc_metrics_all.csv", index=False)
    (OUT / "metric_convention_all.json").write_text(json.dumps({
        "roc_auc": "macro one-vs-rest multiclass ROC-AUC over classes 1..5",
        "pr_auc": "macro unweighted mean of per-class one-vs-rest average precision (AP), not trapezoidal PR curve area",
        "test_years": TEST, "test_used_for_selection": False,
        "note": "This supplemental post-hoc metric calculation uses already selected and saved v3 models; it does not retrain or alter v3 model selection. Majority sequence AUC/AP use one-hot probabilities derived from saved hard labels, so they are tie-score baselines."}, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__": main()
