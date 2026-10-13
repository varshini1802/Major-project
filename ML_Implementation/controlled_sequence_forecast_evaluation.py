"""Paired baseline/enhanced GRU and LSTM experiments on the v3 common cohort.

The locked test split is evaluated only after validation-based epoch/C selection.
Outputs are written to a new versioned directory and never overwrite v2/v3.
"""
from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3"
IN = BASE / "inputs"
OUT = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence"
KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TARGET = "TARGET_VIOLENT_LEVEL"
CLASSES = np.arange(1, 6)
TRAIN = list(range(2001, 2011))
VAL = [2011, 2012]
TEST = [2013, 2014]
SEED = 42


def scaled_sequences(train: np.ndarray, *others: np.ndarray):
    nf = train.shape[-1]
    scaler = StandardScaler().fit(train.reshape(-1, nf))
    scaled = [scaler.transform(train.reshape(-1, nf)).reshape(train.shape).astype(np.float32)]
    for values in others:
        scaled.append(scaler.transform(values.reshape(-1, nf)).reshape(values.shape).astype(np.float32))
    return scaled, scaler


def probability_frame(probabilities: np.ndarray, estimator_classes=None) -> np.ndarray:
    """Return columns in canonical class order 1..5, filling absent classes with 0."""
    result = np.zeros((len(probabilities), len(CLASSES)), dtype=float)
    if estimator_classes is None:  # Keras output indices map to class labels 1..5.
        if probabilities.shape[1] != len(CLASSES):
            raise ValueError("Neural output does not contain five class probabilities")
        return probabilities
    for j, cls in enumerate(estimator_classes):
        if int(cls) in CLASSES:
            result[:, int(cls) - 1] = probabilities[:, j]
    return result


def score_probabilities(y: np.ndarray, prob: np.ndarray) -> dict:
    present = np.isin(CLASSES, np.unique(y))
    if not present.all():
        raise ValueError(f"Test cohort lacks classes required for macro AUC: {CLASSES[~present].tolist()}")
    per_class_roc = roc_auc_score(y, prob, labels=CLASSES, multi_class="ovr", average=None)
    per_class_ap = [average_precision_score((y == c).astype(int), prob[:, c - 1]) for c in CLASSES]
    return {
        "macro_ovr_roc_auc": float(np.mean(per_class_roc)),
        "macro_ovr_average_precision": float(np.mean(per_class_ap)),
        "per_class_roc_auc": per_class_roc.tolist(),
        "per_class_average_precision": per_class_ap,
        "auc_convention": "macro one-vs-rest ROC-AUC; macro unweighted mean of five one-vs-rest average-precision scores",
    }


def save_evaluation(folder: Path, meta: pd.DataFrame, y: np.ndarray,
                    pred: np.ndarray, prob: np.ndarray) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(classification_report(
        y, pred, labels=CLASSES, output_dict=True, zero_division=0)).T
    report.to_csv(folder / "test_classification_report.csv")
    pd.DataFrame(confusion_matrix(y, pred, labels=CLASSES),
                 index=[f"actual_{c}" for c in CLASSES],
                 columns=[f"predicted_{c}" for c in CLASSES]).to_csv(folder / "test_confusion_matrix.csv")
    result = meta[KEYS + [TARGET, "INPUT_YEARS"]].copy()
    result["PREDICTED_TARGET"] = pred
    result.to_csv(folder / "test_predictions.csv", index=False)
    pframe = result[KEYS + [TARGET]].copy()
    for j, cls in enumerate(CLASSES):
        pframe[f"PROB_CLASS_{cls}"] = prob[:, j]
    pframe.to_csv(folder / "test_probabilities.csv", index=False)
    aucs = score_probabilities(y, prob)
    (folder / "test_auc_metrics.json").write_text(json.dumps(aucs, indent=2) + "\n", encoding="utf-8")
    return {
        "test_rows": int(len(y)), "test_accuracy": float(accuracy_score(y, pred)),
        "test_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_ovr_roc_auc": aucs["macro_ovr_roc_auc"],
        "macro_ovr_average_precision": aucs["macro_ovr_average_precision"],
    }


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing controlled experiment: {OUT}")
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import Dense, Dropout, GRU, Input, LSTM

    random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
    meta = pd.read_csv(IN / "common_sequence_metadata.csv")
    meta["YEAR"] = pd.to_numeric(meta.YEAR, errors="raise").astype(int)
    meta[TARGET] = pd.to_numeric(meta[TARGET], errors="raise").astype(int)
    if meta.duplicated(KEYS).any():
        raise AssertionError("Common sequence target keys are duplicated")
    masks = {"train": meta.YEAR.isin(TRAIN).to_numpy(),
             "validation": meta.YEAR.isin(VAL).to_numpy(),
             "test": meta.YEAR.isin(TEST).to_numpy()}
    if not all(mask.any() for mask in masks.values()):
        raise AssertionError("At least one chronological partition has no sequence samples")
    test_meta = meta.loc[masks["test"], KEYS + [TARGET, "INPUT_YEARS"]].reset_index(drop=True)
    results = []
    selection_rows = []
    distributions = []

    class BestValidationMacroF1(Callback):
        def __init__(self, xval, yval):
            super().__init__(); self.xval = xval; self.yval = yval
            self.best = -1.0; self.best_epoch = 0; self.best_weights = None; self.wait = 0
        def on_epoch_end(self, epoch, logs=None):
            p = np.argmax(self.model.predict(self.xval, verbose=0), axis=1) + 1
            value = f1_score(self.yval, p, average="macro", zero_division=0)
            if value > self.best:
                self.best = float(value); self.best_epoch = epoch + 1
                self.best_weights = self.model.get_weights()
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= 15:
                    self.model.stop_training = True

    OUT.mkdir(parents=True)
    for split, mask in masks.items():
        subset = meta.loc[mask]
        for cls in CLASSES:
            distributions.append({"partition": split, "class": int(cls),
                                  "count": int(subset[TARGET].eq(cls).sum()), "rows": len(subset)})
    test_meta.to_csv(OUT / "locked_test_keys.csv", index=False)

    for variant in ("baseline", "enhanced"):
        archive = np.load(IN / f"{variant}_common_sequences.npz")
        X = archive["X"].astype(np.float32)
        y = archive["y"].astype(int)
        if len(X) != len(meta) or not np.array_equal(y, meta[TARGET].to_numpy()):
            raise AssertionError(f"{variant}: array/target metadata mismatch")
        if not np.array_equal(archive["years"], meta.YEAR.to_numpy()):
            raise AssertionError(f"{variant}: array/year metadata mismatch")
        if not np.isfinite(X).all():
            raise AssertionError(f"{variant}: nonfinite input sequence")
        for other in ("baseline", "enhanced"):
            other_meta = pd.read_csv(IN / "common_sequence_metadata.csv")
            if not other_meta[KEYS + [TARGET, "INPUT_YEARS"]].equals(meta[KEYS + [TARGET, "INPUT_YEARS"]]):
                raise AssertionError(f"{variant}: common cohort differs from {other}")
        xtr, xv, xte = (X[masks[n]] for n in ("train", "validation", "test"))
        ytr, yv = (y[masks[n]] for n in ("train", "validation"))
        yte = y[masks["test"]]  # consumed only in the final evaluation stage below
        mte = meta.loc[masks["test"]].reset_index(drop=True)
        if not mte[KEYS + [TARGET, "INPUT_YEARS"]].equals(test_meta):
            raise AssertionError("Variants do not use the same locked test keys/labels/histories")
        (xtr_s, xv_s), _ = scaled_sequences(xtr, xv)
        factories = {
            "GRU": lambda shape: Sequential([Input(shape=shape), GRU(64), Dropout(.25),
                                               Dense(32, activation="relu"), Dense(5, activation="softmax")]),
            "LSTM": lambda shape: Sequential([Input(shape=shape), LSTM(64), Dropout(.25),
                                                Dense(32, activation="relu"), Dense(5, activation="softmax")]),
        }
        for architecture, factory in factories.items():
            tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
            model = factory((X.shape[1], X.shape[2]))
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            selector = BestValidationMacroF1(xv_s, yv)
            model.fit(xtr_s, ytr - 1, validation_data=(xv_s, yv - 1), epochs=100,
                      batch_size=64, callbacks=[selector], verbose=0)
            if selector.best_weights is None:
                raise RuntimeError(f"{variant}/{architecture}: validation selection did not produce weights")
            selection_rows.append({"variant": variant, "model": architecture,
                                   "selection_metric": "validation macro-F1",
                                   "best_epoch": selector.best_epoch,
                                   "validation_macro_f1": selector.best})
            fit_X = np.concatenate([xtr, xv]); fit_y = np.concatenate([ytr, yv])
            (fit_s, test_s), scaler = scaled_sequences(fit_X, xte)
            tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
            final_model = factory((X.shape[1], X.shape[2]))
            final_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            final_model.fit(fit_s, fit_y - 1, epochs=selector.best_epoch, batch_size=64, verbose=0)
            prob = final_model.predict(test_s, verbose=0)
            pred = np.argmax(prob, axis=1) + 1
            folder = OUT / variant / architecture.lower()
            folder.mkdir(parents=True)
            final_model.save(folder / f"{architecture.lower()}_model.keras")
            (folder / "scaler.json").write_text(json.dumps({
                "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "fit_scope": "train+validation after validation epoch selection"}, indent=2) + "\n", encoding="utf-8")
            metrics = save_evaluation(folder, mte, yte, pred, prob)
            results.append({"variant": variant, "model": architecture, "family": "neural",
                            "feature_count": int(X.shape[-1]), "feature_names": "|".join(archive["feature_names"].tolist()),
                            "seed": SEED, "train_rows": int(masks["train"].sum()),
                            "validation_rows": int(masks["validation"].sum()),
                            "validation_macro_f1": selector.best, "selected_epoch": selector.best_epoch,
                            **metrics})

        # Paired simple non-neural sequence baseline: multinomial logistic regression.
        flat_tr = xtr.reshape(len(xtr), -1); flat_v = xv.reshape(len(xv), -1)
        scaler = StandardScaler().fit(flat_tr)
        flat_tr_s, flat_v_s = scaler.transform(flat_tr), scaler.transform(flat_v)
        c_scores = []
        for c in (.01, .1, 1., 10.):
            candidate = LogisticRegression(C=c, max_iter=2000, random_state=SEED)
            candidate.fit(flat_tr_s, ytr)
            pred_val = candidate.predict(flat_v_s)
            c_scores.append((f1_score(yv, pred_val, average="macro", zero_division=0), c))
        best_val, best_c = max(c_scores, key=lambda pair: (pair[0], -pair[1]))
        selection_rows.append({"variant": variant, "model": "LogisticRegression",
                               "selection_metric": "validation macro-F1", "best_epoch": "",
                               "validation_macro_f1": best_val, "selected_C": best_c})
        fit_flat = np.concatenate([flat_tr, flat_v]); fit_y = np.concatenate([ytr, yv])
        final_scaler = StandardScaler().fit(fit_flat)
        final_model = LogisticRegression(C=best_c, max_iter=2000, random_state=SEED)
        final_model.fit(final_scaler.transform(fit_flat), fit_y)
        test_flat = xte.reshape(len(xte), -1)
        prob = probability_frame(final_model.predict_proba(final_scaler.transform(test_flat)), final_model.classes_)
        pred = CLASSES[np.argmax(prob, axis=1)]
        folder = OUT / variant / "logistic_regression"
        folder.mkdir(parents=True)
        pd.DataFrame({"mean": final_scaler.mean_, "scale": final_scaler.scale_}).to_csv(folder / "scaler.csv", index=False)
        import joblib
        joblib.dump(final_model, folder / "model.joblib")
        metrics = save_evaluation(folder, mte, yte, pred, prob)
        results.append({"variant": variant, "model": "LogisticRegression", "family": "non-neural",
                        "feature_count": int(X.shape[-1]), "feature_names": "|".join(archive["feature_names"].tolist()),
                        "seed": SEED, "train_rows": int(masks["train"].sum()),
                        "validation_rows": int(masks["validation"].sum()),
                        "validation_macro_f1": best_val, "selected_C": best_c, **metrics})

        # Prespecified majority baseline, trained on train+validation only.
        majority = int(pd.Series(fit_y).value_counts().sort_values(ascending=False).index[0])
        majority_pred = np.full(len(yte), majority, dtype=int)
        majority_prob = np.zeros((len(yte), len(CLASSES)), dtype=float)
        majority_prob[:, majority - 1] = 1.0
        folder = OUT / variant / "majority_baseline"
        metrics = save_evaluation(folder, mte, yte, majority_pred, majority_prob)
        results.append({"variant": variant, "model": "Majority baseline", "family": "non-neural",
                        "feature_count": 0, "feature_names": "none", "seed": SEED,
                        "train_rows": int(masks["train"].sum()), "validation_rows": int(masks["validation"].sum()),
                        "validation_macro_f1": float("nan"), "selected_C": "",
                        "majority_class": majority, **metrics})

    pd.DataFrame(results).to_csv(OUT / "test_results.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(OUT / "validation_selection.csv", index=False)
    pd.DataFrame(distributions).to_csv(OUT / "class_distributions.csv", index=False)
    config = {
        "objective": "controlled future-year burden forecast comparison on identical shared strict sequences",
        "target": TARGET, "thresholds": [107, 237, 362, 563], "threshold_fit_years": "2001-2010",
        "train_years": TRAIN, "validation_years": VAL, "locked_test_years": TEST,
        "common_test_rows": len(test_meta), "seed": SEED, "lookback": 3,
        "same_architecture_pairs": ["GRU baseline vs GRU enhanced", "LSTM baseline vs LSTM enhanced"],
        "neural_config": {"units": 64, "dropout": 0.25, "dense_units": 32, "optimizer": "Adam",
                          "loss": "sparse_categorical_crossentropy", "batch_size": 64, "max_epochs": 100,
                          "epoch_selection": "maximum validation macro-F1; stop after 15 epochs without improvement; final refit uses selected epoch on train+validation"},
        "neural_preprocessing": "feature-wise StandardScaler fit on training sequences for validation selection, then refit on train+validation for test",
        "logistic_config": {"C_candidates": [0.01, 0.1, 1.0, 10.0], "selection": "validation macro-F1",
                            "inputs": "flattened ordered 3-year sequence; standardizer fit on training only then refit train+validation"},
        "test_metrics": "accuracy, macro-F1, weighted-F1, per-class report/support, confusion matrix, macro OvR ROC-AUC, macro mean OvR average precision",
        "auc_definition": "5-class macro one-vs-rest ROC-AUC and unweighted mean of per-class one-vs-rest average precision; all five classes must occur in test",
        "test_used_for_selection": False,
    }
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
