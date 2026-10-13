"""Corrected 3-year sequence forecasting experiment with locked test years."""
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v2"
SEQ = BASE / "sequences"
OUT = BASE / "sequential_models"
TRAIN = list(range(2001, 2011))
VAL = [2011, 2012]
TEST = [2013, 2014]
SEED = 42


def scale_fit(train, other=None):
    n, steps, nf = train.shape
    scaler = StandardScaler().fit(train.reshape(-1, nf))
    transformed = scaler.transform(train.reshape(-1, nf)).reshape(n, steps, nf).astype(np.float32)
    if other is None:
        return transformed, scaler
    m = other.shape[0]
    o = scaler.transform(other.reshape(-1, nf)).reshape(m, steps, nf).astype(np.float32)
    return transformed, o, scaler


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite corrected model output: {OUT}")
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, LSTM, GRU
    except ImportError as exc:
        raise SystemExit("TensorFlow required for corrected sequence experiments") from exc
    OUT.mkdir(parents=True)
    random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
    results, distributions = [], []

    def builders(shape):
        def compile_model(layers):
            m = Sequential([Input(shape=shape), *layers, Dense(5, activation="softmax")])
            m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            return m
        return {
            "ANN": lambda: compile_model([Flatten(), Dense(128, activation="relu"), Dropout(.25), Dense(64, activation="relu")]),
            "LSTM": lambda: compile_model([LSTM(64), Dropout(.25), Dense(32, activation="relu")]),
            "GRU": lambda: compile_model([GRU(64), Dropout(.25), Dense(32, activation="relu")]),
        }

    for variant in ("baseline", "enhanced"):
        archive = np.load(SEQ / f"{variant}_sequences.npz")
        X = archive["X"].astype(np.float32)
        y = archive["y"].astype(int)
        years = archive["years"].astype(int)
        meta = pd.read_csv(SEQ / f"{variant}_sequence_metadata.csv")
        if len(X) != len(meta) or not np.array_equal(y, meta.TARGET_VIOLENT_LEVEL.to_numpy()):
            raise ValueError(f"{variant}: metadata/array target mismatch")
        masks = [np.isin(years, yrs) for yrs in (TRAIN, VAL, TEST)]
        if any(not m.any() for m in masks):
            raise ValueError(f"{variant}: empty temporal partition")
        Xtr, Xv, Xte = (X[m] for m in masks)
        ytr, yv, yte = (y[m] for m in masks)
        for part, values, labels in (("train", Xtr, ytr), ("validation", Xv, yv), ("test", Xte, yte)):
            for cls in range(1, 6):
                distributions.append({"dataset": variant, "partition": part, "class": cls,
                                      "count": int((labels == cls).sum()), "rows": len(labels)})
        tr_scaled, v_scaled, scaler_train = scale_fit(Xtr, Xv)
        candidates = []
        factories = builders((X.shape[1], X.shape[2]))
        for name, factory in factories.items():
            tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
            model = factory()
            history = model.fit(tr_scaled, ytr - 1, validation_data=(v_scaled, yv - 1),
                                epochs=100, batch_size=64,
                                callbacks=[EarlyStopping(monitor="val_loss", patience=12,
                                                         restore_best_weights=True)], verbose=0)
            vp = np.argmax(model.predict(v_scaled, verbose=0), axis=1) + 1
            best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
            candidates.append((name, factory, best_epoch,
                               f1_score(yv, vp, average="macro", zero_division=0)))
        # Include a training-only majority baseline in validation model selection.
        dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(ytr), 1)), ytr)
        dummy_val = dummy.predict(np.zeros((len(yv), 1)))
        candidates.append(("Most frequent baseline", None, 0,
                           f1_score(yv, dummy_val, average="macro", zero_division=0)))
        candidates.sort(key=lambda item: (-item[3], item[0]))
        chosen = candidates[0]
        name, factory, epochs, val_score = chosen
        folder = OUT / variant
        folder.mkdir()
        pd.DataFrame([{"model": n, "best_epoch": e, "validation_macro_f1": s}
                      for n, _, e, s in candidates]).to_csv(folder / "validation_model_selection.csv", index=False)
        if factory is None:
            final = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(ytr)+len(yv), 1)), np.concatenate([ytr, yv]))
            pred = final.predict(np.zeros((len(yte), 1)))
        else:
            # Refit selected preprocessing on train+validation after selection; fixed epoch count from validation.
            xfit = np.concatenate([Xtr, Xv]); yfit = np.concatenate([ytr, yv])
            xfit_scaled, xte_scaled, scaler_fit = scale_fit(xfit, Xte)
            tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
            final = factory()
            final.fit(xfit_scaled, yfit - 1, epochs=max(1, epochs), batch_size=64, verbose=0)
            pred = np.argmax(final.predict(xte_scaled, verbose=0), axis=1) + 1
            final.save(folder / f"{name.lower()}_model.keras")
            with open(folder / "scaler.json", "w", encoding="utf-8") as f:
                json.dump({"mean": scaler_fit.mean_.tolist(), "scale": scaler_fit.scale_.tolist(),
                           "fit_scope": "train+validation after validation-based model/epoch selection"}, f)
        test_meta = meta.iloc[np.flatnonzero(masks[2])].copy()
        test_meta["PREDICTED_TARGET"] = pred
        test_meta.to_csv(folder / "test_predictions.csv", index=False)
        pd.DataFrame(confusion_matrix(yte, pred, labels=[1, 2, 3, 4, 5]),
                     index=[f"actual_{i}" for i in range(1, 6)],
                     columns=[f"predicted_{i}" for i in range(1, 6)]).to_csv(folder / "test_confusion_matrix.csv")
        pd.DataFrame(classification_report(yte, pred, labels=[1, 2, 3, 4, 5],
                                           output_dict=True, zero_division=0)).T.to_csv(folder / "test_classification_report.csv")
        results.append({"dataset": variant, "selected_model": name, "validation_macro_f1": val_score,
                        "best_epoch": epochs, "train_rows": len(ytr), "validation_rows": len(yv),
                        "test_rows": len(yte), "test_accuracy": accuracy_score(yte, pred),
                        "test_macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
                        "test_weighted_f1": f1_score(yte, pred, average="weighted", zero_division=0)})
        # Evaluate the prespecified majority-class baseline after validation selection is frozen.
        base_final = DummyClassifier(strategy="most_frequent").fit(
            np.zeros((len(ytr) + len(yv), 1)), np.concatenate([ytr, yv]))
        base_pred = base_final.predict(np.zeros((len(yte), 1)))
        base_dir = folder / "most_frequent_baseline"
        base_dir.mkdir()
        pd.DataFrame(classification_report(yte, base_pred, labels=[1, 2, 3, 4, 5],
                                           output_dict=True, zero_division=0)).T.to_csv(base_dir / "test_classification_report.csv")
        pd.DataFrame(confusion_matrix(yte, base_pred, labels=[1, 2, 3, 4, 5]),
                     index=[f"actual_{i}" for i in range(1, 6)],
                     columns=[f"predicted_{i}" for i in range(1, 6)]).to_csv(base_dir / "test_confusion_matrix.csv")
        base_meta = meta.iloc[np.flatnonzero(masks[2])].copy()
        base_meta["PREDICTED_TARGET"] = base_pred
        base_meta.to_csv(base_dir / "test_predictions.csv", index=False)
        base_val = f1_score(yv, dummy_val, average="macro", zero_division=0)
        results.append({"dataset": variant, "selected_model": "Most frequent baseline",
                        "validation_macro_f1": base_val, "best_epoch": 0,
                        "train_rows": len(ytr), "validation_rows": len(yv), "test_rows": len(yte),
                        "test_accuracy": accuracy_score(yte, base_pred),
                        "test_macro_f1": f1_score(yte, base_pred, average="macro", zero_division=0),
                        "test_weighted_f1": f1_score(yte, base_pred, average="weighted", zero_division=0)})
    pd.DataFrame(results).to_csv(OUT / "test_results.csv", index=False)
    pd.DataFrame(distributions).to_csv(OUT / "class_distributions.csv", index=False)
    config = {"objective": "future-year violent-crime burden forecasting", "lookback": 3,
              "history": "t-3,t-2,t-1 exact same-district complete feature records",
              "train_years": TRAIN, "validation_years": VAL, "locked_test_years": TEST,
              "target_thresholds": [107, 237, 362, 563], "seed": SEED,
              "incomplete_sequences": "rejected; no padding or imputation",
              "selection": "validation macro-F1 and validation-loss best epoch; selected model refit on train+validation before locked test evaluation"}
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
