"""Evaluate baseline/enhanced sequence models on identical eligible keys."""
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3"
IN = BASE / "inputs"
OUT = BASE / "common_sequence_models"
KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TARGET = "TARGET_VIOLENT_LEVEL"
TRAIN = list(range(2001, 2011)); VAL = [2011, 2012]; TEST = [2013, 2014]
SEED = 42


def scaler_transform(train, *others):
    n, steps, nf = train.shape
    scaler = StandardScaler().fit(train.reshape(-1, nf))
    result = [scaler.transform(train.reshape(-1, nf)).reshape(n, steps, nf).astype(np.float32)]
    for values in others:
        result.append(scaler.transform(values.reshape(-1, nf)).reshape(len(values), steps, nf).astype(np.float32))
    return result, scaler


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, LSTM, GRU

    OUT.mkdir(); random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
    meta = pd.read_csv(IN / "common_sequence_metadata.csv")
    meta["YEAR"] = meta.YEAR.astype(int); meta[TARGET] = meta[TARGET].astype(int)
    results = []; common_test_keys = None

    def model_factories(shape):
        def compile_with(layers):
            model = Sequential([Input(shape=shape), *layers, Dense(5, activation="softmax")])
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            return model
        return {
            "ANN": lambda: compile_with([Flatten(), Dense(128, activation="relu"), Dropout(.25), Dense(64, activation="relu")]),
            "LSTM": lambda: compile_with([LSTM(64), Dropout(.25), Dense(32, activation="relu")]),
            "GRU": lambda: compile_with([GRU(64), Dropout(.25), Dense(32, activation="relu")]),
        }

    for variant in ("baseline", "enhanced"):
        archive = np.load(IN / f"{variant}_common_sequences.npz")
        X = archive["X"].astype(np.float32); y = archive["y"].astype(int)
        if len(X) != len(meta) or not np.array_equal(y, meta[TARGET].to_numpy()):
            raise AssertionError(f"{variant} common sequence array/metadata labels differ")
        if not np.array_equal(archive["years"], meta.YEAR.to_numpy()):
            raise AssertionError(f"{variant} common sequence array/metadata years differ")
        if common_test_keys is None:
            common_test_keys = meta.loc[meta.YEAR.isin(TEST), KEYS].reset_index(drop=True)
        elif not common_test_keys.equals(meta.loc[meta.YEAR.isin(TEST), KEYS].reset_index(drop=True)):
            raise AssertionError("Baseline and enhanced do not have identical locked test keys")
        masks = [meta.YEAR.isin(years).to_numpy() for years in (TRAIN, VAL, TEST)]
        Xtr, Xv, Xte = (X[m] for m in masks); ytr, yv, yte = (y[m] for m in masks)
        (tr_scaled, val_scaled), _ = scaler_transform(Xtr, Xv)
        candidates = []
        factories = model_factories((X.shape[1], X.shape[2]))
        for name, factory in factories.items():
            tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
            model = factory()
            history = model.fit(tr_scaled, ytr - 1, validation_data=(val_scaled, yv - 1),
                                epochs=100, batch_size=64,
                                callbacks=[EarlyStopping(monitor="val_loss", patience=12,
                                                         restore_best_weights=True)], verbose=0)
            val_pred = np.argmax(model.predict(val_scaled, verbose=0), axis=1) + 1
            candidates.append((name, factory, int(np.argmin(history.history["val_loss"]) + 1),
                               f1_score(yv, val_pred, average="macro", zero_division=0)))
        dummy_val = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(ytr), 1)), ytr).predict(np.zeros((len(yv), 1)))
        candidates.append(("Most frequent baseline", None, 0,
                           f1_score(yv, dummy_val, average="macro", zero_division=0)))
        candidates.sort(key=lambda item: (-item[3], item[0]))
        variant_out = OUT / variant; variant_out.mkdir()
        pd.DataFrame([{"model": name, "best_epoch": epoch, "validation_macro_f1": score}
                      for name, _, epoch, score in candidates]).to_csv(variant_out / "validation_selection.csv", index=False)
        selected = candidates[0]
        fit_y = np.concatenate([ytr, yv]); test_rows = meta.loc[masks[2]].reset_index(drop=True)
        (fit_scaled, test_scaled), scaler = scaler_transform(np.concatenate([Xtr, Xv]), Xte)
        if selected[1] is None:
            pred_selected = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(fit_y), 1)), fit_y).predict(np.zeros((len(yte), 1)))
            selected_name = selected[0]
        else:
            selected_name, factory, epochs, val_score = selected
            tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
            model = factory(); model.fit(fit_scaled, fit_y - 1, epochs=max(1, epochs), batch_size=64, verbose=0)
            pred_selected = np.argmax(model.predict(test_scaled, verbose=0), axis=1) + 1
            model.save(variant_out / f"{selected_name.lower()}_model.keras")
            (variant_out / "scaler.json").write_text(json.dumps({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "fit_scope": "common-cohort train+validation after validation selection"}, indent=2) + "\n", encoding="utf-8")
        # After validation selection is frozen, evaluate the fixed majority baseline on the same test keys.
        base_pred = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(fit_y), 1)), fit_y).predict(np.zeros((len(yte), 1)))
        for name, score, pred in ((selected_name, selected[3], pred_selected),
                                  ("Most frequent baseline", next(c[3] for c in candidates if c[0] == "Most frequent baseline"), base_pred)):
            folder = variant_out / name.lower().replace(" ", "_"); folder.mkdir(exist_ok=True)
            pd.DataFrame(classification_report(yte, pred, labels=[1, 2, 3, 4, 5],
                                               output_dict=True, zero_division=0)).T.to_csv(folder / "test_classification_report.csv")
            pd.DataFrame(confusion_matrix(yte, pred, labels=[1, 2, 3, 4, 5]),
                         index=[f"actual_{i}" for i in range(1, 6)],
                         columns=[f"predicted_{i}" for i in range(1, 6)]).to_csv(folder / "test_confusion_matrix.csv")
            test_rows.assign(PREDICTED_TARGET=pred).to_csv(folder / "test_predictions.csv", index=False)
            results.append({"variant": variant, "model": name, "validation_macro_f1": score,
                            "train_rows": len(ytr), "validation_rows": len(yv), "test_rows": len(yte),
                            "test_accuracy": accuracy_score(yte, pred),
                            "test_macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
                            "test_weighted_f1": f1_score(yte, pred, average="weighted", zero_division=0)})
    pd.DataFrame(results).to_csv(OUT / "test_results.csv", index=False)
    config = {"comparison": "baseline and enhanced evaluated on exact shared valid target keys",
              "objective": "forecast future-year violent-crime burden", "train_years": TRAIN,
              "validation_years": VAL, "locked_test_years": TEST, "seed": SEED,
              "target_thresholds": [107, 237, 362, 563], "lookback": 3,
              "selection": "validation macro-F1; best epoch by validation loss; final refit on train+validation",
              "common_test_rows": len(common_test_keys)}
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
