"""Retired legacy sequence runner; use strict corrected-v3 artifacts."""
if __name__ == "__main__":
    raise SystemExit(
        "This legacy sequence runner is disabled: it refits target cuts and used "
        "padded incomplete histories. Use common_cohort_sequence_evaluation.py."
    )

from pathlib import Path
import random
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# 1. CONFIGURATION AND PATHS
# ============================================================

SEED = 42
SEQUENCE_LENGTH = 3

random.seed(SEED)
np.random.seed(SEED)

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent

RAW_PATH = PROJECT_DIR / "data" / "processed" / "district_integrated_final.csv"
BASELINE_PATH = PROJECT_DIR / "data" / "processed" / "district_grouped_reduced.csv"
ENHANCED_PATH = PROJECT_DIR / "data" / "processed" / "district_enhanced_features.csv"

OUTPUT_DIR = ML_DIR / "temporal_deep_learning_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TARGET = "TARGET_VIOLENT_LEVEL"

# Three previous years are used as input.
# Target thresholds are learned from the training period only.
TRAIN_YEARS = list(range(2004, 2011))
VALIDATION_YEARS = [2011, 2012]
TEST_YEARS = [2013, 2014]

BASE_FEATURES = [
    "GROUP_HOMICIDE",
    "GROUP_SEXUAL_VIOLENCE",
    "GROUP_KIDNAPPING_ABDUCTION",
    "GROUP_ROBBERY_DACOITY",
    "GROUP_PUBLIC_VIOLENCE",
    "GROUP_PROPERTY_CRIME",
]

BURDEN_GROUPS = {
    "HOMICIDE": [
        "IPC_MURDER",
        "IPC_CULPABLE_HOMICIDE_NOT_AMOUNTING_TO_MURDER",
    ],
    "RIOTS": [
        "IPC_RIOTS",
        "IPC_RIOTS_COMMUNAL",
        "IPC_RIOTS_INDUSTRIAL",
        "IPC_RIOTS_POLITICAL",
        "IPC_RIOTS_CASTE_CONFLICT",
        "IPC_RIOTS_AGRARIAN",
        "IPC_RIOTS_STUDENTS",
        "IPC_RIOTS_SECTARIAN",
    ],
    "SEXUAL_VIOLENCE": [
        "IPC_RAPE",
        "IPC_OTHER_RAPE",
        "IPC_CUSTODIAL_RAPE",
        "IPC_CUSTODIAL_GANG_RAPE",
        "IPC_RAPE_GANG_RAPE",
        "IPC_RAPE_OTHERS",
    ],
    "KIDNAPPING": [
        "IPC_KIDNAPPING_AND_ABDUCTION_OF_WOMEN_AND_GIRLS",
        "IPC_KIDNAPPING_AND_ABDUCTION_OF_OTHERS",
        "IPC_KIDNAPPING_ABDUCTION",
        "IPC_KIDNAPPING_FOR_RANSOM",
    ],
    "ROBBERY_DACOITY": [
        "IPC_ROBBERY",
        "IPC_DACOITY",
        "IPC_OTHER_DACOITY",
        "IPC_DACOITY_WITH_MURDER",
        "IPC_PREPARATION_AND_ASSEMBLY_FOR_DACOITY",
    ],
    "OTHER_VIOLENCE": [
        "IPC_DOWRY_DEATHS",
        "IPC_ARSON",
        "IPC_ATTEMPT_TO_MURDER",
    ],
}

# ============================================================
# 2. CHECK TENSORFLOW
# ============================================================

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        Input,
        Dense,
        Dropout,
        Flatten,
        LSTM,
        GRU,
    )
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError as exc:
    raise SystemExit(
        "\nTensorFlow is not installed in the active virtual environment.\n"
        "Run: python -m pip install tensorflow\n"
        "Then rerun this script.\n"
    ) from exc

tf.keras.utils.set_random_seed(SEED)

# ============================================================
# 3. LOAD DATA AND NORMALIZE IDENTIFIERS
# ============================================================

def load_data(path, name):
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing key columns: {missing}")

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df = df.dropna(subset=KEYS).copy()
    df["YEAR"] = df["YEAR"].astype(int)

    for col in ["STATE", "UNIT_NAME"]:
        df[col] = (
            df[col].astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.upper()
        )

    if df.duplicated(KEYS).any():
        raise ValueError(
            f"{name} has duplicate YEAR/STATE/UNIT_NAME keys "
            "after normalization."
        )

    return df


raw = load_data(RAW_PATH, "Raw integrated dataset")
baseline = load_data(BASELINE_PATH, "Baseline dataset")
enhanced = load_data(ENHANCED_PATH, "Enhanced dataset")

print("Raw dataset:", raw.shape)
print("Baseline dataset:", baseline.shape)
print("Enhanced dataset:", enhanced.shape)

# ============================================================
# 4. CREATE TARGET WITH TRAINING-ONLY CUT POINTS
# ============================================================

burden = raw[KEYS].copy()
burden_components = []

for group_name, columns in BURDEN_GROUPS.items():
    available = [c for c in columns if c in raw.columns]
    if available:
        values = raw[available].apply(pd.to_numeric, errors="coerce")
        burden_components.append(values.sum(axis=1, min_count=1))

if not burden_components:
    raise ValueError(
        "No expected crime source columns were found in the raw dataset."
    )

burden["VIOLENT_CRIME_BURDEN"] = pd.concat(
    burden_components, axis=1
).sum(axis=1, min_count=1)

burden = burden.dropna(subset=["VIOLENT_CRIME_BURDEN"]).copy()

training_burden = burden.loc[
    burden["YEAR"].isin(TRAIN_YEARS), "VIOLENT_CRIME_BURDEN"
]

if training_burden.empty:
    raise ValueError("No burden values exist in the configured training years.")

cut_points = np.unique(
    np.quantile(training_burden, [0.2, 0.4, 0.6, 0.8])
)

if len(cut_points) != 4:
    raise ValueError(
        "Could not derive four distinct target cut points from the training "
        "period. Inspect the burden distribution before proceeding."
    )

burden[TARGET] = (
    np.searchsorted(
        cut_points,
        burden["VIOLENT_CRIME_BURDEN"],
        side="right",
    ) + 1
)

print("\nTraining-only target cut points:", cut_points)
print("Target counts by year:")
print(burden.groupby(["YEAR", TARGET]).size().unstack(fill_value=0))

# ============================================================
# 5. SELECT BASELINE AND ENHANCED FEATURES
# ============================================================

missing_base = [c for c in BASE_FEATURES if c not in baseline.columns]
if missing_base:
    raise ValueError(f"Baseline dataset is missing features: {missing_base}")

enhanced_extra = [
    c for c in enhanced.columns
    if c not in KEYS + [TARGET] + BASE_FEATURES
    and pd.api.types.is_numeric_dtype(enhanced[c])
]

enhanced_features = BASE_FEATURES + enhanced_extra

print("\nEnhanced features:", enhanced_extra)

# ============================================================
# 6. BUILD THREE-YEAR SEQUENCES
# ============================================================

def build_sequences(source_df, feature_columns, dataset_name):
    """
    Each sample predicts target at year t from features at t-3, t-2, t-1.
    Features are matched by normalized STATE and UNIT_NAME, never row order.
    """

    missing = [
        c for c in KEYS + feature_columns if c not in source_df.columns
    ]
    if missing:
        raise ValueError(f"{dataset_name} is missing columns: {missing}")

    source = source_df[KEYS + feature_columns].copy()

    for col in feature_columns:
        source[col] = pd.to_numeric(source[col], errors="coerce")

    lookup = {}
    for row in source.itertuples(index=False, name=None):
        year, state, unit, *values = row
        lookup[(int(year), state, unit)] = np.asarray(
            values, dtype=np.float32
        )

    X_sequences = []
    metadata = []

    eligible_targets = burden[
        burden["YEAR"].isin(
            TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS
        )
    ]

    for row in eligible_targets.itertuples(index=False, name=None):
        year, state, unit, _, target_value = row
        sequence = []

        for historical_year in range(
            int(year) - SEQUENCE_LENGTH, int(year)
        ):
            values = lookup.get((historical_year, state, unit))
            if values is None:
                values = np.full(
                    len(feature_columns), np.nan, dtype=np.float32
                )
            sequence.append(values)

        sequence = np.asarray(sequence, dtype=np.float32)

        # Keep the sample if any historical feature is present.
        if not np.isfinite(sequence).any():
            continue

        X_sequences.append(sequence)
        metadata.append({
            "YEAR": int(year),
            "STATE": state,
            "UNIT_NAME": unit,
            TARGET: int(target_value),
        })

    if not X_sequences:
        raise ValueError(f"{dataset_name}: no usable sequences were created.")

    X = np.stack(X_sequences)
    meta = pd.DataFrame(metadata)

    print(f"\n{dataset_name} sequence data:")
    print("Sequence shape:", X.shape)
    print("Rows by target year:")
    print(meta.groupby("YEAR").size().to_string())

    meta.to_csv(
        OUTPUT_DIR / f"{dataset_name.lower()}_sequence_metadata.csv",
        index=False,
    )

    return X, meta


# ============================================================
# 7. PREPROCESS USING TRAINING DATA ONLY
# ============================================================

def prepare_splits(X, metadata, dataset_name):
    years = metadata["YEAR"].to_numpy()
    y = metadata[TARGET].to_numpy(dtype=np.int32)

    train_mask = np.isin(years, TRAIN_YEARS)
    val_mask = np.isin(years, VALIDATION_YEARS)
    test_mask = np.isin(years, TEST_YEARS)

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(
            f"{dataset_name}: train, validation, or test partition is empty."
        )

    X_train_raw = X[train_mask]
    X_val_raw = X[val_mask]
    X_test_raw = X[test_mask]

    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    print(f"\n{dataset_name} split sizes:")
    print("Train:", len(y_train))
    print("Validation:", len(y_val))
    print("Test:", len(y_test))

    if len(y_test) < 100:
        raise ValueError(
            f"{dataset_name}: only {len(y_test)} test rows remain. "
            "Do not interpret the scores."
        )

    n_timesteps = X.shape[1]
    n_features = X.shape[2]

    # Fit imputation/scaling across training samples and timesteps,
    # preserving the three-timestep sequence for LSTM and GRU.
    X_train_2d = X_train_raw.reshape(-1, n_features)
    X_val_2d = X_val_raw.reshape(-1, n_features)
    X_test_2d = X_test_raw.reshape(-1, n_features)

    # Remove features entirely missing throughout the training period.
    usable_features = np.isfinite(X_train_2d).any(axis=0)
    if not usable_features.any():
        raise ValueError(f"{dataset_name}: all training features are missing.")

    X_train_2d = X_train_2d[:, usable_features]
    X_val_2d = X_val_2d[:, usable_features]
    X_test_2d = X_test_2d[:, usable_features]

    imputer = SimpleImputer(strategy="median")
    X_train_2d = imputer.fit_transform(X_train_2d)
    X_val_2d = imputer.transform(X_val_2d)
    X_test_2d = imputer.transform(X_test_2d)

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_val_2d = scaler.transform(X_val_2d)
    X_test_2d = scaler.transform(X_test_2d)

    n_kept_features = int(usable_features.sum())
    X_train = X_train_2d.reshape(
        len(X_train_raw), n_timesteps, n_kept_features
    )
    X_val = X_val_2d.reshape(
        len(X_val_raw), n_timesteps, n_kept_features
    )
    X_test = X_test_2d.reshape(
        len(X_test_raw), n_timesteps, n_kept_features
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# 8. MODEL BUILDERS
# All models output one five-class prediction per sample.
# ============================================================

def build_ann(input_shape, number_of_classes=5):
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.25),
        Dense(64, activation="relu"),
        Dropout(0.20),
        Dense(number_of_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(input_shape, number_of_classes=5):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False),
        Dropout(0.25),
        Dense(32, activation="relu"),
        Dense(number_of_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gru(input_shape, number_of_classes=5):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=False),
        Dropout(0.25),
        Dense(32, activation="relu"),
        Dense(number_of_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


MODEL_BUILDERS = {
    "ANN": build_ann,
    "LSTM": build_lstm,
    "GRU": build_gru,
}

# ============================================================
# 9. TRAIN AND EVALUATE
# ============================================================

def run_experiment(dataset_name, source_df, feature_columns):
    X_all, metadata = build_sequences(
        source_df, feature_columns, dataset_name
    )

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_splits(
        X_all, metadata, dataset_name
    )

    results = []
    test_metadata = metadata[
        metadata["YEAR"].isin(TEST_YEARS)
    ].reset_index(drop=True)

    if len(test_metadata) != len(y_test):
        raise ValueError("Test metadata and target lengths do not match.")

    for model_name, builder in MODEL_BUILDERS.items():
        print(f"\nTraining {dataset_name} {model_name} ...")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        model = builder(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            number_of_classes=5,
        )

        # Check output shape before training to catch rank errors early.
        output_shape = tuple(model.output_shape)
        if len(output_shape) != 2 or output_shape[-1] != 5:
            raise ValueError(
                f"{model_name} must output (batch, 5), got {output_shape}"
            )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        )

        history = model.fit(
            X_train,
            y_train - 1,
            validation_data=(X_val, y_val - 1),
            epochs=100,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=1,
        )

        val_probabilities = model.predict(X_val, verbose=0)
        val_predictions = np.argmax(val_probabilities, axis=1) + 1

        test_probabilities = model.predict(X_test, verbose=0)
        test_predictions = np.argmax(test_probabilities, axis=1) + 1

        result = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Train_Rows": len(y_train),
            "Validation_Rows": len(y_val),
            "Test_Rows": len(y_test),
            "Best_Epoch": int(np.argmin(history.history["val_loss"]) + 1),
            "Validation_Accuracy": accuracy_score(y_val, val_predictions),
            "Accuracy": accuracy_score(y_test, test_predictions),
            "Precision_Weighted": precision_score(
                y_test, test_predictions,
                average="weighted", zero_division=0
            ),
            "Recall_Weighted": recall_score(
                y_test, test_predictions,
                average="weighted", zero_division=0
            ),
            "F1_Weighted": f1_score(
                y_test, test_predictions,
                average="weighted", zero_division=0
            ),
            "Cohen_Kappa": cohen_kappa_score(y_test, test_predictions),
        }

        try:
            result["ROC_AUC_Weighted_OVR"] = roc_auc_score(
                y_test,
                test_probabilities,
                labels=np.arange(1, 6),
                multi_class="ovr",
                average="weighted",
            )
        except ValueError:
            result["ROC_AUC_Weighted_OVR"] = np.nan

        results.append(result)

        model_dir = OUTPUT_DIR / dataset_name.lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_dir / f"{model_name.lower()}_model.keras")

        predictions_df = test_metadata.copy()
        predictions_df["ACTUAL_TARGET"] = y_test
        predictions_df["PREDICTED_TARGET"] = test_predictions

        for index in range(5):
            predictions_df[f"PROBABILITY_CLASS_{index + 1}"] = (
                test_probabilities[:, index]
            )

        predictions_df.to_csv(
            model_dir / f"{model_name.lower()}_test_predictions.csv",
            index=False,
        )

        pd.DataFrame(
            confusion_matrix(
                y_test, test_predictions, labels=[1, 2, 3, 4, 5]
            ),
            index=[f"Actual_{i}" for i in range(1, 6)],
            columns=[f"Predicted_{i}" for i in range(1, 6)],
        ).to_csv(model_dir / f"{model_name.lower()}_confusion_matrix.csv")

        report = classification_report(
            y_test,
            test_predictions,
            labels=[1, 2, 3, 4, 5],
            zero_division=0,
            output_dict=True,
        )
        pd.DataFrame(report).transpose().to_csv(
            model_dir / f"{model_name.lower()}_classification_report.csv"
        )

        print(
            f"{dataset_name} {model_name}: "
            f"test accuracy={result['Accuracy']:.4f}, "
            f"weighted F1={result['F1_Weighted']:.4f}, "
            f"Kappa={result['Cohen_Kappa']:.4f}"
        )

    return results


# ============================================================
# 10. RUN BOTH EXPERIMENTS
# ============================================================

all_results = []

all_results.extend(
    run_experiment("Baseline", baseline, BASE_FEATURES)
)
all_results.extend(
    run_experiment("Enhanced", enhanced, enhanced_features)
)

results_df = pd.DataFrame(all_results)
results_path = OUTPUT_DIR / "deep_learning_model_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\n========== DEEP-LEARNING COMPARISON ==========")
print(results_df.to_string(index=False))
print("\nResults saved to:", results_path)
print("Models and predictions saved inside:", OUTPUT_DIR)
print("Original source CSV files were not modified.")
