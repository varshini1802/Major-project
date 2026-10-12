
from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
SEQUENCE_LENGTH = 3

random.seed(SEED)
np.random.seed(SEED)

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent

RAW_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "district_integrated_final.csv"
)
BASELINE_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "district_grouped_reduced.csv"
)
ENHANCED_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "district_enhanced_features.csv"
)

OUTPUT_DIR = ML_DIR / "combined_temporal_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TARGET = "TARGET_VIOLENT_LEVEL"

# Identical year splits for all seven models.
TRAIN_YEARS = list(range(2004, 2011))
VALIDATION_YEARS = [2011, 2012]
TEST_YEARS = [2013, 2014]

CLASSES = [1, 2, 3, 4, 5]

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
# 2. LOAD AND NORMALIZE DATA
# ============================================================

def load_data(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} was not found:\n{path}"
        )

    df = pd.read_csv(path)

    missing = [c for c in KEYS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing key columns: {missing}"
        )

    df["YEAR"] = pd.to_numeric(
        df["YEAR"], errors="coerce"
    )

    df = df.dropna(subset=KEYS).copy()
    df["YEAR"] = df["YEAR"].astype(int)

    for col in ["STATE", "UNIT_NAME"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.upper()
        )

    if df.duplicated(KEYS).any():
        raise ValueError(
            f"{name} contains duplicate normalized keys."
        )

    return df


raw = load_data(RAW_PATH, "Raw integrated dataset")
baseline = load_data(BASELINE_PATH, "Baseline dataset")
enhanced = load_data(ENHANCED_PATH, "Enhanced dataset")

print("Raw dataset:", raw.shape)
print("Baseline dataset:", baseline.shape)
print("Enhanced dataset:", enhanced.shape)

missing_base = [
    c for c in BASE_FEATURES if c not in baseline.columns
]
if missing_base:
    raise ValueError(
        f"Missing baseline features: {missing_base}"
    )

missing_enhanced_base = [
    c for c in BASE_FEATURES if c not in enhanced.columns
]
if missing_enhanced_base:
    raise ValueError(
        "Enhanced dataset is missing baseline features: "
        f"{missing_enhanced_base}"
    )

# ============================================================
# 3. CREATE ONE SHARED TARGET
# ============================================================

burden = raw[KEYS].copy()
components = []

for group_name, columns in BURDEN_GROUPS.items():
    available = [c for c in columns if c in raw.columns]

    if available:
        values = raw[available].apply(
            pd.to_numeric, errors="coerce"
        )
        components.append(
            values.sum(axis=1, min_count=1)
        )

if not components:
    raise ValueError(
        "None of the configured crime-burden columns "
        "were found in the raw dataset."
    )

burden["VIOLENT_CRIME_BURDEN"] = pd.concat(
    components, axis=1
).sum(axis=1, min_count=1)

burden = burden.dropna(
    subset=["VIOLENT_CRIME_BURDEN"]
).copy()

train_burden = burden.loc[
    burden["YEAR"].isin(TRAIN_YEARS),
    "VIOLENT_CRIME_BURDEN",
]

if train_burden.empty:
    raise ValueError(
        "No target-burden values found in training years."
    )

# Thresholds are estimated using training years only.
cut_points = np.unique(
    np.quantile(
        train_burden,
        [0.2, 0.4, 0.6, 0.8],
    )
)

if len(cut_points) != 4:
    raise ValueError(
        "Four distinct target thresholds could not be "
        "created. Inspect the training burden distribution."
    )

burden[TARGET] = (
    np.searchsorted(
        cut_points,
        burden["VIOLENT_CRIME_BURDEN"],
        side="right",
    ) + 1
).astype(int)

print("\nShared training-only target cut points:")
print(cut_points)

print("\nTarget class counts by year:")
print(
    burden.groupby(["YEAR", TARGET])
    .size()
    .unstack(fill_value=0)
)

# ============================================================
# 4. CHOOSE ENHANCED FEATURES
# ============================================================

enhanced_extra = [
    c for c in enhanced.columns
    if c not in KEYS + [TARGET] + BASE_FEATURES
    and pd.api.types.is_numeric_dtype(enhanced[c])
]

enhanced_features = BASE_FEATURES + enhanced_extra

print("\nBaseline features:", BASE_FEATURES)
print("Additional enhanced features:", enhanced_extra)

# ============================================================
# 5. BUILD THREE-YEAR SEQUENCES
# ============================================================

def make_lookup(df, feature_columns):
    source = df[KEYS + feature_columns].copy()

    for col in feature_columns:
        source[col] = pd.to_numeric(
            source[col], errors="coerce"
        )

    lookup = {}

    for row in source.itertuples(index=False, name=None):
        year, state, unit, *values = row

        lookup[(int(year), state, unit)] = np.asarray(
            values, dtype=np.float32
        )

    return lookup


baseline_lookup = make_lookup(
    baseline, BASE_FEATURES
)

enhanced_lookup = make_lookup(
    enhanced, enhanced_features
)

eligible_targets = burden[
    burden["YEAR"].isin(
        TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS
    )
].copy()

X_baseline_list = []
X_enhanced_list = []
metadata = []

for row in eligible_targets.itertuples(
    index=False, name=None
):
    year, state, unit, _, target_value = row

    base_sequence = []
    enhanced_sequence = []

    for historical_year in range(
        int(year) - SEQUENCE_LENGTH, int(year)
    ):
        key = (historical_year, state, unit)

        base_values = baseline_lookup.get(key)
        enhanced_values = enhanced_lookup.get(key)

        if base_values is None:
            base_values = np.full(
                len(BASE_FEATURES), np.nan,
                dtype=np.float32,
            )

        if enhanced_values is None:
            enhanced_values = np.full(
                len(enhanced_features), np.nan,
                dtype=np.float32,
            )

        base_sequence.append(base_values)
        enhanced_sequence.append(enhanced_values)

    base_sequence = np.asarray(
        base_sequence, dtype=np.float32
    )
    enhanced_sequence = np.asarray(
        enhanced_sequence, dtype=np.float32
    )

    # Use exactly the same target observations for both
    # feature sets. Both must contain historical information.
    if not np.isfinite(base_sequence).any():
        continue

    if not np.isfinite(enhanced_sequence).any():
        continue

    X_baseline_list.append(base_sequence)
    X_enhanced_list.append(enhanced_sequence)

    metadata.append({
        "YEAR": int(year),
        "STATE": state,
        "UNIT_NAME": unit,
        TARGET: int(target_value),
    })

if not metadata:
    raise ValueError(
        "No common baseline/enhanced temporal samples exist."
    )

X_baseline = np.stack(X_baseline_list)
X_enhanced = np.stack(X_enhanced_list)
metadata = pd.DataFrame(metadata)

# Stable ordering makes the row alignment explicit.
order = np.lexsort((
    metadata["UNIT_NAME"].to_numpy(),
    metadata["STATE"].to_numpy(),
    metadata["YEAR"].to_numpy(),
))

metadata = metadata.iloc[order].reset_index(drop=True)
X_baseline = X_baseline[order]
X_enhanced = X_enhanced[order]

print("\nCommon sample count:", len(metadata))
print("Baseline sequence shape:", X_baseline.shape)
print("Enhanced sequence shape:", X_enhanced.shape)

print("\nCommon samples by target year:")
print(metadata.groupby("YEAR").size().to_string())

metadata.to_csv(
    OUTPUT_DIR / "common_sequence_metadata.csv",
    index=False,
)

# ============================================================
# 6. COMMON TEMPORAL SPLITS
# ============================================================

years = metadata["YEAR"].to_numpy()
y = metadata[TARGET].to_numpy(dtype=int)

train_mask = np.isin(years, TRAIN_YEARS)
val_mask = np.isin(years, VALIDATION_YEARS)
test_mask = np.isin(years, TEST_YEARS)

if not train_mask.any() or not val_mask.any() or not test_mask.any():
    raise ValueError(
        "At least one train/validation/test split is empty."
    )

y_train = y[train_mask]
y_val = y[val_mask]
y_test = y[test_mask]

print("\nCOMMON SPLIT SIZES")
print("Training:", len(y_train))
print("Validation:", len(y_val))
print("Test:", len(y_test))

if len(y_test) < 100:
    raise ValueError(
        "The common test set is unexpectedly small."
    )

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nTest class distribution:")
print(pd.Series(y_test).value_counts().sort_index())

# ============================================================
# 7. PREPROCESSING FIT ON TRAINING DATA ONLY
# ============================================================

def prepare_features(X, name):
    n_samples, n_steps, n_features = X.shape

    X_train_raw = X[train_mask]
    X_val_raw = X[val_mask]
    X_test_raw = X[test_mask]

    # Flatten sample and timestep dimensions only for fitting
    # the feature-wise imputer/scaler.
    train_2d = X_train_raw.reshape(-1, n_features)
    val_2d = X_val_raw.reshape(-1, n_features)
    test_2d = X_test_raw.reshape(-1, n_features)

    usable = np.isfinite(train_2d).any(axis=0)

    if not usable.any():
        raise ValueError(
            f"{name}: every feature is missing in training."
        )

    train_2d = train_2d[:, usable]
    val_2d = val_2d[:, usable]
    test_2d = test_2d[:, usable]

    imputer = SimpleImputer(strategy="median")

    train_2d = imputer.fit_transform(train_2d)
    val_2d = imputer.transform(val_2d)
    test_2d = imputer.transform(test_2d)

    scaler = StandardScaler()

    train_2d = scaler.fit_transform(train_2d)
    val_2d = scaler.transform(val_2d)
    test_2d = scaler.transform(test_2d)

    kept_features = int(usable.sum())

    X_train_seq = train_2d.reshape(
        len(y_train), n_steps, kept_features
    )
    X_val_seq = val_2d.reshape(
        len(y_val), n_steps, kept_features
    )
    X_test_seq = test_2d.reshape(
        len(y_test), n_steps, kept_features
    )

    # Traditional ML receives the same three-year information,
    # represented as a flattened feature vector.
    X_train_flat = X_train_seq.reshape(len(y_train), -1)
    X_val_flat = X_val_seq.reshape(len(y_val), -1)
    X_test_flat = X_test_seq.reshape(len(y_test), -1)

    print(
        f"\n{name}: retained {kept_features} features "
        f"across {n_steps} timesteps."
    )

    return (
        X_train_seq, X_val_seq, X_test_seq,
        X_train_flat, X_val_flat, X_test_flat,
    )


(
    B_train_seq, B_val_seq, B_test_seq,
    B_train_flat, B_val_flat, B_test_flat,
) = prepare_features(X_baseline, "Baseline")

(
    E_train_seq, E_val_seq, E_test_seq,
    E_train_flat, E_val_flat, E_test_flat,
) = prepare_features(X_enhanced, "Enhanced")

# ============================================================
# 8. METRIC CALCULATION AND SAVING
# ============================================================

all_results = []


def evaluate_model(
    dataset_name,
    model_name,
    y_true,
    predictions,
    probabilities,
    train_rows,
    validation_rows,
    model_directory,
):
    row = {
        "Dataset": dataset_name,
        "Model": model_name,
        "Train_Rows": train_rows,
        "Validation_Rows": validation_rows,
        "Test_Rows": len(y_true),
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision_Weighted": precision_score(
            y_true, predictions,
            average="weighted", zero_division=0,
        ),
        "Recall_Weighted": recall_score(
            y_true, predictions,
            average="weighted", zero_division=0,
        ),
        "F1_Weighted": f1_score(
            y_true, predictions,
            average="weighted", zero_division=0,
        ),
        "Cohen_Kappa": cohen_kappa_score(
            y_true, predictions
        ),
    }

    try:
        row["ROC_AUC_Weighted_OVR"] = roc_auc_score(
            y_true,
            probabilities,
            labels=CLASSES,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        row["ROC_AUC_Weighted_OVR"] = np.nan

    all_results.append(row)

    pd.DataFrame(
        confusion_matrix(
            y_true, predictions, labels=CLASSES
        ),
        index=[f"Actual_{c}" for c in CLASSES],
        columns=[f"Predicted_{c}" for c in CLASSES],
    ).to_csv(
        model_directory / "confusion_matrix.csv"
    )

    report = classification_report(
        y_true,
        predictions,
        labels=CLASSES,
        zero_division=0,
        output_dict=True,
    )

    pd.DataFrame(report).transpose().to_csv(
        model_directory / "classification_report.csv"
    )

    print(
        f"{dataset_name} {model_name}: "
        f"Accuracy={row['Accuracy']:.4f}, "
        f"Weighted F1={row['F1_Weighted']:.4f}, "
        f"Kappa={row['Cohen_Kappa']:.4f}, "
        f"ROC-AUC={row['ROC_AUC_Weighted_OVR']:.4f}"
    )

    return row


def save_predictions(
    directory, predictions, probabilities
):
    test_meta = metadata.loc[test_mask].reset_index(drop=True)

    output = test_meta.copy()
    output["ACTUAL_TARGET"] = y_test
    output["PREDICTED_TARGET"] = predictions

    for index in range(len(CLASSES)):
        output[f"PROBABILITY_CLASS_{index + 1}"] = (
            probabilities[:, index]
        )

    output.to_csv(
        directory / "test_predictions.csv",
        index=False,
    )


# ============================================================
# 9. TRADITIONAL ML MODELS
# ============================================================

try:
    from sklearn.base import clone
except ImportError:
    raise SystemExit("scikit-learn is required.")


TRADITIONAL_MODELS = {
    "GaussianNB": GaussianNB(),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
    ),

    "MLP": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        early_stopping=True,
        random_state=SEED,
    ),
}


def run_traditional(
    dataset_name,
    X_train,
    X_val,
    X_test,
):
    dataset_directory = OUTPUT_DIR / dataset_name.lower()
    dataset_directory.mkdir(parents=True, exist_ok=True)

    for model_name, estimator in TRADITIONAL_MODELS.items():
        print(f"\nTraining {dataset_name} {model_name} ...")

        steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]

        if model_name in ["GaussianNB", "MLP"]:
            steps.append(("scaler", StandardScaler()))

        steps.append(("model", clone(estimator)))

        pipeline = Pipeline(steps)
        pipeline.fit(X_train, y_train)

        val_pred = pipeline.predict(X_val)
        test_pred = pipeline.predict(X_test)
        test_prob = pipeline.predict_proba(X_test)

        print(
            "Validation accuracy:",
            f"{accuracy_score(y_val, val_pred):.4f}",
        )

        model_directory = dataset_directory / model_name.lower()
        model_directory.mkdir(parents=True, exist_ok=True)

        evaluate_model(
            dataset_name,
            model_name,
            y_test,
            test_pred,
            test_prob,
            len(y_train),
            len(y_val),
            model_directory,
        )

        save_predictions(
            model_directory, test_pred, test_prob
        )


run_traditional(
    "Baseline",
    B_train_flat, B_val_flat, B_test_flat,
)

run_traditional(
    "Enhanced",
    E_train_flat, E_val_flat, E_test_flat,
)


# ============================================================
# 9B. BASE-PAPER MODELS: RANDOM TREE + META-CLASSIFIER
# Add after the two run_traditional() calls
# ============================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def make_paper_pipeline(estimator, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", estimator))
    return Pipeline(steps)


def run_paper_models(dataset_name, X_train, X_val, X_test):
    """
    Random Tree is implemented as a randomized decision tree.
    The meta-classifier learns from validation-period
    probabilities produced by base models trained on training data.
    The test period is reserved for final evaluation.
    """
    dataset_directory = OUTPUT_DIR / dataset_name.lower()
    dataset_directory.mkdir(parents=True, exist_ok=True)

    paper_models = {
        "Paper_NaiveBayes": make_paper_pipeline(
            GaussianNB(), scale=True
        ),
        # Adaptation: randomized decision tree, not Random Forest.
        "Paper_RandomTree": make_paper_pipeline(
            DecisionTreeClassifier(
                splitter="random",
                max_features="sqrt",
                class_weight="balanced",
                random_state=SEED,
            )
        ),
    }

    fitted_base_models = {}

    for model_name, model in paper_models.items():
        print(f"\nTraining {dataset_name} {model_name} ...")
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)

        model_directory = dataset_directory / model_name.lower()
        model_directory.mkdir(parents=True, exist_ok=True)

        evaluate_model(
            dataset_name, model_name, y_test, test_pred,
            test_prob, len(y_train), len(y_val), model_directory
        )
        save_predictions(model_directory, test_pred, test_prob)

        fitted_base_models[model_name] = model

    # Meta-classifier base learners.
    # Base learners are fitted on TRAINING data only.
    meta_base_models = {
        "NB": make_paper_pipeline(GaussianNB(), scale=True),
        "RandomTree": make_paper_pipeline(
            DecisionTreeClassifier(
                splitter="random",
                max_features="sqrt",
                class_weight="balanced",
                random_state=SEED,
            )
        ),
        "ExtraTrees": make_paper_pipeline(
            ExtraTreesClassifier(
                n_estimators=150,
                random_state=SEED,
                class_weight="balanced",
                n_jobs=-1,
            )
        ),
    }

    for model in meta_base_models.values():
        model.fit(X_train, y_train)

    # Validation probabilities become the meta-classifier inputs.
    # This keeps the test set out of meta-classifier training.
    val_meta_features = np.hstack([
        model.predict_proba(X_val)
        for model in meta_base_models.values()
    ])

    test_meta_features = np.hstack([
        model.predict_proba(X_test)
        for model in meta_base_models.values()
    ])

    meta_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )

    print(f"\nTraining {dataset_name} Paper_MetaClassifier ...")
    meta_model.fit(val_meta_features, y_val)

    meta_pred = meta_model.predict(test_meta_features)
    meta_prob_raw = meta_model.predict_proba(test_meta_features)

    # Align probability columns to class labels [1, 2, 3, 4, 5].
    meta_prob = np.zeros((len(X_test), len(CLASSES)))
    for j, cls in enumerate(meta_model.classes_):
        meta_prob[:, int(cls) - 1] = meta_prob_raw[:, j]

    model_name = "Paper_MetaClassifier"
    model_directory = dataset_directory / model_name.lower()
    model_directory.mkdir(parents=True, exist_ok=True)

    evaluate_model(
        dataset_name, model_name, y_test, meta_pred,
        meta_prob, len(y_train), len(y_val), model_directory
    )
    save_predictions(model_directory, meta_pred, meta_prob)


run_paper_models(
    "Baseline",
    B_train_flat, B_val_flat, B_test_flat,
)

run_paper_models(
    "Enhanced",
    E_train_flat, E_val_flat, E_test_flat,
)



# ============================================================
# 10. DEEP LEARNING MODELS
# ============================================================

try:
    import tensorflow as tf

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        Input,
        Flatten,
        Dense,
        Dropout,
        LSTM,
        GRU,
    )
    from tensorflow.keras.callbacks import EarlyStopping

except ImportError as exc:
    raise SystemExit(
        "\nTensorFlow is required for the deep-learning models.\n"
        "Install it in your active environment using:\n"
        "python -m pip install tensorflow\n"
    ) from exc


tf.keras.utils.set_random_seed(SEED)


def build_ann(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.25),
        Dense(64, activation="relu"),
        Dropout(0.20),
        Dense(5, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False),
        Dropout(0.25),
        Dense(32, activation="relu"),
        Dense(5, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gru(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=False),
        Dropout(0.25),
        Dense(32, activation="relu"),
        Dense(5, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


DL_MODELS = {
    "ANN": build_ann,
    "LSTM": build_lstm,
    "GRU": build_gru,
}


def run_deep_learning(
    dataset_name,
    X_train,
    X_val,
    X_test,
):
    dataset_directory = OUTPUT_DIR / dataset_name.lower()
    dataset_directory.mkdir(parents=True, exist_ok=True)

    for model_name, builder in DL_MODELS.items():
        print(f"\nTraining {dataset_name} {model_name} ...")

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        model = builder(
            (X_train.shape[1], X_train.shape[2])
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

        val_prob = model.predict(X_val, verbose=0)
        val_pred = np.argmax(val_prob, axis=1) + 1

        test_prob = model.predict(X_test, verbose=0)
        test_pred = np.argmax(test_prob, axis=1) + 1

        best_epoch = int(
            np.argmin(history.history["val_loss"]) + 1
        )

        print(
            "Validation accuracy:",
            f"{accuracy_score(y_val, val_pred):.4f}",
            "| Best epoch:",
            best_epoch,
        )

        model_directory = dataset_directory / model_name.lower()
        model_directory.mkdir(parents=True, exist_ok=True)

        model.save(
            model_directory / "model.keras"
        )

        row = evaluate_model(
            dataset_name,
            model_name,
            y_test,
            test_pred,
            test_prob,
            len(y_train),
            len(y_val),
            model_directory,
        )

        row["Best_Epoch"] = best_epoch
        row["Validation_Accuracy"] = accuracy_score(
            y_val, val_pred
        )

        save_predictions(
            model_directory, test_pred, test_prob
        )


run_deep_learning(
    "Baseline",
    B_train_seq, B_val_seq, B_test_seq,
)

run_deep_learning(
    "Enhanced",
    E_train_seq, E_val_seq, E_test_seq,
)

# ============================================================
# 11. FINAL COMPARISON
# ============================================================

results = pd.DataFrame(all_results)

# Add missing deep-learning-only columns for traditional models.
if "Best_Epoch" not in results.columns:
    results["Best_Epoch"] = np.nan

if "Validation_Accuracy" not in results.columns:
    results["Validation_Accuracy"] = np.nan

results = results.sort_values(
    ["F1_Weighted", "Accuracy"],
    ascending=False,
).reset_index(drop=True)

comparison_path = (
    OUTPUT_DIR / "combined_model_comparison.csv"
)

results.to_csv(comparison_path, index=False)

print("\n" + "=" * 72)
print("COMBINED TRADITIONAL ML + DEEP LEARNING RESULTS")
print("=" * 72)

print(
    results[
        [
            "Dataset",
            "Model",
            "Train_Rows",
            "Validation_Rows",
            "Test_Rows",
            "Accuracy",
            "F1_Weighted",
            "Cohen_Kappa",
            "ROC_AUC_Weighted_OVR",
        ]
    ].to_string(index=False)
)

best = results.iloc[0]

print("\nTop result by weighted F1, then accuracy:")
print("Dataset:", best["Dataset"])
print("Model:", best["Model"])
print(f"Accuracy: {best['Accuracy']:.4f}")
print(f"Weighted F1: {best['F1_Weighted']:.4f}")
print(f"Cohen's Kappa: {best['Cohen_Kappa']:.4f}")
print(f"ROC-AUC: {best['ROC_AUC_Weighted_OVR']:.4f}")

print("\nComparison saved to:")
print(comparison_path)

print("\nAll outputs saved inside:")
print(OUTPUT_DIR)

print("\nOriginal source CSV files were not modified.")




import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Results folder
OUTPUT_DIR = Path(__file__).resolve().parent / "combined_temporal_results"

# Load latest model comparison
results = pd.read_csv(
    OUTPUT_DIR / "combined_model_comparison.csv"
)

# Models to compare
models = ["RandomForest", "ExtraTrees", "MLP", "LSTM", "GRU"]
accuracy_data = {}

for model in models:
    accuracy_data[model] = {}
    for dataset in ["Baseline", "Enhanced"]:
        row = results[
            (results["Model"] == model) &
            (results["Dataset"] == dataset)
        ]

        accuracy_data[model][dataset] = (
            row["Accuracy"].iloc[0] * 100
            if not row.empty else 0
        )

plot_data = pd.DataFrame(accuracy_data).T

ax = plot_data.plot(
    kind="bar",
    figsize=(8, 4.5),
    width=0.75
)

ax.set_title("Baseline vs Enhanced Model Accuracy")
ax.set_xlabel("Model")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.legend(title="Feature Set")
plt.xticks(rotation=0)
plt.tight_layout()

graph_path = OUTPUT_DIR / "fig1_baseline_vs_enhanced_accuracy.png"
plt.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.show()

print("Figure saved to:", graph_path)