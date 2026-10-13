
"""Retired legacy runner. Use tabular_forecast_evaluation.py instead."""
if __name__ == "__main__":
    raise SystemExit(
        "This legacy runner is disabled to protect its saved outputs. Run "
        "ML_Implementation/tabular_forecast_evaluation.py after the v3 input audit."
    )

from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from target_creation import create_target

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, roc_auc_score
)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent

RAW_PATH = PROJECT_DIR / "data" / "processed" / "district_integrated_final.csv"
BASELINE_PATH = PROJECT_DIR / "data" / "processed" / "district_grouped_reduced.csv"
ENHANCED_PATH = PROJECT_DIR / "data" / "processed" / "district_enhanced_features.csv"

OUTPUT_DIR = ML_DIR / "temporal_leakage_free_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TARGET = "TARGET_VIOLENT_LEVEL"

TRAIN_YEARS = list(range(2001, 2011))
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
        "IPC_RIOTS", "IPC_RIOTS_COMMUNAL",
        "IPC_RIOTS_INDUSTRIAL", "IPC_RIOTS_POLITICAL",
        "IPC_RIOTS_CASTE_CONFLICT", "IPC_RIOTS_AGRARIAN",
        "IPC_RIOTS_STUDENTS", "IPC_RIOTS_SECTARIAN",
    ],
    "SEXUAL_VIOLENCE": [
        "IPC_RAPE", "IPC_OTHER_RAPE", "IPC_CUSTODIAL_RAPE",
        "IPC_CUSTODIAL_GANG_RAPE", "IPC_RAPE_GANG_RAPE",
        "IPC_RAPE_OTHERS",
    ],
    "KIDNAPPING": [
        "IPC_KIDNAPPING_AND_ABDUCTION_OF_WOMEN_AND_GIRLS",
        "IPC_KIDNAPPING_AND_ABDUCTION_OF_OTHERS",
        "IPC_KIDNAPPING_ABDUCTION",
        "IPC_KIDNAPPING_FOR_RANSOM",
    ],
    "ROBBERY_DACOITY": [
        "IPC_ROBBERY", "IPC_DACOITY", "IPC_OTHER_DACOITY",
        "IPC_DACOITY_WITH_MURDER",
        "IPC_PREPARATION_AND_ASSEMBLY_FOR_DACOITY",
    ],
    "OTHER_VIOLENCE": [
        "IPC_DOWRY_DEATHS", "IPC_ARSON",
        "IPC_ATTEMPT_TO_MURDER",
    ],
}

# ------------------------------------------------------------
# LOAD AND NORMALIZE KEYS
# ------------------------------------------------------------
def load_data(path, name):
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

    df = pd.read_csv(path)

    for col in KEYS:
        if col not in df.columns:
            raise ValueError(f"{name} is missing key column {col}")

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df = df.dropna(subset=KEYS).copy()
    df["YEAR"] = df["YEAR"].astype(int)

    # Normalize capitalization and whitespace for matching.
    for col in ["STATE", "UNIT_NAME"]:
        df[col] = (
            df[col].astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.upper()
        )

    if df.duplicated(KEYS).any():
        raise ValueError(
            f"{name} has duplicate keys after normalization. "
            "Inspect duplicate YEAR/STATE/UNIT_NAME rows."
        )

    return df


raw = load_data(RAW_PATH, "Raw dataset")
baseline = load_data(BASELINE_PATH, "Baseline dataset")
enhanced = load_data(ENHANCED_PATH, "Enhanced dataset")

print("Raw:", raw.shape)
print("Baseline:", baseline.shape)
print("Enhanced:", enhanced.shape)

# ------------------------------------------------------------
# CANONICAL TARGET WITH FROZEN TRAINING-ONLY CUT POINTS
# ------------------------------------------------------------
threshold_table = pd.read_csv(PROJECT_DIR / "outputs/tables/target_thresholds.csv")
threshold_table = threshold_table.sort_values("boundary_index")
cut_points = threshold_table["upper_cut_point"].to_numpy(dtype=float)
if (threshold_table["fit_years"].astype(str).unique().tolist() != ["2001-2010"]
        or not np.array_equal(cut_points, np.array([107., 237., 362., 563.]))):
    raise ValueError("The frozen canonical target thresholds are missing or unexpected.")

canonical_raw, _ = create_target(raw, thresholds=cut_points)
burden = canonical_raw[KEYS + ["VIOLENT_CRIME_BURDEN", TARGET]].dropna(
    subset=["VIOLENT_CRIME_BURDEN", TARGET]
).copy()
burden[TARGET] = burden[TARGET].astype(int)

print("\nTraining-only target cut points:", cut_points)
print("Target counts by year:")
print(burden.groupby(["YEAR", TARGET]).size().unstack(fill_value=0))

# ------------------------------------------------------------
# CREATE LAGGED FEATURES BY YEAR-1 + NORMALIZED KEYS
# ------------------------------------------------------------
def make_lagged_dataset(source_df, feature_columns, name):
    missing = [
        c for c in KEYS + feature_columns
        if c not in source_df.columns
    ]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

    source = source_df[KEYS + feature_columns].copy()

    for col in feature_columns:
        source[col] = pd.to_numeric(source[col], errors="coerce")

    # Shift previous-year records forward by one year.
    lagged = source.copy()
    lagged["YEAR"] = lagged["YEAR"] + 1
    lagged = lagged.rename(
        columns={c: f"LAG1_{c}" for c in feature_columns}
    )

    result = source_df[KEYS].merge(
        lagged,
        on=KEYS,
        how="left",
        validate="one_to_one"
    )

    result = result.merge(
        burden[KEYS + [TARGET]],
        on=KEYS,
        how="inner",
        validate="one_to_one"
    )

    lag_features = [f"LAG1_{c}" for c in feature_columns]

    result = result[
        result["YEAR"].isin(
            TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS
        )
    ].copy()

    # Keep rows with at least one available lagged feature.
    result = result.dropna(subset=lag_features, how="all").copy()
    result[TARGET] = result[TARGET].astype(int)

    print(f"\n{name} rows by year:")
    print(result.groupby("YEAR").size().to_string())

    result.to_csv(
        OUTPUT_DIR / f"{name.lower()}_lagged_data.csv",
        index=False
    )

    return result, lag_features


baseline_data, baseline_features = make_lagged_dataset(
    baseline, BASE_FEATURES, "Baseline"
)

enhanced_extra = [
    c for c in enhanced.columns
    if c not in KEYS + [TARGET] + BASE_FEATURES
    and pd.api.types.is_numeric_dtype(enhanced[c])
]

enhanced_data, enhanced_features = make_lagged_dataset(
    enhanced, BASE_FEATURES + enhanced_extra, "Enhanced"
)

# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------
MODELS = {
    "Gaussian Naive Bayes": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        GaussianNB()
    ),
    "Extra Trees": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=300, random_state=42,
            class_weight="balanced", n_jobs=-1
        )
    ),
    "Random Forest": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300, random_state=42,
            class_weight="balanced", n_jobs=-1
        )
    ),
    "MLP": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            random_state=42
        )
    ),
}

# ------------------------------------------------------------
# TRAIN / VALIDATE / TEST
# ------------------------------------------------------------
def run_experiment(data, features, name):
    train = data[data["YEAR"].isin(TRAIN_YEARS)].copy()
    validation = data[data["YEAR"].isin(VALIDATION_YEARS)].copy()
    test = data[data["YEAR"].isin(TEST_YEARS)].copy()

    print(f"\n{name} split rows:")
    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))

    if train.empty or validation.empty or test.empty:
        raise ValueError(f"{name}: a data split is empty.")

    if len(test) < 100:
        raise ValueError(
            f"{name}: only {len(test)} test rows remain. "
            "Do not interpret model scores."
        )

    X_train, y_train = train[features], train[TARGET]
    X_val, y_val = validation[features], validation[TARGET]
    X_test, y_test = test[features], test[TARGET]

    results = []

    for model_name, factory in MODELS.items():
        print(f"Training {name}: {model_name}")
        model = factory()
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        row = {
            "Dataset": name,
            "Model": model_name,
            "Train_Rows": len(train),
            "Validation_Rows": len(validation),
            "Test_Rows": len(test),
            "Validation_Accuracy": accuracy_score(y_val, val_pred),
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision_Weighted": precision_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "Recall_Weighted": recall_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "F1_Weighted": f1_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "Cohen_Kappa": cohen_kappa_score(y_test, predictions),
        }

        try:
            row["ROC_AUC_Weighted_OVR"] = roc_auc_score(
                y_test,
                probabilities,
                labels=model.classes_,
                multi_class="ovr",
                average="weighted"
            )
        except ValueError:
            row["ROC_AUC_Weighted_OVR"] = np.nan

        results.append(row)
        print(
            f"Test accuracy={row['Accuracy']:.4f}, "
            f"weighted F1={row['F1_Weighted']:.4f}"
        )

    return results


all_results = []
all_results.extend(
    run_experiment(baseline_data, baseline_features, "Baseline")
)
all_results.extend(
    run_experiment(enhanced_data, enhanced_features, "Enhanced")
)

results_df = pd.DataFrame(all_results)
output_file = OUTPUT_DIR / "temporal_model_comparison.csv"
results_df.to_csv(output_file, index=False)

print("\n========== FINAL RESULTS ==========")
print(results_df.to_string(index=False))
print("\nSaved to:", output_file)
print("Original source CSV files were not modified.")
