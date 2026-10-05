"""
Enhanced feature engineering for the Crime_Analysis project.

INPUT
-----
data/processed/district_grouped_reduced.csv

OUTPUT
------
data/processed/district_enhanced_features.csv

Exactly five engineered features are created:

1. TOTAL_GROUPED_CRIME
2. ACTIVE_CRIME_GROUPS
3. CRIME_DIVERSITY_INDEX
4. PREV_YEAR_TOTAL_CRIME
5. YOY_TOTAL_CRIME_CHANGE

IMPORTANT
---------
TARGET_VIOLENT_LEVEL is never used to calculate any feature.

However, the existing project target is itself constructed from
current-year violent-crime burden. Therefore current-year crime-derived
features are explicitly assessed for potential target leakage in the
leakage audit.

No model training is performed here.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "district_grouped_reduced.csv"
)

OUTPUT_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "district_enhanced_features.csv"
)

OUTPUT_TABLES_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

FEATURE_SUMMARY_PATH = (
    OUTPUT_TABLES_DIR
    / "enhanced_feature_summary.csv"
)


# =====================================================================
# EXISTING COLUMNS
# =====================================================================

METADATA_COLUMNS = [
    "YEAR",
    "STATE",
    "UNIT_NAME",
]

GROUP_COLUMNS = [
    "GROUP_HOMICIDE",
    "GROUP_SEXUAL_VIOLENCE",
    "GROUP_KIDNAPPING_ABDUCTION",
    "GROUP_ROBBERY_DACOITY",
    "GROUP_PUBLIC_VIOLENCE",
    "GROUP_PROPERTY_CRIME",
]

TARGET_COLUMN = "TARGET_VIOLENT_LEVEL"

NEW_FEATURE_COLUMNS = [
    "TOTAL_GROUPED_CRIME",
    "ACTIVE_CRIME_GROUPS",
    "CRIME_DIVERSITY_INDEX",
    "PREV_YEAR_TOTAL_CRIME",
    "YOY_TOTAL_CRIME_CHANGE",
]

EXPECTED_INPUT_COLUMNS = (
    METADATA_COLUMNS
    + GROUP_COLUMNS
    + [TARGET_COLUMN]
)

EXPECTED_OUTPUT_COLUMNS = (
    METADATA_COLUMNS
    + GROUP_COLUMNS
    + NEW_FEATURE_COLUMNS
    + [TARGET_COLUMN]
)


# =====================================================================
# VALIDATION
# =====================================================================

def validate_input_columns(
    df: pd.DataFrame
) -> None:
    """
    Ensure the input dataset contains exactly the expected
    existing columns.
    """

    missing = [
        column
        for column in EXPECTED_INPUT_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required input columns:\n"
            + "\n".join(missing)
        )


# =====================================================================
# NUMERIC PREPARATION
# =====================================================================

def prepare_group_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert the six grouped crime columns to numeric.

    Missing values remain missing.
    No global fillna(0) is performed.
    """

    result = df.copy()

    for column in GROUP_COLUMNS:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    result["YEAR"] = pd.to_numeric(
        result["YEAR"],
        errors="coerce"
    )

    return result


# =====================================================================
# 1. TOTAL GROUPED CRIME
# =====================================================================

def create_total_grouped_crime(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the total across the six existing grouped crime
    categories.

    If all six source values are missing, the result remains missing.

    If some values are available, the available values are summed.
    """

    result = df.copy()

    result["TOTAL_GROUPED_CRIME"] = (
        result[GROUP_COLUMNS]
        .sum(
            axis=1,
            min_count=1
        )
    )

    return result


# =====================================================================
# 2. ACTIVE CRIME GROUPS
# =====================================================================

def create_active_crime_groups(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Count how many of the six grouped crime categories have a value
    greater than zero.

    For rows where all six grouped values are missing, the result
    remains missing rather than being interpreted as zero active groups.
    """

    result = df.copy()

    all_missing = (
        result[GROUP_COLUMNS]
        .isna()
        .all(axis=1)
    )

    result["ACTIVE_CRIME_GROUPS"] = (
        result[GROUP_COLUMNS]
        .gt(0)
        .sum(axis=1)
        .astype("float64")
    )

    result.loc[
        all_missing,
        "ACTIVE_CRIME_GROUPS"
    ] = np.nan

    return result


# =====================================================================
# 3. CRIME DIVERSITY INDEX
# =====================================================================

def calculate_shannon_entropy(
    row: pd.Series
) -> float:
    """
    Calculate Shannon entropy for one row using the six grouped
    crime categories.

    Formula:

        H = -sum(p_i * ln(p_i))

    where:

        p_i = crime_group_i / total_grouped_crime

    Zero-valued groups contribute zero to the entropy.

    If the total crime is zero, entropy is defined as 0.0 because
    there is no observed crime distribution.

    If all six source values are missing, entropy remains NaN.
    """

    values = row[GROUP_COLUMNS].to_numpy(
        dtype=float
    )

    if np.all(np.isnan(values)):
        return np.nan

    values = np.nan_to_num(
        values,
        nan=0.0
    )

    total = values.sum()

    if total == 0:
        return 0.0

    probabilities = (
        values / total
    )

    positive_probabilities = (
        probabilities[
            probabilities > 0
        ]
    )

    return float(
        -np.sum(
            positive_probabilities
            * np.log(
                positive_probabilities
            )
        )
    )


def create_crime_diversity_index(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create the Shannon entropy-based crime diversity index.
    """

    result = df.copy()

    result["CRIME_DIVERSITY_INDEX"] = (
        result.apply(
            calculate_shannon_entropy,
            axis=1
        )
    )

    return result


# =====================================================================
# 4. PREVIOUS YEAR TOTAL CRIME
# =====================================================================

def create_previous_year_total_crime(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Obtain TOTAL_GROUPED_CRIME from the actual previous calendar year
    for the same STATE and UNIT_NAME.

    Matching key:

        STATE + UNIT_NAME + YEAR

    Previous year:

        YEAR - 1

    No previous row is assumed from dataframe ordering.

    If the actual previous-year record does not exist, the value
    remains missing.
    """

    result = df.copy()

    lookup = (
        result[
            [
                "STATE",
                "UNIT_NAME",
                "YEAR",
                "TOTAL_GROUPED_CRIME",
            ]
        ]
        .copy()
    )

    lookup["YEAR"] = (
        pd.to_numeric(
            lookup["YEAR"],
            errors="coerce"
        )
    )

    lookup = lookup.rename(
        columns={
            "YEAR":
                "PREVIOUS_YEAR",

            "TOTAL_GROUPED_CRIME":
                "PREV_YEAR_TOTAL_CRIME",
        }
    )

    result["PREVIOUS_YEAR"] = (
        result["YEAR"] - 1
    )

    result = result.merge(
        lookup[
            [
                "STATE",
                "UNIT_NAME",
                "PREVIOUS_YEAR",
                "PREV_YEAR_TOTAL_CRIME",
            ]
        ],
        on=[
            "STATE",
            "UNIT_NAME",
            "PREVIOUS_YEAR",
        ],
        how="left",
        sort=False,
        validate="one_to_one",
    )

    result = result.drop(
        columns=["PREVIOUS_YEAR"]
    )

    return result


# =====================================================================
# 5. YEAR-OVER-YEAR CHANGE
# =====================================================================

def create_yoy_total_crime_change(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate:

        current TOTAL_GROUPED_CRIME
        -
        PREV_YEAR_TOTAL_CRIME

    If the previous-year value does not exist, the result remains
    missing.

    No percentage change is calculated.
    """

    result = df.copy()

    result["YOY_TOTAL_CRIME_CHANGE"] = (
        result["TOTAL_GROUPED_CRIME"]
        - result["PREV_YEAR_TOTAL_CRIME"]
    )

    return result


# =====================================================================
# COMPLETE FEATURE ENGINEERING PIPELINE
# =====================================================================

def engineer_enhanced_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create exactly the five requested engineered features.
    """

    validate_input_columns(df)

    result = prepare_group_columns(df)

    # ---------------------------------------------------------------
    # Current-year aggregate features
    # ---------------------------------------------------------------

    result = create_total_grouped_crime(
        result
    )

    result = create_active_crime_groups(
        result
    )

    result = create_crime_diversity_index(
        result
    )

    # ---------------------------------------------------------------
    # Previous-year feature
    # ---------------------------------------------------------------

    result = create_previous_year_total_crime(
        result
    )

    # ---------------------------------------------------------------
    # Year-over-year change
    # ---------------------------------------------------------------

    result = create_yoy_total_crime_change(
        result
    )

    # ---------------------------------------------------------------
    # Final column ordering
    # ---------------------------------------------------------------

    result = result[
        EXPECTED_OUTPUT_COLUMNS
    ]

    return result


# =====================================================================
# FEATURE SUMMARY
# =====================================================================

def create_enhanced_feature_summary() -> pd.DataFrame:
    """
    Create the required feature summary table.
    """

    return pd.DataFrame([
        {
            "Feature Name":
                "TOTAL_GROUPED_CRIME",

            "Purpose":
                "Total crime count across the six existing grouped crime categories.",

            "Calculation/Source":
                "Sum of GROUP_HOMICIDE, GROUP_SEXUAL_VIOLENCE, GROUP_KIDNAPPING_ABDUCTION, GROUP_ROBBERY_DACOITY, GROUP_PUBLIC_VIOLENCE, and GROUP_PROPERTY_CRIME.",

            "Temporal Information Used":
                "Current year only.",

            "Leakage Risk":
                "HIGH — current-year crime values overlap conceptually with the existing target construction."
        },

        {
            "Feature Name":
                "ACTIVE_CRIME_GROUPS",

            "Purpose":
                "Number of the six crime groups with a count greater than zero.",

            "Calculation/Source":
                "Count of grouped crime features whose value is greater than zero.",

            "Temporal Information Used":
                "Current year only.",

            "Leakage Risk":
                "HIGH — derived from current-year crime values used in the target construction."
        },

        {
            "Feature Name":
                "CRIME_DIVERSITY_INDEX",

            "Purpose":
                "Measure how distributed crime is across the six crime groups.",

            "Calculation/Source":
                "Shannon entropy: -sum(p_i * ln(p_i)); zero groups contribute zero; total crime equal to zero gives entropy 0.0; all-source-missing rows remain missing.",

            "Temporal Information Used":
                "Current year only.",

            "Leakage Risk":
                "HIGH — derived from the current-year crime distribution."
        },

        {
            "Feature Name":
                "PREV_YEAR_TOTAL_CRIME",

            "Purpose":
                "Represent the total grouped crime count from the actual previous calendar year for the same state and unit.",

            "Calculation/Source":
                "Exact match on STATE + UNIT_NAME + YEAR, where matched YEAR = current YEAR - 1.",

            "Temporal Information Used":
                "Previous calendar year only.",

            "Leakage Risk":
                "LOW — uses only historical information and never uses future-year information."
        },

        {
            "Feature Name":
                "YOY_TOTAL_CRIME_CHANGE",

            "Purpose":
                "Measure the absolute change in total grouped crime from the previous year to the current year.",

            "Calculation/Source":
                "TOTAL_GROUPED_CRIME - PREV_YEAR_TOTAL_CRIME.",

            "Temporal Information Used":
                "Current year and previous calendar year.",

            "Leakage Risk":
                "HIGH — contains the current-year TOTAL_GROUPED_CRIME component."
        },
    ])


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print("=" * 80)
    print("ENHANCED CRIME FEATURE ENGINEERING")
    print("=" * 80)

    # ---------------------------------------------------------------
    # Load input
    # ---------------------------------------------------------------

    print("\nLoading:")
    print(INPUT_DATA)

    df = pd.read_csv(
        INPUT_DATA
    )

    print(
        f"\nOriginal shape: {df.shape}"
    )

    # ---------------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------------

    validate_input_columns(df)

    original_row_count = len(df)
    original_column_count = len(df.columns)

    original_columns = df.columns.tolist()

    # ---------------------------------------------------------------
    # Create enhanced dataset
    # ---------------------------------------------------------------

    enhanced_df = (
        engineer_enhanced_features(
            df
        )
    )

    # ---------------------------------------------------------------
    # Validate output columns
    # ---------------------------------------------------------------

    new_columns = [
        column
        for column in enhanced_df.columns
        if column not in original_columns
    ]

    print("\nNew columns created:")
    for column in new_columns:
        print(f"- {column}")

    if new_columns != NEW_FEATURE_COLUMNS:
        raise ValueError(
            "The output does not contain exactly the five requested "
            "new features."
        )

    # ---------------------------------------------------------------
    # Row count
    # ---------------------------------------------------------------

    print(
        "\nRow count:"
    )

    print(
        "Original :", original_row_count
    )

    print(
        "Enhanced :", len(enhanced_df)
    )

    if len(enhanced_df) != original_row_count:
        raise ValueError(
            "Row count changed during feature engineering."
        )

    # ---------------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------------

    print(
        "\nMissing values:"
    )

    print(
        enhanced_df
        .isna()
        .sum()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Infinite values
    # ---------------------------------------------------------------

    numeric_columns = (
        enhanced_df
        .select_dtypes(
            include="number"
        )
        .columns
    )

    infinite_counts = np.isinf(
        enhanced_df[
            numeric_columns
        ].to_numpy()
    ).sum()

    print(
        "\nTotal infinite numeric values:",
        int(infinite_counts)
    )

    if infinite_counts != 0:
        raise ValueError(
            "Infinite values detected."
        )

    # ---------------------------------------------------------------
    # Duplicate rows
    # ---------------------------------------------------------------

    duplicate_rows = (
        enhanced_df
        .duplicated()
        .sum()
    )

    print(
        "\nDuplicate rows:",
        int(duplicate_rows)
    )

    # ---------------------------------------------------------------
    # Previous-year matching
    # ---------------------------------------------------------------

    previous_year_available = (
        enhanced_df[
            "PREV_YEAR_TOTAL_CRIME"
        ]
        .notna()
    )

    previous_year_count = (
        int(
            previous_year_available.sum()
        )
    )

    previous_year_missing = (
        int(
            (~previous_year_available).sum()
        )
    )

    print(
        "\nPrevious-year matching:"
    )

    print(
        "Records with previous-year information :",
        previous_year_count
    )

    print(
        "Records without previous-year information:",
        previous_year_missing
    )

    # ---------------------------------------------------------------
    # Target distribution
    # ---------------------------------------------------------------

    print(
        "\nTarget distribution:"
    )

    print(
        enhanced_df[
            TARGET_COLUMN
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Data types
    # ---------------------------------------------------------------

    print(
        "\nData types:"
    )

    print(
        enhanced_df
        .dtypes
        .to_string()
    )

    # ---------------------------------------------------------------
    # Feature summary
    # ---------------------------------------------------------------

    OUTPUT_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_summary = (
        create_enhanced_feature_summary()
    )

    feature_summary.to_csv(
        FEATURE_SUMMARY_PATH,
        index=False
    )

    print(
        "\nFeature summary saved to:"
    )

    print(
        FEATURE_SUMMARY_PATH
    )

    # ---------------------------------------------------------------
    # Save enhanced dataset
    # ---------------------------------------------------------------

    enhanced_df.to_csv(
        OUTPUT_DATA,
        index=False
    )

    print(
        "\nEnhanced dataset saved to:"
    )

    print(
        OUTPUT_DATA
    )

    # ---------------------------------------------------------------
    # Final status
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 80)

    print(
        f"Original shape : "
        f"({original_row_count}, {original_column_count})"
    )

    print(
        f"Enhanced shape : "
        f"{enhanced_df.shape}"
    )

    print(
        "New features   :",
        len(new_columns)
    )

    print(
        "Models trained : NO"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()