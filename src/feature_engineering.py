"""
Feature engineering for the base-paper crime classification pipeline.

This module creates meaningful crime-group features from the existing
NCRB integrated dataset.

IMPORTANT
---------
This module does NOT create the final ML target.

The base paper describes a five-class violent-crime classification
problem, but the exact mathematical rule used to convert the violent
crime attributes into the five class labels is not sufficiently
specified in the paper.

Therefore:

    Original crime columns
            ↓
    Verified crime groups
            ↓
    Engineered predictor features
            ↓
    Target definition handled separately

No target column is created here.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_TABLES_DIR = (
    PROJECT_ROOT / "outputs" / "tables"
)


# =====================================================================
# BASE-PAPER VIOLENT CRIME FEATURES
# =====================================================================

BASE_PAPER_FEATURE_GROUPS = {

    "CHNOT_AMOUNTING_TO_MURDER": [
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

    "RAPE": [
        "IPC_RAPE",
        "IPC_OTHER_RAPE",
        "IPC_CUSTODIAL_RAPE",
        "IPC_CUSTODIAL_GANG_RAPE",
        "IPC_RAPE_GANG_RAPE",
        "IPC_RAPE_OTHERS",
    ],

    "MURDER": [
        "IPC_MURDER",
    ],

    "PREPARATION_AND_ASSEMBLY_FOR_DACOITY": [
        "IPC_PREPARATION_AND_ASSEMBLY_FOR_DACOITY",
    ],

    "DOWRY_DEATH": [
        "IPC_DOWRY_DEATHS",
    ],

    "ROBBERY": [
        "IPC_ROBBERY",
    ],

    "DACOITY": [
        "IPC_DACOITY",
        "IPC_OTHER_DACOITY",
        "IPC_DACOITY_WITH_MURDER",
    ],

    "KIDNAPPING_ABDUCTION": [
        "IPC_KIDNAPPING_AND_ABDUCTION_OF_WOMEN_AND_GIRLS",
        "IPC_KIDNAPPING_AND_ABDUCTION_OF_OTHERS",
        "IPC_KIDNAPPING_ABDUCTION",
        "IPC_KIDNAPPING_FOR_RANSOM",
    ],

    "ARSON": [
        "IPC_ARSON",
    ],

    "ATTEMPT_TO_MURDER": [
        "IPC_ATTEMPT_TO_MURDER",
    ],
}


# =====================================================================
# ADDITIONAL CRIME GROUPS
# =====================================================================

ADDITIONAL_CRIME_GROUPS = {

    "THEFT_BURGLARY_TRESPASS": [
        "IPC_THEFT",
        "IPC_OTHER_THEFT",
        "IPC_BURGLARY",
        "IPC_CRIMINAL_TRESPASS_BURGLARY",
    ],

    "CHEATING_FRAUD": [
        "IPC_CHEATING",
        "IPC_FORGERY",
        "IPC_COUNTERFEITING",
    ],

    "EXTORTION": [
        "IPC_EXTORTION",
    ],

    "CRIMINAL_BREACH_OF_TRUST": [
        "IPC_CRIMINAL_BREACH_OF_TRUST",
    ],

    "HURT_ASSAULT": [
        "IPC_HURT_GREVIOUS_HURT",
        "IPC_GRIEVOUS_HURT",
    ],

    "HUMAN_TRAFFICKING": [
        "IPC_HUMANTRAFFICKING",
    ],

    "CRIMES_AGAINST_STATE": [
        "IPC_OFFENCES_AGAINST_STATE",
        "IPC_SEDITION",
    ],

    "RASH_DRIVING": [
        "IPC_RASH_DRIVING",
    ],

    "UNNATURAL_OFFENCE": [
        "IPC_UNNATURAL_OFFENCE",
    ],
}


# =====================================================================
# BASIC COLUMN VALIDATION
# =====================================================================

def get_existing_columns(
    df: pd.DataFrame,
    columns: list[str]
) -> list[str]:
    """
    Return only columns that actually exist in the dataset.
    """

    existing = set(
        df.columns
    )

    return [
        column
        for column in columns
        if column in existing
    ]


# =====================================================================
# GROUP VALIDATION
# =====================================================================

def validate_feature_groups(
    df: pd.DataFrame,
    feature_groups: dict[str, list[str]]
) -> pd.DataFrame:
    """
    Check which columns from every feature group exist in the dataset.

    Returns a validation table instead of silently inventing features.
    """

    rows = []

    for (
        group_name,
        columns
    ) in feature_groups.items():

        existing_columns = (
            get_existing_columns(
                df,
                columns
            )
        )

        missing_columns = [
            column
            for column in columns
            if column not in df.columns
        ]

        rows.append({
            "feature_group":
                group_name,

            "requested_columns":
                len(columns),

            "existing_columns":
                len(existing_columns),

            "missing_columns":
                len(missing_columns),

            "existing_column_names":
                " | ".join(
                    existing_columns
                ),

            "missing_column_names":
                " | ".join(
                    missing_columns
                ),
        })

    return pd.DataFrame(rows)


# =====================================================================
# SAFE NUMERIC VALUE PREPARATION
# =====================================================================

def prepare_numeric_crime_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert crime columns to numeric where possible.

    Missing values remain missing.

    No global fillna(0) is performed here.
    """

    result = df.copy()

    for column in result.columns:

        if (
            column.startswith("IPC_")
            or column.startswith("SC_")
            or column.startswith("ST_")
            or column.startswith("CHILDREN_")
            or column.startswith("WOMEN_")
        ):

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            )

    return result


# =====================================================================
# GROUPED FEATURE CREATION
# =====================================================================

def create_grouped_sum(
    df: pd.DataFrame,
    columns: list[str],
    output_name: str,
) -> pd.DataFrame:
    """
    Create a grouped crime feature from verified component columns.

    IMPORTANT:
        Missing values are ignored during the row-wise sum.

    min_count=1 ensures that a row containing only missing component
    values remains missing rather than becoming zero.
    """

    result = df.copy()

    existing_columns = (
        get_existing_columns(
            result,
            columns
        )
    )

    if not existing_columns:
        return result

    result[output_name] = (
        result[
            existing_columns
        ]
        .sum(
            axis=1,
            min_count=1
        )
    )

    return result


def create_feature_groups(
    df: pd.DataFrame,
    feature_groups: dict[str, list[str]]
) -> pd.DataFrame:
    """
    Create grouped crime features.

    The original columns are preserved.

    New grouped columns receive the prefix:

        GROUPED_
    """

    result = df.copy()

    for (
        group_name,
        columns
    ) in feature_groups.items():

        output_name = (
            f"GROUPED_{group_name}"
        )

        result = create_grouped_sum(
            result,
            columns,
            output_name
        )

    return result


# =====================================================================
# TIME FEATURES
# =====================================================================

def create_time_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create time-related features supported by the actual dataset.

    The dataset contains YEAR, but no verified MONTH/SEASON field.

    Therefore only year-derived features are created.
    """

    result = df.copy()

    if "YEAR" not in result.columns:
        return result

    result["YEAR"] = pd.to_numeric(
        result["YEAR"],
        errors="coerce"
    )

    # Keep the original YEAR.
    # Create a relative time index for modelling.
    minimum_year = result["YEAR"].min()

    if pd.notna(minimum_year):

        result["YEAR_INDEX"] = (
            result["YEAR"]
            - minimum_year
        )

    return result


# =====================================================================
# LOCATION FEATURES
# =====================================================================

def create_location_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare location features using only information actually present
    in the dataset.

    The dataset contains STATE and UNIT_NAME.

    No latitude, longitude, region, or district is invented.
    """

    result = df.copy()

    if "STATE" in result.columns:

        result["STATE"] = (
            result["STATE"]
            .astype("string")
            .str.strip()
        )

    if "UNIT_NAME" in result.columns:

        result["UNIT_NAME"] = (
            result["UNIT_NAME"]
            .astype("string")
            .str.strip()
        )

    return result


# =====================================================================
# TOTAL VIOLENT CRIME FEATURE
# =====================================================================

def create_violent_crime_total(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a broad violent-crime feature from the verified
    base-paper violent-crime groups.

    This is a predictor feature, NOT the classification target.

    IMPORTANT:
        This feature should only be used if it is not subsequently
        used to directly construct the target in a way that causes
        leakage.

    The individual grouped features are retained separately.
    """

    result = df.copy()

    grouped_columns = [
        f"GROUPED_{group}"
        for group in BASE_PAPER_FEATURE_GROUPS
        if f"GROUPED_{group}" in result.columns
    ]

    if not grouped_columns:
        return result

    result["VIOLENT_CRIME_TOTAL"] = (
        result[
            grouped_columns
        ]
        .sum(
            axis=1,
            min_count=1
        )
    )

    return result


# =====================================================================
# FEATURE CREATION PIPELINE
# =====================================================================

def engineer_features(
    df: pd.DataFrame,
    create_violent_total: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete feature-engineering pipeline.

    Returns
    -------
    engineered_df
        Dataset with engineered predictor features.

    validation_report
        Report showing which requested source columns actually exist.
    """

    result = df.copy()

    # ---------------------------------------------------------------
    # Numeric preparation
    # ---------------------------------------------------------------

    result = prepare_numeric_crime_columns(
        result
    )

    # ---------------------------------------------------------------
    # Validate base-paper groups
    # ---------------------------------------------------------------

    basepaper_validation = (
        validate_feature_groups(
            result,
            BASE_PAPER_FEATURE_GROUPS
        )
    )

    # ---------------------------------------------------------------
    # Validate additional crime groups
    # ---------------------------------------------------------------

    additional_validation = (
        validate_feature_groups(
            result,
            ADDITIONAL_CRIME_GROUPS
        )
    )

    validation_report = pd.concat(
        [
            basepaper_validation,
            additional_validation
        ],
        ignore_index=True
    )

    # ---------------------------------------------------------------
    # Create base-paper grouped features
    # ---------------------------------------------------------------

    result = create_feature_groups(
        result,
        BASE_PAPER_FEATURE_GROUPS
    )

    # ---------------------------------------------------------------
    # Create additional grouped features
    # ---------------------------------------------------------------

    result = create_feature_groups(
        result,
        ADDITIONAL_CRIME_GROUPS
    )

    # ---------------------------------------------------------------
    # Create time features
    # ---------------------------------------------------------------

    result = create_time_features(
        result
    )

    # ---------------------------------------------------------------
    # Create location features
    # ---------------------------------------------------------------

    result = create_location_features(
        result
    )

    # ---------------------------------------------------------------
    # Create broad violent-crime total
    # ---------------------------------------------------------------

    if create_violent_total:

        result = create_violent_crime_total(
            result
        )

    return (
        result,
        validation_report
    )


# =====================================================================
# FEATURE REPORT
# =====================================================================

def create_feature_summary(
    df: pd.DataFrame,
    original_columns: list[str]
) -> pd.DataFrame:
    """
    Summarize newly created features.
    """

    new_columns = [
        column
        for column in df.columns
        if column not in original_columns
    ]

    rows = []

    for column in new_columns:

        rows.append({
            "feature":
                column,

            "dtype":
                str(
                    df[column].dtype
                ),

            "missing_count":
                int(
                    df[column].isna().sum()
                ),

            "missing_percentage":
                round(
                    df[column].isna().mean() * 100,
                    2
                ),

            "unique_values":
                int(
                    df[column].nunique(
                        dropna=True
                    )
                ),
        })

    return pd.DataFrame(rows)


# =====================================================================
# SAVE FEATURE REPORTS
# =====================================================================

def save_feature_reports(
    validation_report: pd.DataFrame,
    feature_summary: pd.DataFrame,
    output_dir: Optional[str | Path] = None
) -> dict[str, Path]:
    """
    Save feature-engineering audit reports.
    """

    if output_dir is None:

        output_dir = (
            OUTPUT_TABLES_DIR
        )

    else:

        output_dir = Path(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    validation_path = (
        output_dir /
        "feature_group_validation.csv"
    )

    summary_path = (
        output_dir /
        "engineered_feature_summary.csv"
    )

    validation_report.to_csv(
        validation_path,
        index=False
    )

    feature_summary.to_csv(
        summary_path,
        index=False
    )

    return {
        "feature_group_validation":
            validation_path,

        "engineered_feature_summary":
            summary_path,
    }


# =====================================================================
# MAIN TEST
# =====================================================================

if __name__ == "__main__":

    from data_loading import load_crime_data

    print("=" * 70)
    print(
        "FEATURE ENGINEERING TEST"
    )
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load original integrated dataset
    # ---------------------------------------------------------------

    df = load_crime_data()

    print(
        f"\nOriginal dataset shape: "
        f"{df.shape}"
    )

    original_columns = (
        df.columns.tolist()
    )

    # ---------------------------------------------------------------
    # Engineer features
    # ---------------------------------------------------------------

    (
        engineered_df,
        validation_report
    ) = engineer_features(
        df,
        create_violent_total=True
    )

    # ---------------------------------------------------------------
    # Print validation report
    # ---------------------------------------------------------------

    print(
        "\nBASE-PAPER / ADDITIONAL FEATURE GROUP VALIDATION"
    )

    print("-" * 70)

    print(
        validation_report.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Feature summary
    # ---------------------------------------------------------------

    feature_summary = (
        create_feature_summary(
            engineered_df,
            original_columns
        )
    )

    print(
        "\nNEWLY CREATED FEATURES"
    )

    print("-" * 70)

    for feature in (
        feature_summary["feature"]
    ):

        print(
            f"- {feature}"
        )

    # ---------------------------------------------------------------
    # Shapes
    # ---------------------------------------------------------------

    print(
        "\nFEATURE ENGINEERING RESULT"
    )

    print("-" * 70)

    print(
        f"Original columns : "
        f"{len(original_columns)}"
    )

    print(
        f"Final columns    : "
        f"{len(engineered_df.columns)}"
    )

    print(
        f"New features     : "
        f"{len(feature_summary)}"
    )

    print(
        f"Rows             : "
        f"{len(engineered_df)}"
    )

    # ---------------------------------------------------------------
    # Save reports
    # ---------------------------------------------------------------

    report_paths = (
        save_feature_reports(
            validation_report,
            feature_summary
        )
    )

    print(
        "\nReports created:"
    )

    for (
        report_name,
        report_path
    ) in report_paths.items():

        print(
            f"{report_name}: "
            f"{report_path}"
        )

    # ---------------------------------------------------------------
    # Display engineered feature names
    # ---------------------------------------------------------------

    print(
        "\nENGINEERED COLUMN NAMES"
    )

    print("-" * 70)

    for column in engineered_df.columns:

        if column not in original_columns:

            print(
                f"- {column}"
            )

    # ---------------------------------------------------------------
    # Preview important engineered features
    # ---------------------------------------------------------------

    important_features = [
        column
        for column in engineered_df.columns
        if (
            column.startswith("GROUPED_")
            or column == "VIOLENT_CRIME_TOTAL"
            or column == "YEAR_INDEX"
        )
    ]

    if important_features:

        print(
            "\nENGINEERED FEATURE PREVIEW"
        )

        print("-" * 70)

        print(
            engineered_df[
                important_features
            ].head().to_string(
                index=False
            )
        )

    print(
        "\nNOTE:"
    )

    print(
        "No final classification target was created."
    )

    print(
        "No missing values were globally replaced with zero."
    )

    print(
        "Original source columns were retained."
    )