"""
Data cleaning and data-quality utilities for the crime analysis pipeline.

This module performs safe structural cleaning and auditing.

Important:
    - The original integrated CSV is never overwritten.
    - Missing crime values are NOT globally replaced with zero.
    - Potentially redundant columns are reported, not blindly deleted.
    - Crime grouping and target creation are handled by separate modules.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"


# ---------------------------------------------------------------------
# Important dataset columns
# ---------------------------------------------------------------------

IDENTIFICATION_COLUMNS = [
    "STATE",
    "UNIT_NAME",
]

TIME_COLUMNS = [
    "YEAR",
]

AVAILABILITY_COLUMNS = [
    "IPC_AVAILABLE",
    "SC_AVAILABLE",
    "ST_AVAILABLE",
    "CHILDREN_AVAILABLE",
    "WOMEN_AVAILABLE",
]


# ---------------------------------------------------------------------
# Basic structural cleaning
# ---------------------------------------------------------------------

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names without changing their meaning.

    Operations:
        - remove leading/trailing whitespace
        - replace repeated whitespace with a single space
        - preserve the existing naming convention otherwise

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset with cleaned column names.
    """

    cleaned_df = df.copy()

    cleaned_columns = (
        pd.Index(cleaned_df.columns)
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    cleaned_df.columns = cleaned_columns

    return cleaned_df


# ---------------------------------------------------------------------
# String-column cleaning
# ---------------------------------------------------------------------

def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean whitespace in text/categorical columns.

    Missing values are preserved.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset with cleaned text fields.
    """

    cleaned_df = df.copy()

    text_columns = cleaned_df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        cleaned_df[column] = cleaned_df[column].apply(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    return cleaned_df


# ---------------------------------------------------------------------
# Numeric-column conversion
# ---------------------------------------------------------------------

def convert_numeric_columns(
    df: pd.DataFrame,
    exclude_columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Convert columns that are intended to be numeric.

    Only columns that can be safely converted are changed.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    exclude_columns : list[str], optional
        Columns that must remain unchanged.

    Returns
    -------
    pandas.DataFrame
        Dataset with numeric columns converted where appropriate.
    """

    cleaned_df = df.copy()

    if exclude_columns is None:
        exclude_columns = (
            IDENTIFICATION_COLUMNS
            + AVAILABILITY_COLUMNS
        )

    exclude_columns = set(exclude_columns)

    for column in cleaned_df.columns:

        if column in exclude_columns:
            continue

        # Attempt conversion only when the column is not already
        # numeric and contains values that can reasonably be numeric.
        if not pd.api.types.is_numeric_dtype(
            cleaned_df[column]
        ):
            converted = pd.to_numeric(
                cleaned_df[column],
                errors="coerce"
            )

            original_non_null = cleaned_df[column].notna().sum()
            converted_non_null = converted.notna().sum()

            # Accept conversion only when essentially all original
            # non-null values can be represented numerically.
            if (
                original_non_null == 0
                or converted_non_null / original_non_null >= 0.95
            ):
                cleaned_df[column] = converted

    return cleaned_df


# ---------------------------------------------------------------------
# Duplicate-row detection
# ---------------------------------------------------------------------

def find_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return duplicate rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    pandas.DataFrame
        Duplicate rows.
    """

    duplicate_mask = df.duplicated(
        keep=False
    )

    return df.loc[duplicate_mask].copy()


def remove_duplicate_rows(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        Dataset without exact duplicate rows.

    removed_count : int
        Number of removed rows.
    """

    before = len(df)

    cleaned_df = df.drop_duplicates(
        keep="first"
    ).copy()

    removed_count = before - len(cleaned_df)

    return cleaned_df, removed_count


# ---------------------------------------------------------------------
# Duplicate-column detection
# ---------------------------------------------------------------------

def find_exact_duplicate_columns(
    df: pd.DataFrame
) -> dict[str, list[str]]:
    """
    Identify columns containing exactly the same values.

    IMPORTANT:
        These columns are only reported.
        They are NOT automatically removed.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    dict
        Mapping of representative column names to duplicate columns.
    """

    duplicate_groups = {}
    processed_columns = set()

    columns = list(df.columns)

    for i, column in enumerate(columns):

        if column in processed_columns:
            continue

        duplicates = []

        for other_column in columns[i + 1:]:

            if other_column in processed_columns:
                continue

            if df[column].equals(df[other_column]):
                duplicates.append(other_column)

        if duplicates:
            duplicate_groups[column] = duplicates
            processed_columns.update(duplicates)

    return duplicate_groups


# ---------------------------------------------------------------------
# Missing-value audit
# ---------------------------------------------------------------------

def create_missing_value_report(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a column-level missing-value report.

    Missing values are measured but NOT filled.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    pandas.DataFrame
        Missing-value statistics for every column.
    """

    report = pd.DataFrame({
        "column": df.columns,
        "dtype": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "missing_count": df.isna().sum().values,
        "total_rows": len(df),
        "missing_percentage": (
            df.isna().mean() * 100
        ).round(2).values,
        "non_missing_count": (
            df.notna().sum().values
        ),
        "unique_values": (
            df.nunique(dropna=True).values
        ),
    })

    report = report.sort_values(
        by="missing_percentage",
        ascending=False
    ).reset_index(drop=True)

    return report


# ---------------------------------------------------------------------
# Missingness by reporting block
# ---------------------------------------------------------------------

def create_availability_report(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarize the availability flags in the integrated dataset.

    The dataset contains separate availability indicators for
    IPC, SC, ST, Children and Women reporting blocks.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    pandas.DataFrame
        Availability summary.
    """

    available_columns = [
        column
        for column in AVAILABILITY_COLUMNS
        if column in df.columns
    ]

    if not available_columns:
        return pd.DataFrame(
            columns=[
                "availability_column",
                "available_count",
                "unavailable_count",
                "missing_count",
                "available_percentage",
            ]
        )

    rows = []

    for column in available_columns:

        series = df[column]

        true_count = int(
            (series == True).sum()
        )

        false_count = int(
            (series == False).sum()
        )

        missing_count = int(
            series.isna().sum()
        )

        non_missing = true_count + false_count

        available_percentage = (
            (true_count / non_missing) * 100
            if non_missing > 0
            else 0.0
        )

        rows.append({
            "availability_column": column,
            "available_count": true_count,
            "unavailable_count": false_count,
            "missing_count": missing_count,
            "available_percentage": round(
                available_percentage,
                2
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Column classification helper
# ---------------------------------------------------------------------

def classify_column_type(
    column_name: str
) -> str:
    """
    Assign a preliminary structural category to a column.

    This is NOT the final crime-group classification.
    Final crime grouping is handled separately.

    Parameters
    ----------
    column_name : str
        Column name.

    Returns
    -------
    str
        Preliminary column category.
    """

    if column_name in IDENTIFICATION_COLUMNS:
        return "IDENTIFICATION_LOCATION"

    if column_name in TIME_COLUMNS:
        return "TIME"

    if column_name in AVAILABILITY_COLUMNS:
        return "AVAILABILITY_FLAG"

    column_upper = column_name.upper()

    if column_upper.startswith("IPC_"):
        return "IPC"

    if column_upper.startswith("SC_"):
        return "SC"

    if column_upper.startswith("ST_"):
        return "ST"

    if column_upper.startswith("CHILDREN_"):
        return "CHILDREN"

    if column_upper.startswith("WOMEN_"):
        return "WOMEN"

    return "OTHER"


def create_column_classification_report(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a preliminary structural classification report.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    pandas.DataFrame
        Column classification report.
    """

    report = pd.DataFrame({
        "column": df.columns,
        "structural_group": [
            classify_column_type(column)
            for column in df.columns
        ],
        "dtype": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "missing_percentage": (
            df.isna().mean() * 100
        ).round(2).values,
        "unique_values": (
            df.nunique(dropna=True).values
        ),
    })

    return report


# ---------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------

def clean_crime_data(
    df: pd.DataFrame,
    remove_duplicate_rows_flag: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Perform safe structural cleaning.

    The function:
        1. Copies the original DataFrame.
        2. Cleans column-name whitespace.
        3. Cleans text-column whitespace.
        4. Converts safely numeric columns.
        5. Optionally removes exact duplicate rows.

    It does NOT:
        - fill missing values
        - delete high-missingness columns
        - remove duplicate columns
        - create crime groups
        - create a target
        - perform balancing
        - perform scaling

    Parameters
    ----------
    df : pandas.DataFrame
        Raw loaded dataset.

    remove_duplicate_rows_flag : bool
        Whether exact duplicate rows should be removed.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        Structurally cleaned dataset.

    audit : dict
        Cleaning audit information.
    """

    cleaned_df = df.copy()

    original_shape = cleaned_df.shape

    # 1. Clean column names
    cleaned_df = clean_column_names(
        cleaned_df
    )

    # 2. Clean text fields
    cleaned_df = clean_text_columns(
        cleaned_df
    )

    # 3. Convert numeric fields safely
    cleaned_df = convert_numeric_columns(
        cleaned_df
    )

    # 4. Remove exact duplicate rows if requested
    duplicate_rows_before = int(
        cleaned_df.duplicated().sum()
    )

    if remove_duplicate_rows_flag:
        cleaned_df, removed_duplicate_rows = (
            remove_duplicate_rows(cleaned_df)
        )
    else:
        removed_duplicate_rows = 0

    final_shape = cleaned_df.shape

    audit = {
        "original_rows": original_shape[0],
        "original_columns": original_shape[1],
        "final_rows": final_shape[0],
        "final_columns": final_shape[1],
        "duplicate_rows_detected": duplicate_rows_before,
        "duplicate_rows_removed": removed_duplicate_rows,
    }

    return cleaned_df, audit


# ---------------------------------------------------------------------
# Save audit reports
# ---------------------------------------------------------------------

def save_cleaning_reports(
    df: pd.DataFrame,
    output_dir: Optional[str | Path] = None
) -> dict[str, Path]:
    """
    Generate and save data-quality reports.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset being audited.

    output_dir : str or Path, optional
        Directory where reports will be saved.

    Returns
    -------
    dict[str, Path]
        Paths of generated reports.
    """

    if output_dir is None:
        output_dir = OUTPUT_TABLES_DIR
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    missing_report = create_missing_value_report(
        df
    )

    availability_report = create_availability_report(
        df
    )

    classification_report = (
        create_column_classification_report(df)
    )

    missing_path = (
        output_dir / "missing_value_report.csv"
    )

    availability_path = (
        output_dir / "availability_report.csv"
    )

    classification_path = (
        output_dir / "column_classification_report.csv"
    )

    missing_report.to_csv(
        missing_path,
        index=False
    )

    availability_report.to_csv(
        availability_path,
        index=False
    )

    classification_report.to_csv(
        classification_path,
        index=False
    )

    return {
        "missing_value_report": missing_path,
        "availability_report": availability_path,
        "column_classification_report": classification_path,
    }


# ---------------------------------------------------------------------
# Module test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    from data_loading import load_crime_data

    print("=" * 70)
    print("DATA CLEANING / AUDIT TEST")
    print("=" * 70)

    # Load original integrated dataset
    df = load_crime_data()

    print("\nOriginal shape:")
    print(df.shape)

    # Perform safe structural cleaning
    cleaned_df, audit = clean_crime_data(df)

    print("\nCleaning audit:")
    for key, value in audit.items():
        print(f"{key}: {value}")

    print("\nCleaned shape:")
    print(cleaned_df.shape)

    # Create reports
    report_paths = save_cleaning_reports(
        cleaned_df
    )

    print("\nReports created:")

    for report_name, report_path in report_paths.items():
        print(f"{report_name}: {report_path}")

    # Show missing-value summary
    missing_report = create_missing_value_report(
        cleaned_df
    )

    print("\nTop 20 columns by missing percentage:")

    print(
        missing_report.head(20).to_string(
            index=False
        )
    )

    # Show availability information
    availability_report = create_availability_report(
        cleaned_df
    )

    print("\nAvailability report:")

    print(
        availability_report.to_string(
            index=False
        )
    )

    # Check duplicate columns
    duplicate_columns = (
        find_exact_duplicate_columns(
            cleaned_df
        )
    )

    print(
        f"\nExact duplicate-column groups detected: "
        f"{len(duplicate_columns)}"
    )

    print("\nNOTE:")
    print(
        "Duplicate columns are reported only. "
        "They have NOT been deleted."
    )