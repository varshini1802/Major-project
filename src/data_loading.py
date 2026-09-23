"""
Data loading utilities for the crime analysis ML pipeline.

This module provides reusable functions for loading the existing
integrated crime dataset without modifying the original CSV file.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# Crime_Analysis/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Existing processed-data directory
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Original integrated dataset used for ML preparation
DEFAULT_DATASET_PATH = (
    PROCESSED_DATA_DIR / "district_integrated_final.csv"
)


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_crime_data(
    file_path: Optional[str | Path] = None
) -> pd.DataFrame:
    """
    Load the integrated crime dataset.

    Parameters
    ----------
    file_path : str or Path, optional
        Path to the CSV file.

        If None, the existing project dataset
        'district_integrated_final.csv' is used.

    Returns
    -------
    pandas.DataFrame
        Loaded crime dataset.

    Notes
    -----
    This function only loads the data.
    It does not:
        - remove columns
        - fill missing values
        - create targets
        - encode categories
        - scale features
        - balance classes

    Those operations will be handled by separate modules.
    """

    if file_path is None:
        file_path = DEFAULT_DATASET_PATH
    else:
        file_path = Path(file_path)

    file_path = file_path.resolve()

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, but received: {file_path.suffix}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            f"The dataset was loaded successfully but contains no rows:\n"
            f"{file_path}"
        )

    return df


# ---------------------------------------------------------------------
# Basic dataset information
# ---------------------------------------------------------------------

def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Return basic structural information about a dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to summarize.

    Returns
    -------
    dict
        Basic dataset statistics.
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_columns": len(numeric_columns),
        "categorical_columns": len(categorical_columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# ---------------------------------------------------------------------
# Column information
# ---------------------------------------------------------------------

def get_column_information(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a reusable summary of every column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to inspect.

    Returns
    -------
    pandas.DataFrame
        Column-level information including:
        - column name
        - data type
        - missing count
        - missing percentage
        - unique value count
    """

    information = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (
            df.isna().mean().mul(100).round(2).values
        ),
        "unique_values": df.nunique(
            dropna=False
        ).values,
    })

    return information


# ---------------------------------------------------------------------
# Data-loading validation
# ---------------------------------------------------------------------

def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str]
) -> None:
    """
    Check whether required columns are present.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to validate.

    required_columns : list[str]
        Columns that must exist.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )


# ---------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    df = load_crime_data()

    summary = get_dataset_summary(df)

    print("=" * 70)
    print("CRIME DATASET LOADED")
    print("=" * 70)

    print(f"Dataset path : {DEFAULT_DATASET_PATH}")
    print(f"Rows         : {summary['rows']}")
    print(f"Columns      : {summary['columns']}")
    print(f"Numeric cols : {summary['numeric_columns']}")
    print(f"Categorical  : {summary['categorical_columns']}")
    print(f"Missing cells: {summary['missing_cells']}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")

    print("\nFirst 10 columns:")
    print(df.columns[:10].tolist())

    print("\nFirst 5 rows:")
    print(df.head())