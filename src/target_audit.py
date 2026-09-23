"""
Target audit for the base-paper crime classification pipeline.

Purpose
-------
The base paper describes a five-class violent-crime classification
problem.

Before constructing a new target, this script checks whether the
integrated dataset already contains a suitable target/class/level
column or any variable that may correspond to the paper's class
representation.

IMPORTANT
---------
This script does NOT create a target.

It only audits the existing dataset.

No rows or columns are modified.
"""

from pathlib import Path

import pandas as pd


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_TABLES_DIR = (
    PROJECT_ROOT / "outputs" / "tables"
)


# =====================================================================
# TARGET-RELATED KEYWORDS
# =====================================================================

TARGET_KEYWORDS = [
    "TARGET",
    "CLASS",
    "LABEL",
    "LEVEL",
    "CATEGORY",
    "GROUP",
    "SEVERITY",
    "RANK",
    "SCORE",
]


# =====================================================================
# SEARCH DATASET COLUMNS
# =====================================================================

def search_target_like_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Search column names for terms that may indicate a target,
    class, level, category, label, score, or severity variable.
    """

    rows = []

    for column in df.columns:

        column_upper = (
            str(column)
            .strip()
            .upper()
        )

        matched_keywords = [
            keyword
            for keyword in TARGET_KEYWORDS
            if keyword in column_upper
        ]

        if matched_keywords:

            rows.append({
                "column":
                    column,

                "matched_keywords":
                    ", ".join(
                        matched_keywords
                    ),

                "dtype":
                    str(
                        df[column].dtype
                    ),

                "non_missing_count":
                    int(
                        df[column].notna().sum()
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
# LOW-CARDINALITY COLUMN AUDIT
# =====================================================================

def find_low_cardinality_numeric_columns(
    df: pd.DataFrame,
    maximum_unique_values: int = 20
) -> pd.DataFrame:
    """
    Identify numeric columns with relatively few unique values.

    Such columns are worth inspecting because a classification label
    may be encoded numerically.
    """

    rows = []

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    for column in numeric_columns:

        unique_count = int(
            df[column].nunique(
                dropna=True
            )
        )

        if (
            unique_count >= 2
            and unique_count
            <= maximum_unique_values
        ):

            rows.append({
                "column":
                    column,

                "unique_values":
                    unique_count,

                "missing_count":
                    int(
                        df[column].isna().sum()
                    ),

                "missing_percentage":
                    round(
                        df[column].isna().mean() * 100,
                        2
                    ),
            })

    result = pd.DataFrame(rows)

    if not result.empty:

        result = result.sort_values(
            [
                "unique_values",
                "column"
            ]
        ).reset_index(
            drop=True
        )

    return result


# =====================================================================
# OBJECT / CATEGORICAL COLUMN AUDIT
# =====================================================================

def audit_categorical_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Audit categorical/string columns that could potentially represent
    a class or level.
    """

    rows = []

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        ).columns
    )

    for column in categorical_columns:

        unique_count = int(
            df[column].nunique(
                dropna=True
            )
        )

        rows.append({
            "column":
                column,

            "unique_values":
                unique_count,

            "missing_count":
                int(
                    df[column].isna().sum()
                ),

            "missing_percentage":
                round(
                    df[column].isna().mean() * 100,
                    2
                ),
        })

    result = pd.DataFrame(rows)

    if not result.empty:

        result = result.sort_values(
            [
                "unique_values",
                "column"
            ]
        ).reset_index(
            drop=True
        )

    return result


# =====================================================================
# VALUE DISTRIBUTION
# =====================================================================

def print_candidate_value_distributions(
    df: pd.DataFrame,
    candidate_columns: list[str]
) -> None:
    """
    Print value distributions for candidate target columns.
    """

    for column in candidate_columns:

        if column not in df.columns:
            continue

        print(
            f"\nCOLUMN: {column}"
        )

        print("-" * 70)

        print(
            df[column]
            .value_counts(
                dropna=False
            )
            .head(20)
            .to_string()
        )


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    from data_loading import load_crime_data

    print("=" * 70)
    print(
        "TARGET / CLASSIFICATION LABEL AUDIT"
    )
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load original dataset
    # ---------------------------------------------------------------

    df = load_crime_data()

    print(
        f"\nDataset shape: {df.shape}"
    )

    # ---------------------------------------------------------------
    # Search target-like column names
    # ---------------------------------------------------------------

    target_like = (
        search_target_like_columns(
            df
        )
    )

    target_like_path = (
        OUTPUT_TABLES_DIR /
        "target_like_columns.csv"
    )

    target_like.to_csv(
        target_like_path,
        index=False
    )

    print(
        "\nTARGET-LIKE COLUMNS"
    )

    print("-" * 70)

    if target_like.empty:

        print(
            "No target-like column names found."
        )

    else:

        print(
            target_like.to_string(
                index=False
            )
        )

    # ---------------------------------------------------------------
    # Low-cardinality numeric columns
    # ---------------------------------------------------------------

    low_cardinality = (
        find_low_cardinality_numeric_columns(
            df
        )
    )

    low_cardinality_path = (
        OUTPUT_TABLES_DIR /
        "low_cardinality_numeric_columns.csv"
    )

    low_cardinality.to_csv(
        low_cardinality_path,
        index=False
    )

    print(
        "\nLOW-CARDINALITY NUMERIC COLUMNS"
    )

    print("-" * 70)

    if low_cardinality.empty:

        print(
            "No low-cardinality numeric columns found."
        )

    else:

        print(
            low_cardinality.to_string(
                index=False
            )
        )

    # ---------------------------------------------------------------
    # Categorical columns
    # ---------------------------------------------------------------

    categorical = (
        audit_categorical_columns(
            df
        )
    )

    categorical_path = (
        OUTPUT_TABLES_DIR /
        "categorical_column_audit.csv"
    )

    categorical.to_csv(
        categorical_path,
        index=False
    )

    print(
        "\nCATEGORICAL COLUMNS"
    )

    print("-" * 70)

    if categorical.empty:

        print(
            "No categorical columns found."
        )

    else:

        print(
            categorical.to_string(
                index=False
            )
        )

    # ---------------------------------------------------------------
    # Candidate distributions
    # ---------------------------------------------------------------

    candidate_columns = []

    if not target_like.empty:

        candidate_columns.extend(
            target_like[
                "column"
            ].tolist()
        )

    # Remove duplicates while preserving order
    candidate_columns = list(
        dict.fromkeys(
            candidate_columns
        )
    )

    if candidate_columns:

        print(
            "\nCANDIDATE VALUE DISTRIBUTIONS"
        )

        print("-" * 70)

        print_candidate_value_distributions(
            df,
            candidate_columns
        )

    # ---------------------------------------------------------------
    # Final report paths
    # ---------------------------------------------------------------

    print(
        "\nREPORTS CREATED"
    )

    print("-" * 70)

    print(
        f"Target-like columns:\n"
        f"{target_like_path}"
    )

    print(
        f"\nLow-cardinality numeric columns:\n"
        f"{low_cardinality_path}"
    )

    print(
        f"\nCategorical column audit:\n"
        f"{categorical_path}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No classification target was created."
    )

    print(
        "The audit only identifies existing candidate columns."
    )