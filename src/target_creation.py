"""
Final target creation for the crime classification pipeline.

Target
------
TARGET_VIOLENT_LEVEL

Definition
----------
A project-defined five-level classification target based on the
distribution of the verified base-paper violent-crime burden.

Levels:

    1 = lowest violent-crime burden
    2 = low-to-moderate burden
    3 = moderate burden
    4 = moderate-to-high burden
    5 = highest violent-crime burden

The five levels are created using rank-based quintiles so that the
valid observations are distributed approximately equally across the
five classes.

IMPORTANT
---------
This is a PROJECT-DEFINED ADAPTATION.

It is NOT claimed to reproduce the exact hidden target-generation
procedure of the original base paper, because that procedure is not
explicitly specified in the available paper.

The violent-crime source variables used to construct the target must
NOT be used as predictors during model training.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "district_integrated_final.csv"
)

OUTPUT_TABLES_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)


# =====================================================================
# VERIFIED BASE-PAPER VIOLENT CRIME GROUPS
# =====================================================================

BASE_PAPER_GROUPS = {

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
# CREATE TARGET
# =====================================================================

def create_target(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create TARGET_VIOLENT_LEVEL.

    Returns
    -------
    dataframe
        Original dataframe plus target column.

    target_source_columns
        Original columns used to construct the target.
    """

    result = df.copy()

    group_totals = []

    target_source_columns = []

    # ---------------------------------------------------------------
    # Build each violent-crime group
    # ---------------------------------------------------------------

    for (
        group_name,
        columns
    ) in BASE_PAPER_GROUPS.items():

        existing_columns = [
            column
            for column in columns
            if column in result.columns
        ]

        target_source_columns.extend(
            existing_columns
        )

        if not existing_columns:
            continue

        numeric_values = (
            result[
                existing_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
        )

        group_total = (
            numeric_values
            .sum(
                axis=1,
                min_count=1
            )
        )

        group_totals.append(
            group_total
        )

    # ---------------------------------------------------------------
    # Calculate total violent-crime burden
    # ---------------------------------------------------------------

    if not group_totals:

        raise ValueError(
            "No verified violent-crime columns were found."
        )

    group_total_df = pd.concat(
        group_totals,
        axis=1
    )

    violent_burden = (
        group_total_df
        .sum(
            axis=1,
            min_count=1
        )
    )

    result[
        "VIOLENT_CRIME_BURDEN"
    ] = violent_burden

    # ---------------------------------------------------------------
    # Create five approximately equal levels
    # ---------------------------------------------------------------

    result[
        "TARGET_VIOLENT_LEVEL"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    valid_mask = (
        violent_burden.notna()
    )

    valid_burden = (
        violent_burden.loc[
            valid_mask
        ]
    )

    if len(valid_burden) == 0:

        raise ValueError(
            "No valid violent-crime burden values exist."
        )

    # Rank first to handle tied values.
    ranked_values = (
        valid_burden
        .rank(
            method="first"
        )
    )

    levels = pd.qcut(
        ranked_values,
        q=5,
        labels=[
            1,
            2,
            3,
            4,
            5
        ]
    )

    result.loc[
        valid_mask,
        "TARGET_VIOLENT_LEVEL"
    ] = (
        levels.astype(int)
    )

    # Remove duplicate source column names
    target_source_columns = list(
        dict.fromkeys(
            target_source_columns
        )
    )

    return (
        result,
        target_source_columns
    )


# =====================================================================
# TARGET DISTRIBUTION
# =====================================================================

def create_target_distribution(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create class distribution report.
    """

    counts = (
        df[
            "TARGET_VIOLENT_LEVEL"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    total = len(df)

    rows = []

    for level, count in counts.items():

        if pd.isna(level):

            label = "MISSING"

        else:

            label = int(level)

        rows.append({
            "target_level":
                label,

            "count":
                int(count),

            "percentage":
                round(
                    count / total * 100,
                    2
                ),
        })

    return pd.DataFrame(rows)


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "FINAL TARGET CREATION"
    )
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------

    print(
        f"\nLoading dataset:\n{DATA_PATH}"
    )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"\nOriginal shape: {df.shape}"
    )

    # ---------------------------------------------------------------
    # Create target
    # ---------------------------------------------------------------

    (
        target_df,
        target_source_columns
    ) = create_target(
        df
    )

    # ---------------------------------------------------------------
    # Print target information
    # ---------------------------------------------------------------

    print(
        "\nTARGET DEFINITION"
    )

    print("-" * 70)

    print(
        "Target column:"
    )

    print(
        "TARGET_VIOLENT_LEVEL"
    )

    print(
        "\nTarget levels:"
    )

    print(
        "1 = lowest violent-crime burden"
    )

    print(
        "2 = low-to-moderate burden"
    )

    print(
        "3 = moderate burden"
    )

    print(
        "4 = moderate-to-high burden"
    )

    print(
        "5 = highest violent-crime burden"
    )

    # ---------------------------------------------------------------
    # Source columns
    # ---------------------------------------------------------------

    print(
        "\nSOURCE COLUMNS USED TO CONSTRUCT TARGET"
    )

    print("-" * 70)

    for column in target_source_columns:

        print(
            f"- {column}"
        )

    # ---------------------------------------------------------------
    # Target distribution
    # ---------------------------------------------------------------

    distribution = (
        create_target_distribution(
            target_df
        )
    )

    print(
        "\nFINAL TARGET DISTRIBUTION"
    )

    print("-" * 70)

    print(
        distribution.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Target burden summary
    # ---------------------------------------------------------------

    print(
        "\nVIOLENT-CRIME BURDEN"
    )

    print("-" * 70)

    print(
        target_df[
            "VIOLENT_CRIME_BURDEN"
        ]
        .describe()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Save target distribution
    # ---------------------------------------------------------------

    OUTPUT_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    distribution_path = (
        OUTPUT_TABLES_DIR
        / "final_target_distribution.csv"
    )

    distribution.to_csv(
        distribution_path,
        index=False
    )

    # ---------------------------------------------------------------
    # Save target-source mapping
    # ---------------------------------------------------------------

    source_mapping = pd.DataFrame({
        "target":
            [
                "TARGET_VIOLENT_LEVEL"
            ]
            * len(target_source_columns),

        "source_column":
            target_source_columns,

        "role":
            [
                "target_construction_source"
            ]
            * len(target_source_columns),
    })

    source_mapping_path = (
        OUTPUT_TABLES_DIR
        / "target_source_column_mapping.csv"
    )

    source_mapping.to_csv(
        source_mapping_path,
        index=False
    )

    # ---------------------------------------------------------------
    # Save target-enabled dataset
    # ---------------------------------------------------------------

    target_dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "district_targeted.csv"
    )

    target_df.to_csv(
        target_dataset_path,
        index=False
    )

    # ---------------------------------------------------------------
    # Final output
    # ---------------------------------------------------------------

    print(
        "\nOUTPUT FILES"
    )

    print("-" * 70)

    print(
        f"Targeted dataset:\n"
        f"{target_dataset_path}"
    )

    print(
        f"\nTarget distribution:\n"
        f"{distribution_path}"
    )

    print(
        f"\nTarget source mapping:\n"
        f"{source_mapping_path}"
    )

    print(
        "\nFINAL SHAPE"
    )

    print("-" * 70)

    print(
        target_df.shape
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "TARGET_VIOLENT_LEVEL is a project-defined adaptation."
    )

    print(
        "The violent-crime source columns used to construct the"
        " target must be excluded from ML predictors."
    )

    print(
        "No train/test split or preprocessing was performed."
    )