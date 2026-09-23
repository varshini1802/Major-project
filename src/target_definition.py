"""
Target definition audit for the crime classification project.

Purpose
-------
The base paper describes a five-class violent-crime classification
problem. However, the integrated 9,856-row dataset does not contain
an explicit target/class column.

This module therefore generates candidate target constructions for
inspection BEFORE selecting the final target.

IMPORTANT
---------
Candidate A:
    Five-level violent-crime burden using quantile-based bins.

Candidate B:
    Dominant violent-crime group.

These are PROJECT-DEFINED candidates.

They must NOT be described as an exact reproduction of the paper's
original target construction unless the original target-generation
rule is independently available.

No ML model is trained here.
"""


from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_TABLES_DIR = (
    PROJECT_ROOT / "outputs" / "tables"
)


# =====================================================================
# BASE-PAPER VIOLENT CRIME GROUPS
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
# HELPER: EXISTING COLUMNS
# =====================================================================

def get_existing_columns(
    df: pd.DataFrame,
    columns: list[str]
) -> list[str]:
    """
    Return only columns that exist in the dataset.
    """

    existing = set(df.columns)

    return [
        column
        for column in columns
        if column in existing
    ]


# =====================================================================
# CREATE VIOLENT-CRIME GROUP TOTALS
# =====================================================================

def create_violent_group_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create temporary violent-crime group totals.

    These are used ONLY for target-definition auditing.

    Missing values are preserved when every component is missing.
    """

    result = df.copy()

    for (
        group_name,
        columns
    ) in BASE_PAPER_GROUPS.items():

        existing_columns = (
            get_existing_columns(
                result,
                columns
            )
        )

        output_name = (
            f"_TARGETGROUP_{group_name}"
        )

        if not existing_columns:

            result[output_name] = np.nan

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

        result[output_name] = (
            numeric_values
            .sum(
                axis=1,
                min_count=1
            )
        )

    return result


# =====================================================================
# CREATE TOTAL VIOLENT-CRIME BURDEN
# =====================================================================

def create_violent_crime_burden(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a temporary total across the verified base-paper violent
    crime groups.

    This feature is for target auditing only.
    """

    result = df.copy()

    group_columns = [
        column
        for column in result.columns
        if column.startswith(
            "_TARGETGROUP_"
        )
    ]

    result["_TARGET_VIOLENT_CRIME_BURDEN"] = (
        result[
            group_columns
        ]
        .sum(
            axis=1,
            min_count=1
        )
    )

    return result


# =====================================================================
# CANDIDATE A: FIVE QUANTILE LEVELS
# =====================================================================

def create_five_level_candidate(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a candidate five-level target based on the distribution
    of total violent-crime burden.

    Labels:

        1 = lowest burden
        2
        3
        4
        5 = highest burden

    This is a PROJECT-DEFINED candidate.

    It is NOT claimed to reproduce the original paper's hidden
    target-generation procedure.
    """

    result = df.copy()

    burden = (
        result[
            "_TARGET_VIOLENT_CRIME_BURDEN"
        ]
    )

    valid = burden.notna()

    result[
        "_CANDIDATE_FIVE_LEVEL"
    ] = np.nan

    if valid.sum() == 0:

        return result

    valid_burden = (
        burden.loc[valid]
    )

    # Rank first so tied values do not collapse quantile bins.
    ranks = (
        valid_burden
        .rank(
            method="first"
        )
    )

    try:

        levels = pd.qcut(
            ranks,
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
            valid,
            "_CANDIDATE_FIVE_LEVEL"
        ] = (
            levels.astype(float)
        )

    except ValueError:

        # If five unique bins cannot be created,
        # leave the candidate undefined.
        pass

    return result


# =====================================================================
# CANDIDATE B: DOMINANT VIOLENT CRIME GROUP
# =====================================================================

def create_dominant_crime_candidate(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify the dominant violent-crime group for each row.

    The result is an exploratory candidate only.

    No consolidation into five classes is performed because the paper
    does not specify such a consolidation rule.
    """

    result = df.copy()

    group_columns = [
        column
        for column in result.columns
        if column.startswith(
            "_TARGETGROUP_"
        )
    ]

    if not group_columns:

        result[
            "_CANDIDATE_DOMINANT_GROUP"
        ] = pd.NA

        return result

    renamed = {
        column:
            column.replace(
                "_TARGETGROUP_",
                ""
            )
        for column in group_columns
    }

    group_values = (
        result[
            group_columns
        ]
        .rename(
            columns=renamed
        )
    )

    result[
        "_CANDIDATE_DOMINANT_GROUP"
    ] = (
        group_values
        .idxmax(
            axis=1,
            skipna=True
        )
    )

    # Rows where every violent-crime group is missing
    all_missing = (
        group_values
        .isna()
        .all(axis=1)
    )

    result.loc[
        all_missing,
        "_CANDIDATE_DOMINANT_GROUP"
    ] = pd.NA

    return result


# =====================================================================
# TARGET DISTRIBUTION REPORT
# =====================================================================

def create_target_distribution_report(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Create frequency and percentage distribution for a candidate.
    """

    counts = (
        df[column]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    total = len(df)

    rows = []

    for value, count in counts.items():

        if pd.isna(value):

            label = "MISSING"

        else:

            label = str(value)

        rows.append({
            "candidate":
                column,

            "class_or_group":
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
# SUMMARY OF VIOLENT CRIME BURDEN
# =====================================================================

def create_burden_summary(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create descriptive statistics for violent-crime burden.
    """

    series = (
        df[
            "_TARGET_VIOLENT_CRIME_BURDEN"
        ]
    )

    return pd.DataFrame({
        "statistic": [
            "count",
            "missing",
            "mean",
            "median",
            "minimum",
            "maximum",
            "std",
            "q25",
            "q75",
        ],

        "value": [
            int(
                series.notna().sum()
            ),

            int(
                series.isna().sum()
            ),

            series.mean(),

            series.median(),

            series.min(),

            series.max(),

            series.std(),

            series.quantile(
                0.25
            ),

            series.quantile(
                0.75
            ),
        ],
    })


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    from data_loading import load_crime_data

    print("=" * 70)
    print(
        "TARGET DEFINITION AUDIT"
    )
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------

    df = load_crime_data()

    print(
        f"\nDataset shape: {df.shape}"
    )

    # ---------------------------------------------------------------
    # Create temporary violent crime group totals
    # ---------------------------------------------------------------

    audit_df = (
        create_violent_group_features(
            df
        )
    )

    # ---------------------------------------------------------------
    # Create total violent-crime burden
    # ---------------------------------------------------------------

    audit_df = (
        create_violent_crime_burden(
            audit_df
        )
    )

    print(
        "\nVIOLENT-CRIME BURDEN SUMMARY"
    )

    print("-" * 70)

    burden_summary = (
        create_burden_summary(
            audit_df
        )
    )

    print(
        burden_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Candidate A
    # ---------------------------------------------------------------

    audit_df = (
        create_five_level_candidate(
            audit_df
        )
    )

    print(
        "\nCANDIDATE A: FIVE-LEVEL VIOLENT-CRIME BURDEN"
    )

    print("-" * 70)

    five_level_distribution = (
        create_target_distribution_report(
            audit_df,
            "_CANDIDATE_FIVE_LEVEL"
        )
    )

    print(
        five_level_distribution.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Candidate B
    # ---------------------------------------------------------------

    audit_df = (
        create_dominant_crime_candidate(
            audit_df
        )
    )

    print(
        "\nCANDIDATE B: DOMINANT VIOLENT-CRIME GROUP"
    )

    print("-" * 70)

    dominant_distribution = (
        create_target_distribution_report(
            audit_df,
            "_CANDIDATE_DOMINANT_GROUP"
        )
    )

    print(
        dominant_distribution.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Save reports
    # ---------------------------------------------------------------

    burden_path = (
        OUTPUT_TABLES_DIR /
        "target_violent_crime_burden_summary.csv"
    )

    five_level_path = (
        OUTPUT_TABLES_DIR /
        "candidate_five_level_distribution.csv"
    )

    dominant_path = (
        OUTPUT_TABLES_DIR /
        "candidate_dominant_crime_distribution.csv"
    )

    group_columns = [
        column
        for column in audit_df.columns
        if column.startswith(
            "_TARGETGROUP_"
        )
    ]

    group_summary = (
        audit_df[
            group_columns
        ]
        .describe()
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index":
                    "violent_crime_group"
            }
        )
    )

    group_summary_path = (
        OUTPUT_TABLES_DIR /
        "target_violent_crime_group_summary.csv"
    )

    burden_summary.to_csv(
        burden_path,
        index=False
    )

    five_level_distribution.to_csv(
        five_level_path,
        index=False
    )

    dominant_distribution.to_csv(
        dominant_path,
        index=False
    )

    group_summary.to_csv(
        group_summary_path,
        index=False
    )

    # ---------------------------------------------------------------
    # Print paths
    # ---------------------------------------------------------------

    print(
        "\nREPORTS CREATED"
    )

    print("-" * 70)

    print(
        f"Violent-crime burden:\n"
        f"{burden_path}"
    )

    print(
        f"\nFive-level candidate:\n"
        f"{five_level_path}"
    )

    print(
        f"\nDominant-group candidate:\n"
        f"{dominant_path}"
    )

    print(
        f"\nViolent-crime group summary:\n"
        f"{group_summary_path}"
    )

    print(
        "\nIMPORTANT"
    )

    print(
        "No final target has been selected."
    )

    print(
        "No ML target column has been added to the dataset."
    )

    print(
        "The candidate targets are for methodological inspection."
    )