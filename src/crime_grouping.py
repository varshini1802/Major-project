"""
Crime grouping and base-paper feature mapping utilities.

This module:

1. Classifies all dataset columns into structural blocks.
2. Classifies crime columns into preliminary crime families.
3. Identifies aggregate/total columns.
4. Identifies currently unclassified crime columns.
5. Verifies the violent-crime attributes mentioned in the base paper.
6. Searches the dataset for alternative names of missing paper attributes.
7. Saves all inspection reports to outputs/tables/.

IMPORTANT
---------
This module does NOT:

- modify the original CSV
- fill missing values
- delete columns
- create the ML target
- create final aggregated crime features
- perform train/test splitting
- perform class balancing

Those steps will be handled later.
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
# STRUCTURAL GROUPS
# =====================================================================

STRUCTURAL_GROUPS = [
    "IDENTIFICATION_LOCATION",
    "TIME",
    "AVAILABILITY",
    "IPC",
    "SC",
    "ST",
    "CHILDREN",
    "WOMEN",
    "OTHER",
]


# =====================================================================
# CRIME-FAMILY KEYWORDS
# =====================================================================

CRIME_FAMILY_KEYWORDS = {

    "MURDER_HOMICIDE": [
        "MURDER",
        "CULPABLE_HOMICIDE",
        "INFANTICIDE",
    ],

    "ATTEMPT_MURDER": [
        "ATTEMPT_TO_MURDER",
    ],

    "RAPE_SEXUAL_OFFENCES": [
        "RAPE",
        "SEXUAL_HARASSMENT",
        "SEXUAL_ASSAULT",
        "VOYEURISM",
        "STALKING",
        "OUTRAGE_HER_MODESTY",
        "INSULT_TO_MODESTY",
    ],

    "KIDNAPPING_ABDUCTION": [
        "KIDNAPPING",
        "KIDNAPING",
        "ABDUCTION",
    ],

    "DACOITY_ROBBERY": [
        "DACOITY",
        "ROBBERY",
    ],

    "THEFT_BURGLARY_TRESPASS": [
        "THEFT",
        "BURGLARY",
        "TRESPASS",
        "HOUSE_BREAKING",
        "HOUSE_TRESPASS",
    ],

    "RIOTS_PUBLIC_ORDER": [
        "RIOT",
        "RIOTS",
        "UNLAWFUL_ASSEMBLY",
        "ASSEMBLY_FOR_DACOITY",
    ],

    "ARSON": [
        "ARSON",
    ],

    "CHEATING_FRAUD": [
        "CHEATING",
        "FORGERY",
        "COUNTERFEITING",
        "FRAUD",
    ],

    "EXTORTION": [
        "EXTORTION",
    ],

    "CRIMINAL_BREACH_OF_TRUST": [
        "CRIMINAL_BREACH_OF_TRUST",
    ],

    "HURT_ASSAULT": [
        "HURT",
        "GRIEVOUS_HURT",
    ],

    "HUMAN_TRAFFICKING": [
        "HUMANTRAFFICKING",
        "HUMAN_TRAFFICKING",
        "TRAFFICKING",
    ],

    "DOWRY_DEATH": [
        "DOWRY_DEATH",
        "DOWRY_DEATHS",
    ],

    "CRIMES_AGAINST_STATE": [
        "OFFENCES_AGAINST_STATE",
        "OFFENSES_AGAINST_STATE",
        "SEDITION",
    ],

    "RASH_DRIVING": [
        "RASH_DRIVING",
    ],

    "UNNATURAL_OFFENCE": [
        "UNNATURAL_OFFENCE",
    ],
}


# =====================================================================
# AGGREGATE / TOTAL DETECTION
# =====================================================================

TOTAL_KEYWORDS = [
    "TOTAL",
    "OTHER_CRIMES",
    "OTHER_IPC_CRIMES",
]


# =====================================================================
# BASE-PAPER VIOLENT CRIME MAPPING
# =====================================================================

BASE_PAPER_VIOLENT_CRIMES = {

    "CHNOT_AMOUNTING_TO_MURDER": [
        "IPC_CULPABLE_HOMICIDE_NOT_AMOUNTING_TO_MURDER",
        "IPC_CULPABLE_HOMICIDE",
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
        "IPC_OTHER_RIOTS",
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
        "IPC_OTHER_MURDER",
    ],

    "PREPARATION_FOR_DACOITY": [
        "IPC_PREPARATION_FOR_DACOITY",
        "IPC_PREPARATION_OF_DACOITY",
    ],

    "DOWRY_DEATH": [
        "IPC_DOWRY_DEATHS",
        "IPC_DOWRY_DEATH",
    ],

    "ROBBERY": [
        "IPC_ROBBERY",
        "IPC_OTHER_ROBBERY",
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
        "IPC_KIDNAPPING_FOR_MARRIAGE",
    ],

    "ASSEMBLY_FOR_DACOITY": [
        "IPC_ASSEMBLY_FOR_DACOITY",
        "IPC_ASSEMBLY_OF_PERSONS_FOR_DACOITY",
    ],

    "ARSON": [
        "IPC_ARSON",
    ],

    "ATTEMPT_TO_MURDER": [
        "IPC_ATTEMPT_TO_MURDER",
    ],
}


# =====================================================================
# STRUCTURAL CLASSIFICATION
# =====================================================================

def classify_structural_group(
    column_name: str
) -> str:
    """
    Identify the major structural block of a dataset column.
    """

    column_upper = str(
        column_name
    ).strip().upper()

    if column_upper in {
        "STATE",
        "UNIT_NAME",
    }:
        return "IDENTIFICATION_LOCATION"

    if column_upper == "YEAR":
        return "TIME"

    if column_upper.endswith("_AVAILABLE"):
        return "AVAILABILITY"

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


# =====================================================================
# AGGREGATE DETECTION
# =====================================================================

def is_aggregate_column(
    column_name: str
) -> bool:
    """
    Determine whether a column appears to represent a total,
    aggregate, or broad category.
    """

    column_upper = str(
        column_name
    ).strip().upper()

    return any(
        keyword in column_upper
        for keyword in TOTAL_KEYWORDS
    )


# =====================================================================
# CRIME FAMILY CLASSIFICATION
# =====================================================================

def classify_crime_family(
    column_name: str
) -> str:
    """
    Assign a preliminary crime family to a crime column.
    """

    column_upper = str(
        column_name
    ).strip().upper()

    structural_group = (
        classify_structural_group(
            column_upper
        )
    )

    if structural_group == "IDENTIFICATION_LOCATION":
        return "IDENTIFICATION_LOCATION"

    if structural_group == "TIME":
        return "TIME"

    if structural_group == "AVAILABILITY":
        return "AVAILABILITY"

    if structural_group == "OTHER":
        return "OTHER"

    matched_families = []

    for family, keywords in (
        CRIME_FAMILY_KEYWORDS.items()
    ):

        for keyword in keywords:

            if keyword in column_upper:

                matched_families.append(
                    family
                )

                break

    if matched_families:
        return matched_families[0]

    return "OTHER_CRIME"


# =====================================================================
# COMPLETE COLUMN MAPPING
# =====================================================================

def create_crime_column_mapping(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a complete classification table for all dataset columns.
    """

    rows = []

    for column in df.columns:

        structural_group = (
            classify_structural_group(
                column
            )
        )

        aggregate_flag = (
            is_aggregate_column(
                column
            )
        )

        crime_family = (
            classify_crime_family(
                column
            )
        )

        rows.append({
            "column": column,

            "structural_group":
                structural_group,

            "crime_family":
                crime_family,

            "is_aggregate":
                aggregate_flag,

            "dtype":
                str(df[column].dtype),

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
# STRUCTURAL GROUPING
# =====================================================================

def get_columns_by_structural_group(
    mapping: pd.DataFrame
) -> dict[str, list[str]]:
    """
    Group columns by their major structural block.
    """

    grouped = {}

    for group in STRUCTURAL_GROUPS:

        grouped[group] = (
            mapping.loc[
                mapping["structural_group"] == group,
                "column"
            ].tolist()
        )

    return grouped


# =====================================================================
# CRIME FAMILY GROUPING
# =====================================================================

def get_columns_by_crime_family(
    mapping: pd.DataFrame
) -> dict[str, list[str]]:
    """
    Group crime columns by preliminary crime family.
    """

    grouped = {}

    crime_mapping = mapping[
        mapping["structural_group"].isin([
            "IPC",
            "SC",
            "ST",
            "CHILDREN",
            "WOMEN",
        ])
    ]

    for family in sorted(
        crime_mapping[
            "crime_family"
        ].unique()
    ):

        grouped[family] = (
            crime_mapping.loc[
                crime_mapping[
                    "crime_family"
                ] == family,
                "column"
            ].tolist()
        )

    return grouped


# =====================================================================
# AGGREGATE / DETAILED SEPARATION
# =====================================================================

def separate_aggregate_columns(
    mapping: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """
    Separate aggregate/total columns from non-aggregate columns.
    """

    aggregate_columns = (
        mapping.loc[
            mapping["is_aggregate"] == True,
            "column"
        ].tolist()
    )

    detailed_columns = (
        mapping.loc[
            mapping["is_aggregate"] == False,
            "column"
        ].tolist()
    )

    return (
        aggregate_columns,
        detailed_columns
    )


# =====================================================================
# UNCLASSIFIED CRIME COLUMNS
# =====================================================================

def get_unclassified_crime_columns(
    mapping: pd.DataFrame
) -> list[str]:
    """
    Return crime columns currently classified as OTHER_CRIME.
    """

    mask = (
        mapping[
            "structural_group"
        ].isin([
            "IPC",
            "SC",
            "ST",
            "CHILDREN",
            "WOMEN",
        ])
        &
        (
            mapping[
                "crime_family"
            ] == "OTHER_CRIME"
        )
    )

    return (
        mapping.loc[
            mask,
            "column"
        ].tolist()
    )


def create_unclassified_crime_report(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a detailed report of currently unclassified crime columns.
    """

    mapping = (
        create_crime_column_mapping(
            df
        )
    )

    columns = (
        get_unclassified_crime_columns(
            mapping
        )
    )

    report = mapping[
        mapping["column"].isin(
            columns
        )
    ].copy()

    return report.sort_values(
        [
            "structural_group",
            "column"
        ]
    ).reset_index(
        drop=True
    )


# =====================================================================
# BASE-PAPER COLUMN VERIFICATION
# =====================================================================

def verify_base_paper_violent_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Verify which base-paper violent-crime columns exist in the
    integrated dataset.
    """

    rows = []

    existing_columns = set(
        df.columns
    )

    for (
        violent_group,
        candidate_columns
    ) in BASE_PAPER_VIOLENT_CRIMES.items():

        for column in candidate_columns:

            exists = (
                column in existing_columns
            )

            rows.append({
                "violent_crime_group":
                    violent_group,

                "column":
                    column,

                "exists_in_dataset":
                    exists,
            })

    return pd.DataFrame(rows)


def get_existing_base_paper_violent_columns(
    df: pd.DataFrame
) -> dict[str, list[str]]:
    """
    Return only base-paper violent-crime columns that exist in
    the integrated dataset.
    """

    existing_columns = set(
        df.columns
    )

    verified = {}

    for (
        violent_group,
        candidate_columns
    ) in BASE_PAPER_VIOLENT_CRIMES.items():

        verified[
            violent_group
        ] = [
            column
            for column in candidate_columns
            if column in existing_columns
        ]

    return verified


# =====================================================================
# COLUMN-NAME SEARCH UTILITY
# =====================================================================

def search_dataset_columns(
    df: pd.DataFrame,
    keywords: list[str]
) -> pd.DataFrame:
    """
    Search actual dataset column names using keywords.

    Used to find alternative names for base-paper attributes.
    """

    rows = []

    for column in df.columns:

        column_upper = str(
            column
        ).upper()

        matched_keywords = [
            keyword
            for keyword in keywords
            if keyword.upper()
            in column_upper
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

                "missing_percentage":
                    round(
                        df[column].isna().mean() * 100,
                        2
                    ),
            })

    return pd.DataFrame(rows)


# =====================================================================
# GROUP SUMMARY
# =====================================================================

def create_group_summary(
    mapping: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a summary of columns in each structural/crime group.
    """

    summary = (
        mapping
        .groupby(
            [
                "structural_group",
                "crime_family"
            ],
            dropna=False
        )
        .agg(
            column_count=(
                "column",
                "count"
            ),

            aggregate_columns=(
                "is_aggregate",
                "sum"
            ),

            average_missing_percentage=(
                "missing_percentage",
                "mean"
            ),
        )
        .reset_index()
    )

    summary[
        "average_missing_percentage"
    ] = (
        summary[
            "average_missing_percentage"
        ]
        .round(2)
    )

    return summary.sort_values(
        [
            "structural_group",
            "crime_family"
        ]
    ).reset_index(
        drop=True
    )


# =====================================================================
# SAVE GROUPING REPORTS
# =====================================================================

def save_crime_grouping_reports(
    df: pd.DataFrame,
    output_dir: Optional[str | Path] = None
) -> dict[str, Path]:
    """
    Create and save crime grouping reports.
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

    mapping = (
        create_crime_column_mapping(
            df
        )
    )

    summary = (
        create_group_summary(
            mapping
        )
    )

    mapping_path = (
        output_dir /
        "crime_column_group_mapping.csv"
    )

    summary_path = (
        output_dir /
        "crime_group_summary.csv"
    )

    mapping.to_csv(
        mapping_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    return {
        "column_mapping":
            mapping_path,

        "group_summary":
            summary_path,
    }


# =====================================================================
# MAIN TEST
# =====================================================================

if __name__ == "__main__":

    # Import the existing data-loading module
    from data_loading import load_crime_data

    print("=" * 70)
    print(
        "CRIME GROUPING / COLUMN CLASSIFICATION TEST"
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
    # Create complete column mapping
    # ---------------------------------------------------------------

    mapping = (
        create_crime_column_mapping(
            df
        )
    )

    print(
        f"Total columns classified: "
        f"{len(mapping)}"
    )

    # ---------------------------------------------------------------
    # Structural groups
    # ---------------------------------------------------------------

    structural_groups = (
        get_columns_by_structural_group(
            mapping
        )
    )

    print(
        "\nSTRUCTURAL GROUPS"
    )

    print("-" * 70)

    for (
        group,
        columns
    ) in structural_groups.items():

        print(
            f"{group:<28} : "
            f"{len(columns)} columns"
        )

    # ---------------------------------------------------------------
    # Crime families
    # ---------------------------------------------------------------

    crime_families = (
        get_columns_by_crime_family(
            mapping
        )
    )

    print(
        "\nCRIME FAMILIES"
    )

    print("-" * 70)

    for (
        family,
        columns
    ) in crime_families.items():

        print(
            f"{family:<32} : "
            f"{len(columns)} columns"
        )

    # ---------------------------------------------------------------
    # Aggregate columns
    # ---------------------------------------------------------------

    (
        aggregate_columns,
        detailed_columns
    ) = separate_aggregate_columns(
        mapping
    )

    print(
        "\nAggregate / total columns detected: "
        f"{len(aggregate_columns)}"
    )

    print(
        "Non-aggregate columns: "
        f"{len(detailed_columns)}"
    )

    print(
        "\nFirst aggregate columns:"
    )

    for column in aggregate_columns[:30]:

        print(
            f"  - {column}"
        )

    # ---------------------------------------------------------------
    # Save grouping reports
    # ---------------------------------------------------------------

    report_paths = (
        save_crime_grouping_reports(
            df
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
    # Unclassified crime columns
    # ---------------------------------------------------------------

    unclassified_report = (
        create_unclassified_crime_report(
            df
        )
    )

    unclassified_path = (
        OUTPUT_TABLES_DIR /
        "unclassified_crime_columns.csv"
    )

    unclassified_report.to_csv(
        unclassified_path,
        index=False
    )

    print(
        "\nUnclassified crime columns: "
        f"{len(unclassified_report)}"
    )

    print(
        "\nUNCLASSIFIED CRIME COLUMNS"
    )

    print("-" * 70)

    for column in (
        unclassified_report[
            "column"
        ]
    ):

        print(
            f"- {column}"
        )

    print(
        f"\nUnclassified report saved to:\n"
        f"{unclassified_path}"
    )

    # ---------------------------------------------------------------
    # Base-paper violent crime verification
    # ---------------------------------------------------------------

    violent_mapping = (
        verify_base_paper_violent_columns(
            df
        )
    )

    violent_mapping_path = (
        OUTPUT_TABLES_DIR /
        "basepaper_violent_crime_mapping.csv"
    )

    violent_mapping.to_csv(
        violent_mapping_path,
        index=False
    )

    print(
        "\nBASE-PAPER VIOLENT CRIME MAPPING"
    )

    print("-" * 70)

    print(
        violent_mapping.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Verified base-paper violent columns
    # ---------------------------------------------------------------

    verified_violent_columns = (
        get_existing_base_paper_violent_columns(
            df
        )
    )

    print(
        "\nVERIFIED BASE-PAPER VIOLENT CRIME COLUMNS"
    )

    print("-" * 70)

    for (
        group,
        columns
    ) in verified_violent_columns.items():

        print(
            f"\n{group}:"
        )

        if columns:

            for column in columns:

                print(
                    f"  - {column}"
                )

        else:

            print(
                "  No exact matching column found."
            )

    print(
        f"\nMapping saved to:\n"
        f"{violent_mapping_path}"
    )

    # ---------------------------------------------------------------
    # Search for alternative names
    # ---------------------------------------------------------------

    search_keywords = [
        "PREP",
        "DACOITY",
        "ASSEMBLY",
        "CULPABLE",
    ]

    search_results = (
        search_dataset_columns(
            df,
            search_keywords
        )
    )

    search_path = (
        OUTPUT_TABLES_DIR /
        "basepaper_column_keyword_search.csv"
    )

    search_results.to_csv(
        search_path,
        index=False
    )

    print(
        "\nBASE-PAPER COLUMN KEYWORD SEARCH"
    )

    print("-" * 70)

    if search_results.empty:

        print(
            "No matching columns were found."
        )

    else:

        print(
            search_results.to_string(
                index=False
            )
        )

    print(
        f"\nKeyword search saved to:\n"
        f"{search_path}"
    )

    # ---------------------------------------------------------------
    # Final group summary
    # ---------------------------------------------------------------

    summary = (
        create_group_summary(
            mapping
        )
    )

    print(
        "\nGROUP SUMMARY"
    )

    print("-" * 70)

    print(
        summary.to_string(
            index=False
        )
    )