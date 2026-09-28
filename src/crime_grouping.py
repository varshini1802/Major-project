"""
STEP 6 — Final Reduced Dataset Audit

Audits:
1. Final shape
2. Exact column structure
3. Duplicate rows
4. Duplicate metadata keys
5. Target distribution
6. Missing values
7. Grouped feature data types
8. Remaining unwanted columns
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "district_grouped_reduced.csv"
)


EXPECTED_COLUMNS = [
    "YEAR",
    "STATE",
    "UNIT_NAME",
    "GROUP_HOMICIDE",
    "GROUP_SEXUAL_VIOLENCE",
    "GROUP_KIDNAPPING_ABDUCTION",
    "GROUP_ROBBERY_DACOITY",
    "GROUP_PUBLIC_VIOLENCE",
    "GROUP_PROPERTY_CRIME",
    "TARGET_VIOLENT_LEVEL",
]

KEY_COLUMNS = [
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


def main():

    print("\n" + "=" * 80)
    print("STEP 6 — FINAL REDUCED DATASET AUDIT")
    print("=" * 80)

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    print("\nLoading final dataset...")
    df = pd.read_csv(FINAL_DATA)

    print("Dataset path:")
    print(FINAL_DATA)

    print("\nDataset shape:")
    print(df.shape)

    # ------------------------------------------------------------------
    # 1. SHAPE VALIDATION
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("1. SHAPE VALIDATION")
    print("=" * 80)

    expected_shape = (9856, 10)

    print("Expected shape :", expected_shape)
    print("Actual shape   :", df.shape)

    shape_valid = df.shape == expected_shape

    print("Shape valid    :", shape_valid)

    if not shape_valid:
        raise ValueError(
            f"Unexpected dataset shape: {df.shape}. "
            f"Expected {expected_shape}."
        )

    # ------------------------------------------------------------------
    # 2. COLUMN VALIDATION
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("2. COLUMN VALIDATION")
    print("=" * 80)

    print("\nExpected columns:")
    for column in EXPECTED_COLUMNS:
        print(" ", column)

    print("\nActual columns:")
    for column in df.columns:
        print(" ", column)

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    extra_columns = [
        column for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    print("\nMissing expected columns:", missing_columns)
    print("Unexpected extra columns :", extra_columns)

    columns_valid = (
        list(df.columns) == EXPECTED_COLUMNS
    )

    print("Column structure valid   :", columns_valid)

    if not columns_valid:
        raise ValueError(
            "Final dataset columns do not exactly match the expected structure."
        )

    # ------------------------------------------------------------------
    # 3. DUPLICATE ROW AUDIT
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("3. DUPLICATE ROW AUDIT")
    print("=" * 80)

    duplicate_rows = df.duplicated().sum()

    print("Duplicate rows:", duplicate_rows)

    if duplicate_rows == 0:
        print("Duplicate row audit: PASSED")
    else:
        print("Duplicate row audit: WARNING")

    # ------------------------------------------------------------------
    # 4. DUPLICATE KEY AUDIT
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("4. METADATA KEY AUDIT")
    print("=" * 80)

    duplicate_keys = df.duplicated(
        subset=KEY_COLUMNS
    ).sum()

    print("Key columns:")
    print(KEY_COLUMNS)

    print("\nDuplicate metadata keys:", duplicate_keys)

    if duplicate_keys != 0:
        raise ValueError(
            "Duplicate YEAR + STATE + UNIT_NAME keys detected."
        )

    print("Metadata key audit: PASSED")

    # ------------------------------------------------------------------
    # 5. TARGET DISTRIBUTION
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("5. TARGET DISTRIBUTION")
    print("=" * 80)

    target_distribution = (
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
    )

    print(target_distribution)

    print("\nTarget missing values:")
    print(df[TARGET_COLUMN].isna().sum())

    # ------------------------------------------------------------------
    # 6. GROUP FEATURE MISSING VALUES
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("6. GROUP FEATURE MISSING VALUES")
    print("=" * 80)

    group_missing = df[GROUP_COLUMNS].isna().sum()

    print(group_missing)

    print("\nRows where ALL six group features are missing:")

    all_groups_missing = df[GROUP_COLUMNS].isna().all(axis=1).sum()

    print(all_groups_missing)

    # ------------------------------------------------------------------
    # 7. DATA TYPES
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("7. DATA TYPE AUDIT")
    print("=" * 80)

    print(df.dtypes)

    # ------------------------------------------------------------------
    # 8. GROUP FEATURE SUMMARY
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("8. GROUP FEATURE SUMMARY")
    print("=" * 80)

    print(
        df[GROUP_COLUMNS].describe()
    )

    # ------------------------------------------------------------------
    # 9. UNWANTED COLUMN CHECK
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("9. UNWANTED COLUMN CHECK")
    print("=" * 80)

    unwanted_patterns = [
        "IPC_",
        "SC_",
        "ST_",
        "WOMEN_",
        "CHILDREN_",
    ]

    unwanted_columns = []

    for column in df.columns:

        for pattern in unwanted_patterns:

            if column.startswith(pattern):

                unwanted_columns.append(column)
                break

    print("Remaining detailed crime columns:")
    print(unwanted_columns)

    if len(unwanted_columns) == 0:
        print("\nUnwanted detailed crime columns: NONE")
    else:
        raise ValueError(
            "Detailed crime columns still remain in final dataset."
        )

    # ------------------------------------------------------------------
    # 10. FINAL VALIDATION
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("10. FINAL VALIDATION")
    print("=" * 80)

    all_groups_missing_valid = (
        all_groups_missing == 225
    )

    target_missing_valid = (
        df[TARGET_COLUMN].isna().sum() == 225
    )

    final_validation = (
        shape_valid
        and columns_valid
        and duplicate_keys == 0
        and all_groups_missing_valid
        and target_missing_valid
        and len(unwanted_columns) == 0
    )

    print("Shape validation              :", shape_valid)
    print("Column validation             :", columns_valid)
    print("Duplicate key validation      :", duplicate_keys == 0)
    print("Target missing = 225          :", target_missing_valid)
    print("All groups missing rows = 225 :", all_groups_missing_valid)
    print("No detailed crime columns     :", len(unwanted_columns) == 0)

    print("\nFINAL AUDIT RESULT:", final_validation)

    if not final_validation:
        raise ValueError(
            "FINAL DATASET AUDIT FAILED."
        )

    print("\n" + "=" * 80)
    print("STEP 6 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\nFinal dataset is ready for the modeling pipeline.")
    print("Shape:", df.shape)
    print("Columns:", len(df.columns))
    print("File:", FINAL_DATA)

    print("=" * 80)


if __name__ == "__main__":
    main()