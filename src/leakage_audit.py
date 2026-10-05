from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "district_enhanced_features.csv"
)


# ============================================================
# EXPECTED COLUMNS
# ============================================================

GROUP_COLUMNS = [
    "GROUP_HOMICIDE",
    "GROUP_SEXUAL_VIOLENCE",
    "GROUP_KIDNAPPING_ABDUCTION",
    "GROUP_ROBBERY_DACOITY",
    "GROUP_PUBLIC_VIOLENCE",
    "GROUP_PROPERTY_CRIME",
]

NEW_FEATURE_COLUMNS = [
    "TOTAL_GROUPED_CRIME",
    "ACTIVE_CRIME_GROUPS",
    "CRIME_DIVERSITY_INDEX",
    "PREV_YEAR_TOTAL_CRIME",
    "YOY_TOTAL_CRIME_CHANGE",
]

TARGET_COLUMN = "TARGET_VIOLENT_LEVEL"

KEY_COLUMNS = [
    "YEAR",
    "STATE",
    "UNIT_NAME",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LEAKAGE AUDIT")
print("=" * 80)

print("\nLoading:")
print(INPUT_DATA)

df = pd.read_csv(INPUT_DATA)

print("\nDataset shape:", df.shape)


# ============================================================
# 1. COLUMN AUDIT
# ============================================================

expected_columns = (
    KEY_COLUMNS
    + GROUP_COLUMNS
    + NEW_FEATURE_COLUMNS
    + [TARGET_COLUMN]
)

print("\n" + "-" * 80)
print("1. COLUMN AUDIT")
print("-" * 80)

print("Expected columns:", len(expected_columns))
print("Actual columns  :", len(df.columns))

missing_expected = [
    col for col in expected_columns
    if col not in df.columns
]

unexpected_columns = [
    col for col in df.columns
    if col not in expected_columns
]

print("Missing expected columns:", missing_expected)
print("Unexpected columns:", unexpected_columns)

assert not missing_expected, (
    f"Missing expected columns: {missing_expected}"
)

assert not unexpected_columns, (
    f"Unexpected columns found: {unexpected_columns}"
)

print("PASS: Column structure is correct.")


# ============================================================
# 2. DUPLICATE / KEY AUDIT
# ============================================================

print("\n" + "-" * 80)
print("2. DUPLICATE / KEY AUDIT")
print("-" * 80)

duplicate_rows = df.duplicated().sum()

duplicate_keys = df.duplicated(
    subset=["YEAR", "STATE", "UNIT_NAME"]
).sum()

print("Duplicate complete rows:", duplicate_rows)
print("Duplicate YEAR-STATE-UNIT_NAME keys:", duplicate_keys)

assert duplicate_rows == 0, "Duplicate rows detected."
assert duplicate_keys == 0, "Duplicate keys detected."

print("PASS: No duplicate rows or keys.")


# ============================================================
# 3. TARGET PRESERVATION AUDIT
# ============================================================

print("\n" + "-" * 80)
print("3. TARGET PRESERVATION AUDIT")
print("-" * 80)

target_counts = df[TARGET_COLUMN].value_counts(
    dropna=False
).sort_index()

print("\nTarget distribution:")
print(target_counts)

expected_target_counts = {
    1.0: 1927,
    2.0: 1926,
    3.0: 1926,
    4.0: 1926,
    5.0: 1926,
}

for target_value, expected_count in expected_target_counts.items():
    actual_count = (
        df[TARGET_COLUMN] == target_value
    ).sum()

    assert actual_count == expected_count, (
        f"Target class {target_value} changed: "
        f"expected {expected_count}, got {actual_count}"
    )

missing_target = df[TARGET_COLUMN].isna().sum()

assert missing_target == 225, (
    f"Expected 225 missing targets, got {missing_target}"
)

print("\nPASS: Target distribution is unchanged.")


# ============================================================
# 4. TARGET NOT USED AS FEATURE
# ============================================================

print("\n" + "-" * 80)
print("4. TARGET-AS-FEATURE AUDIT")
print("-" * 80)

model_feature_columns = KEY_COLUMNS + GROUP_COLUMNS + NEW_FEATURE_COLUMNS

assert TARGET_COLUMN not in model_feature_columns

print("Target column:")
print(" ", TARGET_COLUMN)

print("\nFeature columns:")
for col in model_feature_columns:
    print(" ", col)

print("\nPASS: TARGET_VIOLENT_LEVEL is not included as an engineered feature.")


# ============================================================
# 5. DIRECT TARGET COPY AUDIT
# ============================================================

print("\n" + "-" * 80)
print("5. DIRECT TARGET COPY AUDIT")
print("-" * 80)

target_numeric = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce"
)

direct_copy_features = []

for feature in NEW_FEATURE_COLUMNS:
    feature_numeric = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    common_mask = (
        target_numeric.notna()
        & feature_numeric.notna()
    )

    if common_mask.sum() == 0:
        continue

    identical = np.array_equal(
        feature_numeric[common_mask].to_numpy(),
        target_numeric[common_mask].to_numpy()
    )

    if identical:
        direct_copy_features.append(feature)

print("Features exactly identical to target:")
print(direct_copy_features)

assert not direct_copy_features, (
    f"Direct target-copy detected: {direct_copy_features}"
)

print("PASS: No engineered feature is an exact copy of the target.")


# ============================================================
# 6. PREVIOUS-YEAR TEMPORAL AUDIT
# ============================================================

print("\n" + "-" * 80)
print("6. PREVIOUS-YEAR TEMPORAL AUDIT")
print("-" * 80)

# Reconstruct the total independently from the six grouped
# crime categories.

reconstructed_total = df[GROUP_COLUMNS].sum(
    axis=1,
    min_count=1
)

# Verify TOTAL_GROUPED_CRIME against the reconstructed value.

total_match_mask = (
    reconstructed_total.notna()
    & df["TOTAL_GROUPED_CRIME"].notna()
)

total_matches = np.isclose(
    reconstructed_total[total_match_mask],
    df.loc[total_match_mask, "TOTAL_GROUPED_CRIME"]
)

print(
    "TOTAL_GROUPED_CRIME reconstruction mismatches:",
    (~total_matches).sum()
)

assert (~total_matches).sum() == 0

print("PASS: TOTAL_GROUPED_CRIME is correctly reconstructed.")


# ------------------------------------------------------------
# Build independent previous-year lookup
# ------------------------------------------------------------

lookup = df[
    [
        "STATE",
        "UNIT_NAME",
        "YEAR",
        "TOTAL_GROUPED_CRIME",
    ]
].copy()

lookup = lookup.rename(
    columns={
        "YEAR": "EXPECTED_PREVIOUS_YEAR",
        "TOTAL_GROUPED_CRIME": "EXPECTED_PREV_TOTAL",
    }
)

audit_df = df[
    [
        "YEAR",
        "STATE",
        "UNIT_NAME",
        "PREV_YEAR_TOTAL_CRIME",
    ]
].copy()

audit_df["EXPECTED_PREVIOUS_YEAR"] = (
    audit_df["YEAR"] - 1
)

audit_df = audit_df.merge(
    lookup,
    on=[
        "STATE",
        "UNIT_NAME",
        "EXPECTED_PREVIOUS_YEAR",
    ],
    how="left",
    validate="one_to_one",
)

# Check that stored previous-year value matches
# the actual YEAR-1 record.

previous_available = (
    audit_df["PREV_YEAR_TOTAL_CRIME"].notna()
)

expected_available = (
    audit_df["EXPECTED_PREV_TOTAL"].notna()
)

availability_match = (
    previous_available == expected_available
)

print(
    "\nPrevious-year availability mismatches:",
    (~availability_match).sum()
)

assert (~availability_match).sum() == 0


# Compare actual previous-year values.

comparison_mask = (
    previous_available
    & expected_available
)

previous_value_matches = np.isclose(
    audit_df.loc[
        comparison_mask,
        "PREV_YEAR_TOTAL_CRIME"
    ],
    audit_df.loc[
        comparison_mask,
        "EXPECTED_PREV_TOTAL"
    ]
)

print(
    "Previous-year value mismatches:",
    (~previous_value_matches).sum()
)

assert (~previous_value_matches).sum() == 0

print(
    "Records with previous-year information:",
    previous_available.sum()
)

print(
    "Records without previous-year information:",
    (~previous_available).sum()
)

print(
    "\nPASS: PREV_YEAR_TOTAL_CRIME uses the actual YEAR-1 "
    "record for the same STATE and UNIT_NAME."
)


# ============================================================
# 7. FUTURE INFORMATION AUDIT
# ============================================================

print("\n" + "-" * 80)
print("7. FUTURE INFORMATION AUDIT")
print("-" * 80)

# The previous-year feature must always use YEAR - 1.
# No future year is allowed.

future_year_mask = (
    audit_df["EXPECTED_PREVIOUS_YEAR"]
    >= audit_df["YEAR"]
)

print(
    "Rows attempting to use current/future year:",
    future_year_mask.sum()
)

assert future_year_mask.sum() == 0

print(
    "PASS: PREV_YEAR_TOTAL_CRIME does not use current "
    "or future-year records."
)


# ============================================================
# 8. YOY CHANGE AUDIT
# ============================================================

print("\n" + "-" * 80)
print("8. YOY CHANGE AUDIT")
print("-" * 80)

expected_yoy = (
    df["TOTAL_GROUPED_CRIME"]
    - df["PREV_YEAR_TOTAL_CRIME"]
)

yoy_mask = (
    expected_yoy.notna()
    & df["YOY_TOTAL_CRIME_CHANGE"].notna()
)

yoy_matches = np.isclose(
    expected_yoy[yoy_mask],
    df.loc[
        yoy_mask,
        "YOY_TOTAL_CRIME_CHANGE"
    ]
)

print(
    "YOY_TOTAL_CRIME_CHANGE mismatches:",
    (~yoy_matches).sum()
)

assert (~yoy_matches).sum() == 0

print(
    "PASS: YOY_TOTAL_CRIME_CHANGE equals "
    "current total minus actual previous-year total."
)


# ============================================================
# 9. CURRENT-YEAR TARGET LEAKAGE AUDIT
# ============================================================

print("\n" + "-" * 80)
print("9. CURRENT-YEAR TARGET LEAKAGE ASSESSMENT")
print("-" * 80)

print(
    "\nThe following features are derived from current-year "
    "crime counts:"
)

current_year_features = [
    "TOTAL_GROUPED_CRIME",
    "ACTIVE_CRIME_GROUPS",
    "CRIME_DIVERSITY_INDEX",
    "YOY_TOTAL_CRIME_CHANGE",
]

for feature in current_year_features:
    print(" ", feature)

print(
    "\nIMPORTANT:"
)

print(
    "The project target TARGET_VIOLENT_LEVEL was created "
    "from a current-year violent-crime burden."
)

print(
    "Therefore, these current-year crime-derived features "
    "have target-leakage risk when used to predict the "
    "existing target."
)

print(
    "\nPREV_YEAR_TOTAL_CRIME is temporally prior to the "
    "current-year target and is therefore the lower-risk "
    "temporal feature."
)

print(
    "\nSTATUS: DATASET STRUCTURE PASSES, BUT CURRENT-YEAR "
    "FEATURES MUST NOT BE DESCRIBED AS LEAKAGE-FREE FOR "
    "THE EXISTING TARGET."
)


# ============================================================
# 10. MISSING / INFINITE VALUE AUDIT
# ============================================================

print("\n" + "-" * 80)
print("10. NUMERIC VALUE AUDIT")
print("-" * 80)

numeric_columns = GROUP_COLUMNS + NEW_FEATURE_COLUMNS

infinite_counts = {}

for col in numeric_columns:
    infinite_counts[col] = np.isinf(
        pd.to_numeric(df[col], errors="coerce")
    ).sum()

total_infinite = sum(infinite_counts.values())

print("Total infinite numeric values:", total_infinite)

assert total_infinite == 0

print("PASS: No infinite numeric values.")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("LEAKAGE AUDIT COMPLETED")
print("=" * 80)

print("\nStructural checks : PASS")
print("Target preservation: PASS")
print("Direct target copy : PASS")
print("Previous-year match: PASS")
print("Future-year check  : PASS")
print("YOY calculation    : PASS")
print("Infinite values    : PASS")

print(
    "\nIMPORTANT RESEARCH NOTE:"
)

print(
    "Current-year crime-derived features have potential "
    "target leakage because the existing target itself is "
    "constructed from current-year violent-crime burden."
)

print(
    "\nNo model training was performed."
)

print("=" * 80)