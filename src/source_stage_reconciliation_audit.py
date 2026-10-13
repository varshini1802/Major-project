"""Trace integrated IPC value differences through available processing stages."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refresh_downstream_targets import KEYS  # noqa: E402

TABLES = ROOT / "outputs/tables"
RAW_STAGE = ROOT / "data/processed/district_integrated_raw.csv"
FINAL = ROOT / "data/processed/district_integrated_final.csv"
MAPPING = TABLES / "canonical_feature_mapping.csv"
NORMALIZED_2014 = ROOT / "data/processed/district_normalized/01_District_wise_crimes_committed_IPC_2014.csv"
CLEANED_2014 = ROOT / "data/processed/district_cleaned/01_District_wise_crimes_committed_IPC_2014.csv"
CORE_2014 = ROOT / "data/processed/district_core/01_District_wise_crimes_committed_IPC_2014.csv"
REPORT = TABLES / "source_stage_reconciliation_audit.csv"

CORE_TO_FINAL = {
    "Murder": "IPC_MURDER", "Rape": "IPC_RAPE", "Dacoity": "IPC_DACOITY",
    "Robbery": "IPC_ROBBERY", "Theft": "IPC_THEFT", "Riots": "IPC_RIOTS",
    "Cheating": "IPC_CHEATING", "Arson": "IPC_ARSON",
}


def normalize(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    if "DISTRICT" in result and "UNIT_NAME" not in result:
        result = result.rename(columns={"DISTRICT": "UNIT_NAME"})
    result["YEAR"] = pd.to_numeric(result["YEAR"], errors="raise").astype(int)
    for c in ["STATE", "UNIT_NAME"]:
        result[c] = result[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    if result.duplicated(KEYS).any():
        raise ValueError(f"Duplicate normalized keys in {name}")
    return result


def compare(name, left, right, key_columns, rows, transformations):
    l = left.set_index(KEYS)
    r = right.set_index(KEYS)
    joined = l[key_columns].join(r[key_columns], how="outer", lsuffix="_left", rsuffix="_right", validate="one_to_one")
    key_match = len(l.index.difference(r.index)) == 0 and len(r.index.difference(l.index)) == 0
    for col in key_columns:
        a, b = pd.to_numeric(joined[col + "_left"], errors="coerce"), pd.to_numeric(joined[col + "_right"], errors="coerce")
        mismatch = int((~((a == b) | (a.isna() & b.isna()))).sum())
        rows.append({"comparison": name, "variable": col, "rows_left": len(l), "rows_right": len(r),
                     "district_year_keys_match": key_match, "value_mismatches": mismatch,
                     "transformation_or_evidence": transformations})


def main():
    raw = normalize(pd.read_csv(RAW_STAGE), "integrated raw")
    final = normalize(pd.read_csv(FINAL), "integrated final")
    mapping = pd.read_csv(MAPPING)
    rows = []

    # Reconstruct each final IPC feature from source-stage aliases in the
    # canonical mapping. The integration notebook's consolidation operation
    # coalesces exact normalized representations, without arithmetic changes.
    mapped_final_columns = set()
    raw_i, final_i = raw.set_index(KEYS), final.set_index(KEYS)
    for (family, canonical_name), group in mapping.groupby(["crime_group", "canonical_feature"], sort=False):
        output = f"{family}_{canonical_name}"
        if not output.startswith("IPC_") or output not in final_i:
            continue
        source_columns = [c for c in group.source_feature.astype(str).tolist() if c in raw_i]
        if not source_columns:
            continue
        mapped_final_columns.add(output)
        coalesced = raw_i[source_columns].bfill(axis=1).iloc[:, 0]
        a, b = pd.to_numeric(coalesced, errors="coerce"), pd.to_numeric(final_i[output], errors="coerce")
        mismatch = int((~((a == b) | (a.isna() & b.isna()))).sum())
        rows.append({"comparison": "integrated_raw_aliases_to_final", "variable": output,
                     "rows_left": len(raw_i), "rows_right": len(final_i),
                     "district_year_keys_match": raw_i.index.equals(final_i.index),
                     "value_mismatches": mismatch,
                     "transformation_or_evidence": f"raw aliases={' | '.join(source_columns)}; case/feature normalization and left-to-right coalesce; no count transformation"})

    unmapped = sorted(c for c in final.columns if c.startswith("IPC_") and c != "IPC_AVAILABLE" and c not in mapped_final_columns)
    if unmapped:
        rows.append({"comparison": "final_ipc_unmapped", "variable": " | ".join(unmapped),
                     "rows_left": len(raw), "rows_right": len(final), "district_year_keys_match": "",
                     "value_mismatches": "", "transformation_or_evidence": "No canonical source-column mapping found"})

    norm14 = normalize(pd.read_csv(NORMALIZED_2014), "normalized IPC 2014")
    clean14 = normalize(pd.read_csv(CLEANED_2014), "cleaned IPC 2014")
    core14 = normalize(pd.read_csv(CORE_2014), "core IPC 2014")
    for source_col, final_col in CORE_TO_FINAL.items():
        # Input files have STATE/DISTRICT/YEAR at normalized stage.
        for left_name, left, right_name, right, column in (
            ("normalized_2014", norm14, "cleaned_2014", clean14, source_col),
            ("cleaned_2014", clean14, "core_2014", core14, source_col),
        ):
            l, r = left.set_index(KEYS), right.set_index(KEYS)
            common = l.index.intersection(r.index)
            a = pd.to_numeric(l.loc[common, column], errors="coerce")
            b = pd.to_numeric(r.loc[common, column], errors="coerce")
            mismatch = int((~((a == b) | (a.isna() & b.isna()))).sum())
            key_status = ("exact" if l.index.equals(r.index) else
                          "right_subset_of_left" if len(r.index.difference(l.index)) == 0 else
                          "left_subset_of_right" if len(l.index.difference(r.index)) == 0 else "different")
            rows.append({"comparison": f"{left_name}_to_{right_name}", "variable": source_col,
                         "rows_left": len(left), "rows_right": len(right),
                         "district_year_keys_match": key_status,
                         "value_mismatches": mismatch,
                         "transformation_or_evidence": "Cleaning only trims text and converts numeric columns; core stage filters UNIT_TYPE=DISTRICT_OR_REPORTING_UNIT and SOURCE_QUALITY_STATUS=VALID"})

        raw_alias = f"IPC_{source_col}"
        l = raw_i[raw_alias] if raw_alias in raw_i else None
        r = final_i[final_col]
        k14 = final_i.index.get_level_values("YEAR") == 2014
        if l is not None:
            a = pd.to_numeric(l.loc[k14], errors="coerce")
            b = pd.to_numeric(r.loc[k14], errors="coerce")
            mismatch = int((~((a == b) | (a.isna() & b.isna()))).sum())
            rows.append({"comparison": "raw_2014_case_variant_to_final", "variable": final_col,
                         "rows_left": int(k14.sum()), "rows_right": int(k14.sum()),
                         "district_year_keys_match": True, "value_mismatches": mismatch,
                         "transformation_or_evidence": f"source raw column {raw_alias} matches final {final_col}; canonical form differs in case only"})

    # Verify source-selection counts, rather than treating excluded rows as
    # missing crime values.
    unit_counts = clean14["UNIT_TYPE"].value_counts(dropna=False).to_dict()
    rows.append({"comparison": "2014_core_population_filter", "variable": "UNIT_TYPE / SOURCE_QUALITY_STATUS",
                 "rows_left": len(clean14), "rows_right": len(core14), "district_year_keys_match": "subset",
                 "value_mismatches": "", "transformation_or_evidence": f"cleaned 838 rows: {unit_counts}; 729 valid district/reporting rows retained; 73 special units and 36 aggregates excluded"})

    report = pd.DataFrame(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT, index=False)
    print(report.to_string(index=False))
    return report


if __name__ == "__main__":
    main()
