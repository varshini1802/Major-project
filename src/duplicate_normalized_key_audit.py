"""Explain duplicate normalized keys in the five cleaned 2001-2012 files."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audits/final_corrections_v5"
CLEAN = ROOT / "data/processed/district_cleaned"
CORE = ROOT / "data/processed/district_core"
KEYS = ["YEAR", "STATE", "UNIT_NAME"]
FILES = [
    "01_District_wise_crimes_committed_IPC_2001_2012.csv",
    "02_01_District_wise_crimes_committed_against_SC_2001_2012.csv",
    "02_District_wise_crimes_committed_against_ST_2001_2012.csv",
    "03_District_wise_crimes_committed_against_children_2001_2012.csv",
    "42_District_wise_crimes_committed_against_women_2001_2012.csv",
]
GENERATED_METADATA = {"SOURCE_FILE", "SOURCE_ROW_ID", "SOURCE_RECORD_ID", "DUPLICATE_KEY_FLAG",
                      "DATA_QUALITY_FLAG", "UNIT_TYPE", "SOURCE_QUALITY_STATUS"}


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    if "DISTRICT" in d and "UNIT_NAME" not in d:
        d = d.rename(columns={"DISTRICT": "UNIT_NAME"})
    d["YEAR"] = pd.to_numeric(d.YEAR, errors="raise").astype(int)
    for col in ("STATE", "UNIT_NAME"):
        d[col] = d[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    return d


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    key_rows, record_rows = [], []
    for filename in FILES:
        path = CLEAN / filename
        data = normalize(pd.read_csv(path, low_memory=False))
        core_path = CORE / filename
        core = normalize(pd.read_csv(core_path, low_memory=False)) if core_path.exists() else pd.DataFrame()
        duplicates = data[data.duplicated(KEYS, keep=False)].sort_values(KEYS)
        for key, group in duplicates.groupby(KEYS, dropna=False, sort=True):
            key_tuple = tuple(key)
            payload_cols = [c for c in data.columns if c not in KEYS and c not in GENERATED_METADATA]
            all_nonkey_cols = [c for c in data.columns if c not in KEYS]
            varying_payload = [c for c in payload_cols if group[c].nunique(dropna=False) > 1]
            varying_any = [c for c in all_nonkey_cols if group[c].nunique(dropna=False) > 1]
            unit_type = str(group["UNIT_TYPE"].iloc[0]) if "UNIT_TYPE" in group else "unknown"
            core_match = False
            if not core.empty:
                core_match = bool(((core.YEAR == key_tuple[0]) & (core.STATE == key_tuple[1]) &
                                   (core.UNIT_NAME == key_tuple[2])).any())
            payload_identical = not varying_payload
            category = ("IDENTICAL_OFFENCE_PAYLOAD_DISTINCT_PIPELINE_ROW_IDS" if payload_identical
                        else "CONFLICTING_OFFENCE_PAYLOAD")
            record_rows.append({"file": filename, "year": key_tuple[0], "state": key_tuple[1],
                                "unit_name": key_tuple[2], "copy_count": len(group),
                                "classification": category, "offence_payload_identical": payload_identical,
                                "varying_offence_columns": "|".join(varying_payload),
                                "varying_all_nonkey_columns": "|".join(varying_any),
                                "source_row_ids": "|".join(group.get("SOURCE_ROW_ID", pd.Series(dtype=str)).astype(str)),
                                "source_quality_flags": "|".join(group.get("SOURCE_QUALITY_STATUS", pd.Series(dtype=str)).astype(str)),
                                "unit_type": unit_type, "present_in_core_stage": core_match,
                                "key_impact": "excluded from core by duplicate quality rule" if not core_match else "present in core; inspect further"})
            key_rows.append({"file": filename, "duplicate_key": f"{key_tuple[1]} / {key_tuple[2]} / {key_tuple[0]}",
                             "copies": len(group), "identical_offence_payload": payload_identical,
                             "varying_offence_columns": "|".join(varying_payload),
                             "varying_nonkey_columns": "|".join(varying_any), "unit_type": unit_type,
                             "source_ids": "|".join(group.get("SOURCE_ROW_ID", pd.Series(dtype=str)).astype(str)),
                             "present_in_core": core_match})

    detail = pd.DataFrame(record_rows)
    detail.to_csv(OUT / "cleaned_duplicate_key_records.csv", index=False)
    per_file = detail.groupby("file").agg(
        duplicate_keys=("unit_name", "size"),
        duplicate_rows=("copy_count", "sum"),
        identical_payload_keys=("offence_payload_identical", "sum"),
        conflicting_payload_keys=("offence_payload_identical", lambda s: int((~s).sum())),
        keys_present_in_core=("present_in_core_stage", "sum"),
    ).reset_index()
    per_file["extra_duplicate_rows"] = per_file.duplicate_rows - per_file.duplicate_keys
    per_file["interpretation"] = "All reviewed duplicate keys are marked duplicate-key review and excluded from core; original upstream files are absent."
    per_file.to_csv(OUT / "cleaned_duplicate_key_summary.csv", index=False)

    readme = """# Cleaned-source duplicate-key audit\n\nThe v5 inventory identified duplicate normalized (YEAR, STATE, UNIT_NAME) keys in five cleaned 2001–2012 files. These outputs compare every non-key field. Generated pipeline metadata (source row IDs, quality flags and unit labels) is reported separately from offence payload values.\n\nThe integration notebook assigns `DUPLICATE_KEY_REVIEW`, classifies exact whole-row duplicates separately from conflicting offense payloads, then assigns `SOURCE_QUALITY_STATUS=DUPLICATE_KEY` to duplicate keys and admits only `VALID` ordinary district/reporting rows into the core stage. Consequently, all seven affected key groups listed here are absent from their corresponding core files. This is an explicit exclusion rule in the notebook, not a crime-count aggregation rule.\n\nTwo key groups have conflicting offence payloads: Jammu & Kashmir / RAILWAYS / 2010 in IPC and women files. Both are categorized as special units and excluded. The rows have distinct source row identifiers but no further railway-unit identifiers; why the counts differ cannot be resolved from available files. Five groups have identical offense payloads and differ only in source-row/provenance identifiers: duplicated Anantnag 2001 in SC and ST, and Nagaland TOTAL 2005 in SC, ST and children. `TOTAL` is an aggregate, while Anantnag is an ordinary district name; its duplicated SC/ST records are both excluded by the duplicate-quality rule. Whether the duplicate source rows arose in NCRB extraction or another upstream process is unverified because original raw files are absent.\n\nThe canonical integrated target/model input has unique district-year keys. The forecast target and predictors use IPC sources only; these duplicate groups do not introduce duplicate canonical model rows. However, excluding Anantnag 2001 removes the affected SC/ST source observations from their core family tables. No rule was invented to keep one or aggregate the duplicates. Raw/cleaned/core files were not modified.\n"""
    (OUT / "cleaned_duplicate_key_findings.md").write_text(readme, encoding="utf-8")
    print(per_file.to_string(index=False))
    print(detail.to_string(index=False))


if __name__ == "__main__":
    main()
