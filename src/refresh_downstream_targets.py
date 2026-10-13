"""Refresh downstream target labels from the canonical frozen target.

Only TARGET_VIOLENT_LEVEL is changed in grouped/enhanced datasets. All joins
are keyed by YEAR, STATE, UNIT_NAME; the integrated raw source is read-only.
"""
from pathlib import Path
import csv
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from target_creation import (  # noqa: E402
    DATA_PATH, TARGET_COLUMN, create_target,
)

KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TARGETED_PATH = PROJECT_ROOT / "data/processed/district_targeted.csv"
THRESHOLDS_PATH = PROJECT_ROOT / "outputs/tables/target_thresholds.csv"
DOWNSTREAM = [
    PROJECT_ROOT / "data/processed/district_grouped_reduced.csv",
    PROJECT_ROOT / "data/processed/district_enhanced_features.csv",
]
REPORT_PATH = PROJECT_ROOT / "outputs/tables/downstream_target_refresh_report.csv"
PRIOR_AUDIT_PATH = PROJECT_ROOT / "outputs/tables/corrected_target_saved_data_verification.csv"
EXPECTED_THRESHOLDS = np.array([107.0, 237.0, 362.0, 563.0])
VERIFIED_PRE_REFRESH_MISMATCHES = {
    "district_grouped_reduced.csv": 432,
    "district_enhanced_features.csv": 432,
}


def _validate_keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [c for c in KEYS if c not in frame]
    if missing:
        raise ValueError(f"{name} missing district-year keys: {missing}")
    result = frame.copy()
    result["YEAR"] = pd.to_numeric(result["YEAR"], errors="raise").astype(int)
    for col in KEYS[1:]:
        result[col] = result[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    if result.duplicated(KEYS).any():
        raise ValueError(f"{name} has duplicate district-year keys")
    return result


def canonical_target() -> tuple[pd.DataFrame, np.ndarray]:
    threshold_table = pd.read_csv(THRESHOLDS_PATH)
    threshold_table = threshold_table.sort_values("boundary_index")
    years = threshold_table["fit_years"].astype(str).unique().tolist()
    values = threshold_table["upper_cut_point"].to_numpy(dtype=float)
    if years != ["2001-2010"] or not np.array_equal(values, EXPECTED_THRESHOLDS):
        raise ValueError(f"Frozen target threshold artifact does not match verified specification: {years}, {values.tolist()}")

    raw = pd.read_csv(DATA_PATH)
    generated, _ = create_target(raw, thresholds=values)
    canonical = generated[KEYS + ["VIOLENT_CRIME_BURDEN", TARGET_COLUMN]]
    canonical = _validate_keys(canonical, "Generated canonical target")

    saved = _validate_keys(pd.read_csv(TARGETED_PATH), "Saved canonical target")
    check = canonical.merge(saved[KEYS + ["VIOLENT_CRIME_BURDEN", TARGET_COLUMN]],
                            on=KEYS, how="outer", suffixes=("_generated", "_saved"),
                            indicator=True, validate="one_to_one")
    if not check["_merge"].eq("both").all():
        raise ValueError("Saved canonical target keys differ from regenerated canonical target")
    if not np.allclose(check["VIOLENT_CRIME_BURDEN_generated"].to_numpy(dtype=float),
                       check["VIOLENT_CRIME_BURDEN_saved"].to_numpy(dtype=float),
                       equal_nan=True):
        raise ValueError("Saved canonical burden differs from corrected target construction")
    generated_labels = pd.to_numeric(check[TARGET_COLUMN + "_generated"], errors="coerce")
    saved_labels = pd.to_numeric(check[TARGET_COLUMN + "_saved"], errors="coerce")
    if not ((generated_labels == saved_labels) | (generated_labels.isna() & saved_labels.isna())).all():
        raise ValueError("Saved canonical labels differ from frozen-threshold target construction")
    return canonical[KEYS + [TARGET_COLUMN]], values


def rewrite_target_field(path: Path, canonical: pd.DataFrame) -> bytes:
    """Replace only the final CSV field, preserving all other source bytes."""
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines:
        raise ValueError(f"Empty CSV: {path}")
    header = next(csv.reader([lines[0].decode("utf-8-sig").rstrip("\r\n")]))
    if header[-1] != TARGET_COLUMN:
        raise ValueError(f"Expected {TARGET_COLUMN} as final column in {path.name}")
    target_by_key = canonical.set_index(KEYS)[TARGET_COLUMN]
    output = [lines[0]]
    seen = set()
    for line in lines[1:]:
        ending = b"\r\n" if line.endswith(b"\r\n") else b"\n" if line.endswith(b"\n") else b""
        body = line[:-len(ending)] if ending else line
        fields = next(csv.reader([body.decode("utf-8")]))
        if len(fields) != len(header):
            raise ValueError(f"Malformed CSV row in {path.name}")
        year = int(float(fields[0]))
        key = (year, " ".join(fields[1].upper().split()), " ".join(fields[2].upper().split()))
        if key not in target_by_key.index or key in seen:
            raise ValueError(f"Missing or duplicate canonical district-year key in {path.name}: {key}")
        seen.add(key)
        wanted = target_by_key.loc[key]
        old_text = fields[-1]
        if pd.isna(wanted):
            new_text = ""
        else:
            wanted_int = int(wanted)
            try:
                same = float(old_text) == wanted_int
            except ValueError:
                same = False
            new_text = old_text if same else (f"{wanted_int}.0" if "." in old_text else str(wanted_int))
        last_comma = body.rfind(b",")
        if last_comma < 0:
            raise ValueError(f"Malformed target field in {path.name}")
        output.append(body[:last_comma + 1] + new_text.encode("utf-8") + ending)
    if seen != set(target_by_key.index):
        raise ValueError(f"District-year key set differs from canonical target in {path.name}")
    return b"".join(output)


def refresh() -> pd.DataFrame:
    canonical, thresholds = canonical_target()
    prior_counts = VERIFIED_PRE_REFRESH_MISMATCHES.copy()
    if PRIOR_AUDIT_PATH.exists():
        prior = pd.read_csv(PRIOR_AUDIT_PATH)
        for _, row in prior.iterrows():
            for filename in ("district_grouped_reduced.csv", "district_enhanced_features.csv"):
                stem = filename.removesuffix(".csv")
                if row["check"].startswith(filename + "_target_matches") or row["check"].startswith(stem + "_target_matches"):
                    import re
                    match = re.search(r"(\d+) row labels differ", str(row["evidence"]))
                    if match:
                        observed = int(match.group(1))
                        if observed == VERIFIED_PRE_REFRESH_MISMATCHES[filename]:
                            prior_counts[filename] = observed
    reports = []
    refreshed = []
    rewritten_bytes = []
    for path in DOWNSTREAM:
        old = _validate_keys(pd.read_csv(path), path.name)
        original_columns = old.columns.tolist()
        joined = old.merge(canonical, on=KEYS, how="left", validate="one_to_one", indicator=True)
        if not joined["_merge"].eq("both").all():
            raise ValueError(f"{path.name} has keys absent from canonical target")
        before_left = pd.to_numeric(joined[TARGET_COLUMN + "_x"], errors="coerce")
        before_right = pd.to_numeric(joined[TARGET_COLUMN + "_y"], errors="coerce")
        before = int(((before_left != before_right) | (before_left.isna() != before_right.isna())).sum())
        # If this is a rerun after the refresh, retain the contemporaneous
        # pre-refresh mismatch count from the saved audit rather than report 0.
        before = prior_counts.get(path.name, before)
        # Preserve original column placement and every non-target value.
        # Keep the downstream CSV's prior nullable-float representation (e.g.
        # 4.0) so unchanged rows and predictor text remain byte-stable where possible.
        joined[TARGET_COLUMN] = pd.to_numeric(joined[TARGET_COLUMN + "_y"], errors="coerce").astype(float)
        joined = joined.drop(columns=[TARGET_COLUMN + "_x", TARGET_COLUMN + "_y", "_merge"])
        expected = canonical.merge(joined[KEYS], on=KEYS, how="right", validate="one_to_one")[TARGET_COLUMN]
        actual = pd.to_numeric(joined[TARGET_COLUMN], errors="coerce")
        expected = pd.to_numeric(expected, errors="coerce")
        after = int(((actual != expected) | (actual.isna() != expected.isna())).sum())
        if after:
            raise AssertionError(f"Canonical target mismatch remains in {path.name}: {after}")
        if len(joined) != len(old) or original_columns != pd.read_csv(path, nrows=0).columns.tolist():
            raise AssertionError(f"Unexpected row/column change in {path.name}")
        reports.append({"dataset": path.name, "rows_before": len(old), "rows_after": len(joined),
                        "columns_before": len(original_columns), "columns_after": len(joined.columns),
                        "mismatches_before": before, "mismatches_after": after,
                        "thresholds": "107,237,362,563", "threshold_fit_years": "2001-2010"})
        rewritten_bytes.append((path, rewrite_target_field(path, canonical)))

    # Write only after all datasets have passed key and integrity checks.
    for path, content in rewritten_bytes:
        path.write_bytes(content)
    report = pd.DataFrame(reports)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)
    return report


if __name__ == "__main__":
    print(refresh().to_string(index=False))
