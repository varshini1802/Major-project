"""Reconcile grouped-formula missingness to stored grouped values by keys."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audits/final_corrections_v5"
KEYS = ["YEAR", "STATE", "UNIT_NAME"]


def norm(frame):
    d = frame.copy()
    d.YEAR = pd.to_numeric(d.YEAR, errors="raise").astype(int)
    for c in KEYS[1:]:
        d[c] = d[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    if d.duplicated(KEYS).any():
        raise AssertionError("Duplicate district-year key prevents one-to-one missingness reconciliation")
    return d


def equal(a, b):
    a = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    b = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return (a == b) | (np.isnan(a) & np.isnan(b))


def main():
    raw = norm(pd.read_csv(ROOT / "data/processed/district_integrated_final.csv", low_memory=False))
    grouped = norm(pd.read_csv(ROOT / "data/processed/district_grouped_reduced.csv", low_memory=False))
    join = raw[KEYS].merge(grouped[KEYS], on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not join._merge.eq("both").all():
        raise AssertionError("Source and grouped datasets have different district-year keys")
    grouped = grouped.set_index(KEYS).loc[pd.MultiIndex.from_frame(raw[KEYS])].reset_index(drop=True)
    checks = []
    formulas = pd.read_csv(ROOT / "outputs/tables/group_predictor_formula_recovery.csv")
    for r in formulas.itertuples(index=False):
        sources = [c.strip() for c in r.source_columns.split("|")]
        values = raw[sources].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        expected = values.sum(axis=1, min_count=1)
        observed_n = values.notna().sum(axis=1)
        actual = pd.to_numeric(grouped[r.predictor_group], errors="coerce")
        eq = equal(expected, actual)
        missing_target_assigned = int(grouped.loc[grouped.TARGET_VIOLENT_LEVEL.isna(), r.predictor_group].notna().sum())
        checks.append({"group": r.predictor_group, "source_components": len(sources),
                       "all_components_observed_rows": int(observed_n.eq(len(sources)).sum()),
                       "some_components_missing_rows": int((observed_n.gt(0) & observed_n.lt(len(sources))).sum()),
                       "all_components_missing_rows": int(observed_n.eq(0).sum()),
                       "partial_sum_rows": int((observed_n.gt(0) & observed_n.lt(len(sources))).sum()),
                       "expected_observed_zero_totals": int(expected.eq(0).sum()),
                       "expected_positive_totals": int(expected.gt(0).sum()),
                       "stored_missing_rows": int(actual.isna().sum()),
                       "stored_observed_zero_rows": int(actual.eq(0).sum()),
                       "stored_positive_rows": int(actual.gt(0).sum()),
                       "source_formula_value_mismatches": int((~eq).sum()),
                       "missing_target_rows_with_group_value": missing_target_assigned,
                       "result": "PASS" if not (~eq).any() and missing_target_assigned == 0 else "FAIL",
                       "interpretation": "Partial sums include observed components only and are not complete totals; missingness cause is not established."})
    report = pd.DataFrame(checks)
    report.to_csv(OUT / "group_missingness_source_to_stored_reconciliation.csv", index=False)
    print(report.to_string(index=False))
    if not report.result.eq("PASS").all():
        raise AssertionError("Grouped missingness reconciliation failed")


if __name__ == "__main__":
    main()
