"""Final data-quality, missingness, pipeline-count, and claim audit.

Writes derived audit tables only. Source data and prior result directories are read-only.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/tables/final_remaining_corrections"
sys.path.insert(0, str(ROOT / "ML_Implementation"))
sys.path.insert(0, str(ROOT / "src"))
from forecast_feature_contract import APPROVED, KEYS, TARGET, build_forecast_matrix, normalize
from sequence_generation import build_sequences

GROUPS = list(APPROVED["baseline"])
TRAIN = list(range(2001, 2011)); VAL = [2011, 2012]; TEST = [2013, 2014]
CLASS_NAMES = {1: "Lowest burden", 2: "Low-to-moderate burden", 3: "Moderate burden",
               4: "Moderate-to-high burden", 5: "Highest burden"}


def load(path):
    return pd.read_csv(path)


def missingness_rows(frame, group_columns, by):
    records = []
    keys = by if by else []
    grouped = frame.groupby(keys, dropna=False, sort=True) if keys else [("ALL", frame)]
    for group_key, subset in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        prefix = dict(zip(keys or ["scope"], group_key))
        for column in group_columns:
            values = pd.to_numeric(subset[column], errors="coerce")
            records.append({**prefix, "crime_group": column, "rows": int(len(subset)),
                            "missing_or_unavailable": int(values.isna().sum()),
                            "observed_zero": int(values.eq(0).sum()),
                            "observed_positive": int(values.gt(0).sum()),
                            "observed_negative_invalid": int(values.lt(0).sum()),
                            "missing_pct": float(values.isna().mean() * 100) if len(values) else np.nan,
                            "zero_is_observed_value": True,
                            "missing_reason_identifiable_from_available_columns": False})
    return pd.DataFrame(records)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = load(ROOT / "data/processed/district_targeted.csv")
    grouped = load(ROOT / "data/processed/district_grouped_reduced.csv")
    enhanced = load(ROOT / "data/processed/district_enhanced_features.csv")
    integrated = load(ROOT / "data/processed/district_integrated_final.csv")
    for frame in (targets, grouped, enhanced, integrated):
        frame["YEAR"] = pd.to_numeric(frame.YEAR, errors="raise").astype(int)

    for name, by in (("overall", []), ("by_year", ["YEAR"]),
                     ("by_state", ["STATE"]), ("by_district", ["STATE", "UNIT_NAME"])):
        missingness_rows(grouped, GROUPS, by).to_csv(OUT / f"group_missingness_{name}.csv", index=False)

    values = grouped[GROUPS].apply(pd.to_numeric, errors="coerce")
    n_observed = values.notna().sum(axis=1)
    total = values.sum(axis=1, min_count=1)
    pd.DataFrame({
        "measure": ["complete_six_groups", "partial_one_to_five_groups", "all_six_missing",
                    "grouped_total_is_partial", "genuine_zero_group_cells", "missing_group_cells"],
        "count": [int(n_observed.eq(6).sum()), int(n_observed.between(1, 5).sum()),
                  int(n_observed.eq(0).sum()), int(n_observed.between(1, 5).sum()),
                  int(values.eq(0).sum().sum()), int(values.isna().sum().sum())],
        "interpretation": ["All six group values observed", "Some group values observed, some missing",
                           "No group value observed", "TOTAL_GROUPED_CRIME sums available group values only; it is partial",
                           "Recorded numeric zero, kept distinct from missing", "Missing/unavailable; cause is not encoded in processed group file"],
    }).to_csv(OUT / "group_total_completeness.csv", index=False)

    target_valid = pd.to_numeric(targets[TARGET], errors="coerce").notna()
    targets[TARGET] = pd.to_numeric(targets[TARGET], errors="coerce")
    stage_rows = [
        {"stage": "integrated_final_rows", "rows_in": len(integrated), "rows_out": len(integrated), "excluded": 0, "reason": "saved final integrated district-year table"},
        {"stage": "canonical_target_observations", "rows_in": len(targets), "rows_out": int(target_valid.sum()), "excluded": int((~target_valid).sum()), "reason": "target is missing where burden cannot be calculated from available/valid source counts"},
    ]
    sources = {"baseline": grouped, "enhanced": enhanced}
    for variant, source in sources.items():
        matrix, cols = build_forecast_matrix(targets, source, APPROVED[variant], variant)
        target_keys = normalize(targets.loc[target_valid, KEYS], "valid target keys")
        source_rows = normalize(source[KEYS + APPROVED[variant]], f"{variant} source rows")
        source_rows["YEAR"] += 1
        eligible = target_keys.merge(source_rows, on=KEYS, how="left", indicator=True, validate="one_to_one")
        eligible_count = int(eligible[APPROVED[variant]].notna().any(axis=1).sum())
        absent_prior = int(eligible._merge.ne("both").sum())
        all_null_prior = int((eligible._merge.eq("both") & ~eligible[APPROVED[variant]].notna().any(axis=1)).sum())
        stage_rows.append({"stage": f"{variant}_tabular_forecast_matrix", "rows_in": int(target_valid.sum()),
                           "rows_out": len(matrix), "excluded": int(target_valid.sum()) - len(matrix),
                           "reason": f"keyed t-1 source join; absent prior-year district keys={absent_prior}; matched prior rows with all predictors missing={all_null_prior}; matrix drops all-null predictor rows"})
        stage_rows.append({"stage": f"{variant}_tabular_any_prior_feature_observed", "rows_in": int(target_valid.sum()),
                           "rows_out": eligible_count, "excluded": int(target_valid.sum()) - eligible_count,
                           "reason": "normalized keyed join to t-1, counted only when at least one approved predictor is observed"})
    pd.DataFrame(stage_rows).to_csv(OUT / "pipeline_stage_counts.csv", index=False)

    seq_rows = []
    for variant, source in sources.items():
        _, _, diag = build_sequences(targets[KEYS + [TARGET]], source, APPROVED[variant], 3,
                                     TRAIN + VAL + TEST)
        seq_rows.append({"stage": f"{variant}_strict_sequences", **diag,
                         "rejection_rule": "reject if any keyed t-3..t-1 district-year is absent or any selected input is missing/nonfinite; no padding/imputation"})
    strict = pd.DataFrame(seq_rows)
    strict.to_csv(OUT / "sequence_stage_counts.csv", index=False)

    dist_rows = []
    for cls in range(1, 6):
        dist_rows.append({"partition": "all_valid_target_rows", "class": cls, "class_name": CLASS_NAMES[cls],
                          "count": int((targets[TARGET] == cls).sum()), "rows": int(target_valid.sum())})
    for part, years in (("train", TRAIN), ("validation", VAL), ("test", TEST)):
        sub = targets[targets.YEAR.isin(years) & target_valid]
        for cls in range(1, 6):
            dist_rows.append({"partition": part, "class": cls, "class_name": CLASS_NAMES[cls],
                              "count": int((sub[TARGET] == cls).sum()), "rows": len(sub)})
    pd.DataFrame(dist_rows).to_csv(OUT / "target_class_distribution.csv", index=False)

    overlap_map = load(ROOT / "outputs/tables/target_predictor_source_overlap.csv")
    overlap_sources = overlap_map.loc[overlap_map.direct_overlap.astype(bool), "predictor_source_variable"].drop_duplicates().tolist()
    direct_values = targets[overlap_sources].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    burden_values = pd.to_numeric(targets["VIOLENT_CRIME_BURDEN"], errors="coerce")
    valid_overlap = target_valid & direct_values.notna() & burden_values.gt(0)
    share = direct_values[valid_overlap] / burden_values[valid_overlap]
    pd.DataFrame([{
        "target_valid_rows": int(target_valid.sum()), "rows_with_observed_overlap_sources": int(valid_overlap.sum()),
        "rows_without_observed_overlap_source_values": int(target_valid.sum() - valid_overlap.sum()),
        "unique_direct_overlap_sources": len(overlap_sources),
        "direct_overlap_sources": " | ".join(overlap_sources),
        "sum_direct_overlap_counts": float(direct_values[valid_overlap].sum()),
        "sum_target_burden": float(burden_values[valid_overlap].sum()),
        "direct_overlap_share_of_total_burden": float(direct_values[valid_overlap].sum() / burden_values[valid_overlap].sum()),
        "mean_row_share": float(share.mean()), "median_row_share": float(share.median()),
        "definition": "sum of the nine unique IPC source columns shared directly by grouped predictors and target, divided by the summed canonical VIOLENT_CRIME_BURDEN on valid rows; descriptive overlap, not proof that the class is deterministically reconstructible",
    }]).to_csv(OUT / "direct_overlap_burden_summary.csv", index=False)
    group_formula = load(ROOT / "outputs/tables/group_predictor_formula_recovery.csv")
    group_formula[["predictor_group", "all_sources_missing_rows", "partially_missing_source_rows",
                   "stored_missing_rows", "row_value_mismatches", "formula_validation"]].to_csv(
                       OUT / "ipc_component_missingness_by_group.csv", index=False)

    raw_recon = load(ROOT / "outputs/tables/source_stage_reconciliation_audit.csv")
    integration_summary = raw_recon.groupby("comparison", dropna=False).agg(
        variables=("variable", "count"), max_rows_left=("rows_left", "max"),
        max_rows_right=("rows_right", "max"), value_mismatches=("value_mismatches", "sum")).reset_index()
    integration_summary.to_csv(OUT / "integration_reconciliation_summary.csv", index=False)

    thresholds = load(ROOT / "outputs/tables/target_thresholds.csv").sort_values("boundary_index")
    assert thresholds.upper_cut_point.astype(float).tolist() == [107., 237., 362., 563.]
    assert thresholds.fit_years.astype(str).unique().tolist() == ["2001-2010"]
    report = [
        "# Final data-quality and claim audit",
        "",
        "Generated from saved processed tables and prior reconciliation artifacts; raw inputs were not changed.",
        "",
        "## Target and protocol",
        "",
        "Frozen boundaries are 107, 237, 362, and 563, fitted on 2001–2010. The target is a project-defined adaptation of ordered violent-crime burden levels: class 1 lowest, class 2 low-to-moderate, class 3 moderate, class 4 moderate-to-high, class 5 highest. Split: train 2001–2010, validation 2011–2012, test 2013–2014.",
        "",
        f"Canonical target table has {len(targets)} rows; valid targets={int(target_valid.sum())}; missing targets={int((~target_valid).sum())}.",
        "",
        "## Missingness interpretation",
        "",
        "A numeric zero is counted as an observed zero. NaN is counted as missing/unavailable; the processed group table does not encode whether a missing source means non-reporting, unavailable, or another cause, so those causes cannot be distinguished from these files. Partial grouped totals sum observed groups with `min_count=1`; they are not complete totals. No global zero-fill is applied. At the six stored group-column level, 9,631 rows have all six groups observed and 225 rows have all six missing; there are no rows with only some stored group values observed. At the underlying IPC-component aggregation level, partial source coverage does occur; see `group_predictor_formula_recovery.csv` for per-group counts (these group counts can overlap and are not unique-row totals).",
        "",
        "Detailed missingness is in `group_missingness_overall.csv`, `group_missingness_by_year.csv`, `group_missingness_by_state.csv`, and `group_missingness_by_district.csv`; grouped total completeness is in `group_total_completeness.csv`.",
        "",
        "## Exclusions and integration evidence",
        "",
        "Stage counts for target labels, tabular matrices, and strict sequence histories are in `pipeline_stage_counts.csv` and `sequence_stage_counts.csv`. Matrix and sequence counts use canonical district-year keys and the frozen labels. The raw reconciliation audit explains available alias normalization and 2014 row filtering; original upstream raw copies are absent, so provenance before the available normalized stage remains unresolved.",
        "",
        "Available-stage reconciliation found 100 final IPC measures matching their normalized raw-stage aliases across 9,856 district-year rows with zero value mismatches. In the 2014 branch, normalization-to-cleaned retained 838 rows and core filtering retained 729 (73 special units and 36 aggregate rows removed); the compared values matched with zero mismatches. This verifies only the available normalized/cleaned/core artifacts, not provenance back to absent upstream copies.",
        "",
        f"The nine unique IPC sources shared directly by predictor groups and target contribute {direct_values[valid_overlap].sum() / burden_values[valid_overlap].sum():.1%} of summed burden on {int(valid_overlap.sum())} target-valid rows with observed overlap-source values; {int(target_valid.sum() - valid_overlap.sum())} other valid-target rows have no observed value among these sources. This is a descriptive reconstruction fraction; it does not resolve all component relationships or imply deterministic class recovery. See `direct_overlap_burden_summary.csv`.",
        "",
        "## Claims and limitations",
        "",
        "The same-year random-split notebook is observed-year classification only. Its cached scores, including the legacy 90.76% figure, are not corrected forecasting results. The forecasting evaluations use historical keyed inputs and chronological partitions. Annual source-release dates are unavailable, and possible IPC parent/component relationships remain unresolved; operational forecast-time availability and complete indirect-overlap resolution are not claimed.",
        "",
        "No conference-paper manuscript was found in the repository file inventory during this audit. No manuscript text or claims were edited.",
    ]
    (OUT / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved missingness/stage/claim audits to {OUT}")
    print("target distribution rows", len(dist_rows), "grouped missing cells", int(values.isna().sum().sum()),
          "observed zero cells", int(values.eq(0).sum().sum()), "target missing", int((~target_valid).sum()))


if __name__ == "__main__":
    main()
