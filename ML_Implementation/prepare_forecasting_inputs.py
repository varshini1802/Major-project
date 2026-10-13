"""Rebuild explicit forecasting matrices and a strict shared sequence cohort."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "ML_Implementation"
sys.path.insert(0, str(ML))
sys.path.insert(0, str(ROOT / "src"))
from forecast_feature_contract import (APPROVED, ENGINEERED, GROUPS, KEYS, TARGET,
                                       build_forecast_matrix, validate_lineage_report,
                                       validate_matrix)
from sequence_generation import build_sequences
from sequence_cohort import common_sequence_cohort
from target_creation import BASE_PAPER_GROUPS

PREV = ML / "corrected_temporal_results/corrected_v2"
OUT = ML / "corrected_temporal_results/corrected_v3"
TRAIN = list(range(2001, 2011)); VAL = [2011, 2012]; TEST = [2013, 2014]


def load(path):
    return pd.read_csv(path)


def lineage_report():
    formulas = pd.read_csv(ROOT / "outputs/tables/group_predictor_formula_recovery.csv").set_index("predictor_group")
    overlap = pd.read_csv(ROOT / "outputs/tables/target_predictor_source_overlap.csv")
    group_meta = {}
    for group in GROUPS:
        row = formulas.loc[group]
        rows = overlap[overlap.predictor_group.eq(group)]
        related = sorted({str(v) for v in rows.documented_related_target_variables.dropna()
                          if str(v).strip() and str(v).lower() != "nan"})
        interpretation = " | ".join(rows.interpretation.astype(str).unique())
        group_meta[group] = {
            "source_columns": row.source_columns,
            "formula": f"{row.aggregation}; {row.missing_handling}",
            "direct_overlap": bool(rows.direct_overlap.astype(bool).any()),
            "indirect_overlap": "potential relationship noted; not resolved as component/parent" if related or "Possible parent" in interpretation else "no indirect mapping established in available audit",
            "related_target_variables": " | ".join(related),
        }
    group_sources = " | ".join(group_meta[g]["source_columns"] for g in GROUPS)
    feature_definitions = {**group_meta,
        "TOTAL_GROUPED_CRIME": {"source_columns": group_sources, "formula": "sum of six GROUP_* values, min_count=1; all missing => NaN; partial values summed", "direct_overlap": False, "indirect_overlap": "inherits component-group target overlaps", "related_target_variables": "inherited from GROUP_* components"},
        "ACTIVE_CRIME_GROUPS": {"source_columns": group_sources, "formula": "count of six GROUP_* values > 0; all six missing => NaN", "direct_overlap": False, "indirect_overlap": "inherits component-group target overlaps", "related_target_variables": "inherited from GROUP_* components"},
        "CRIME_DIVERSITY_INDEX": {"source_columns": group_sources, "formula": "Shannon entropy; partial missing values treated as zero; all missing => NaN; zero total => 0", "direct_overlap": False, "indirect_overlap": "inherits component-group target overlaps", "related_target_variables": "inherited from GROUP_* components"},
        "PREV_YEAR_TOTAL_CRIME": {"source_columns": group_sources, "formula": "exact STATE+UNIT_NAME+YEAR-1 match of TOTAL_GROUPED_CRIME; absent prior row => NaN", "direct_overlap": False, "indirect_overlap": "inherits component-group target overlaps; underlying observation is one year earlier", "related_target_variables": "inherited from GROUP_* components"},
        "YOY_TOTAL_CRIME_CHANGE": {"source_columns": group_sources, "formula": "TOTAL_GROUPED_CRIME - PREV_YEAR_TOTAL_CRIME; absolute difference, not percentage", "direct_overlap": False, "indirect_overlap": "inherits component-group target overlaps", "related_target_variables": "inherited from GROUP_* components"},
    }
    def source_years(feature, experiment):
        sequence = "sequence" in experiment or experiment == "notebook_lstm_sequence_code"
        if sequence:
            return {"PREV_YEAR_TOTAL_CRIME": "t-4,t-3,t-2 underlying grouped totals",
                    "YOY_TOTAL_CRIME_CHANGE": "grouped totals at t-3,t-2,t-1 and prior totals at t-4,t-3,t-2"}.get(
                        feature, "t-3,t-2,t-1")
        return {"PREV_YEAR_TOTAL_CRIME": "t-2 underlying grouped total",
                "YOY_TOTAL_CRIME_CHANGE": "grouped totals at t-1 and t-2"}.get(feature, "t-1")
    rows = []
    for experiment, names, availability, usage in (
        ("corrected_tabular_baseline", GROUPS, "feature source year s=t-1", "RETAINED; source IPC overlap is historical, not same-year"),
        ("corrected_tabular_enhanced", GROUPS + ENGINEERED, "feature row s=t-1; PREV_YEAR reaches t-2", "RETAINED; historical inputs only"),
        ("corrected_sequence_baseline", GROUPS, "history rows t-3,t-2,t-1", "RETAINED; all three rows required"),
        ("corrected_sequence_enhanced", GROUPS + ENGINEERED, "history rows t-3,t-2,t-1; PREV_YEAR reaches one additional year back", "RETAINED; all three rows required"),
        ("retired_temporal_leakage_free_runner", GROUPS + ENGINEERED, "legacy keyed lag source row s=t-1", "DISABLED; replaced by explicit corrected-v3 matrix contract; prior outputs preserved"),
        ("retired_combined_temporal_sequence_runner", GROUPS + ENGINEERED, "legacy histories intended as t-3,t-2,t-1", "DISABLED; recomputed noncanonical 2004-2010 thresholds; prior outputs preserved"),
        ("retired_temporal_deep_learning_sequence_runner", GROUPS + ENGINEERED, "legacy histories intended as t-3,t-2,t-1", "DISABLED; refit thresholds and padded missing years; prior outputs preserved"),
    ):
        for name in names:
            meta = feature_definitions[name]
            rows.append({"experiment": experiment, "predictor": name,
                         "original_source_columns": meta["source_columns"],
                         "aggregation_or_transformation": meta["formula"],
                         "source_observation_year": source_years(name, experiment),
                         "target_year": "t", "direct_target_source_overlap": meta["direct_overlap"],
                         "indirect_overlap_status": meta["indirect_overlap"],
                         "related_target_sources": meta["related_target_variables"],
                         "available_by_forecast_origin": "year-eligible if prior-year data released before origin; publication timing unverified",
                         "decision": usage,
                         "evidence": "src/feature_engineering.py; src/target_creation.py; outputs/tables/group_predictor_formula_recovery.csv; outputs/tables/target_predictor_source_overlap.csv; runner code and v3 keyed matrix/sequence checks"})
    # Same-year notebook predictors are retained only for explicitly observed-year classification.
    for name in GROUPS + ["PREV_YEAR_TOTAL_CRIME"]:
        meta = feature_definitions[name]
        rows.append({"experiment": "notebook_observed_year_classification", "predictor": name,
                     "original_source_columns": meta["source_columns"],
                     "aggregation_or_transformation": meta["formula"],
                     "source_observation_year": "target year t for GROUP_*; t-1 for PREV_YEAR_TOTAL_CRIME",
                     "target_year": "t", "direct_target_source_overlap": meta["direct_overlap"],
                     "indirect_overlap_status": meta["indirect_overlap"],
                     "related_target_sources": meta["related_target_variables"],
                     "available_by_forecast_origin": "No for same-year groups; release timing also unavailable",
                     "decision": "RESTRICTED TO OBSERVED-YEAR CLASSIFICATION; do not report as forecast input",
                     "evidence": "notebooks/observed_year_classification_models.ipynb Steps 14, 20, and 28 same-year feature lists; target source overlap report"})
    for name in GROUPS + ["PREV_YEAR_TOTAL_CRIME"]:
        meta = feature_definitions[name]
        rows.append({"experiment": "notebook_lstm_sequence_code", "predictor": name,
                     "original_source_columns": meta["source_columns"],
                     "aggregation_or_transformation": meta["formula"],
                     "source_observation_year": source_years(name, "notebook_lstm_sequence_code"),
                     "target_year": "t", "direct_target_source_overlap": meta["direct_overlap"],
                     "indirect_overlap_status": meta["indirect_overlap"],
                     "related_target_sources": meta["related_target_variables"],
                     "available_by_forecast_origin": "year-eligible if historical annual records were released before origin; release timing unverified",
                     "decision": "HISTORICAL ONLY; notebook builder corrected to exact keyed histories; cached notebook outputs remain legacy",
                     "evidence": "notebooks/observed_year_classification_models.ipynb LSTM_BASELINE_FEATURES/LSTM_ENHANCED_FEATURES; corrected create_lstm_sequences cell; common-cohort assertions"})
    candidate_years = {
        **{name: "t (same-year crime count)" for name in GROUPS},
        "TOTAL_GROUPED_CRIME": "t (aggregate of same-year group counts)",
        "ACTIVE_CRIME_GROUPS": "t (same-year group presence)",
        "CRIME_DIVERSITY_INDEX": "t (distribution of same-year group counts)",
        "PREV_YEAR_TOTAL_CRIME": "t-1 underlying total; stored on target-year row, not passed in that form",
        "YOY_TOTAL_CRIME_CHANGE": "t and t-1 (change includes target-year total)",
    }
    for name in GROUPS + ENGINEERED:
        meta = feature_definitions[name]
        rows.append({"experiment": "same_year_feature_candidates_excluded", "predictor": name,
                     "original_source_columns": meta["source_columns"],
                     "aggregation_or_transformation": meta["formula"],
                     "source_observation_year": candidate_years[name],
                     "target_year": "t", "direct_target_source_overlap": meta["direct_overlap"],
                     "indirect_overlap_status": meta["indirect_overlap"],
                     "related_target_sources": meta["related_target_variables"],
                     "available_by_forecast_origin": "No for t counts/aggregates; t-1 underlying PREV_YEAR feature is only used through approved prior-row lag",
                     "decision": "EXCLUDED in target-year-row form; use only approved keyed historical form where defined",
                     "evidence": "src/feature_engineering.py formulas; explicit v3 feature allowlist; t-1 source-key reconstruction"})
    return pd.DataFrame(rows)


def main():
    if OUT.exists() and any(path.is_file() for path in OUT.rglob("*")):
        raise FileExistsError(f"Refusing to overwrite corrected-v3 artifacts: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    data_out = OUT / "inputs"; data_out.mkdir(exist_ok=True)
    targets = load(ROOT / "data/processed/district_targeted.csv")
    thresholds = pd.read_csv(ROOT / "outputs/tables/target_thresholds.csv").sort_values("boundary_index")
    cuts = thresholds.upper_cut_point.astype(float).tolist()
    if cuts != [107., 237., 362., 563.] or thresholds.fit_years.astype(str).unique().tolist() != ["2001-2010"]:
        raise AssertionError("Frozen target threshold artifact changed")
    targets = targets[KEYS + [TARGET]]
    sources = {"baseline": load(ROOT / "data/processed/district_grouped_reduced.csv"),
               "enhanced": load(ROOT / "data/processed/district_enhanced_features.csv")}
    matrices = {}
    for variant in ("baseline", "enhanced"):
        matrix, cols = build_forecast_matrix(targets, sources[variant], APPROVED[variant], variant)
        validate_matrix(matrix, cols, variant)
        matrix.to_csv(data_out / f"{variant}_forecast_matrix.csv", index=False)
        matrices[variant] = (matrix, cols)

    # Independently regenerate strict sequences and require exact agreement with authoritative v2 artifacts.
    strict = {}
    target_years = TRAIN + VAL + TEST
    for variant in ("baseline", "enhanced"):
        old = np.load(PREV / "sequences" / f"{variant}_sequences.npz")
        old_meta = pd.read_csv(PREV / "sequences" / f"{variant}_sequence_metadata.csv")
        x, meta, diag = build_sequences(targets, sources[variant], APPROVED[variant], 3, target_years)
        if not np.array_equal(x, old["X"]):
            raise AssertionError(f"{variant}: strict source regeneration differs from corrected-v2 array")
        if not np.array_equal(meta[KEYS + [TARGET, "INPUT_YEARS"]].astype(str).to_numpy(),
                              old_meta[KEYS + [TARGET, "INPUT_YEARS"]].astype(str).to_numpy()):
            raise AssertionError(f"{variant}: regenerated metadata differs from corrected-v2")
        strict[variant] = (x, meta, diag)
    base_X, base_meta, enh_X, enh_meta = common_sequence_cohort(
        strict["baseline"][0], strict["baseline"][1], strict["enhanced"][0], strict["enhanced"][1])
    if not base_meta[KEYS + [TARGET, "INPUT_YEARS"]].equals(enh_meta[KEYS + [TARGET, "INPUT_YEARS"]]):
        raise AssertionError("Common cohort baseline/enhanced target keys or labels differ")
    cohort_meta = base_meta.copy()
    cohort_meta.to_csv(data_out / "common_sequence_metadata.csv", index=False)
    for variant, x in (("baseline", base_X), ("enhanced", enh_X)):
        np.savez_compressed(data_out / f"{variant}_common_sequences.npz", X=x,
                            y=cohort_meta[TARGET].to_numpy(dtype=np.int64),
                            years=cohort_meta.YEAR.to_numpy(dtype=np.int64),
                            states=cohort_meta.STATE.to_numpy(dtype=str),
                            units=cohort_meta.UNIT_NAME.to_numpy(dtype=str),
                            input_years=cohort_meta.INPUT_YEARS.to_numpy(dtype=str),
                            feature_names=np.asarray(APPROVED[variant], dtype=str))
    splits = []
    for part, years in (("train", TRAIN), ("validation", VAL), ("test", TEST)):
        subset = cohort_meta[cohort_meta.YEAR.isin(years)]
        for cls in range(1, 6):
            splits.append({"partition": part, "years": f"{min(years)}-{max(years)}", "class": cls,
                           "count": int((subset[TARGET] == cls).sum()), "rows": len(subset)})
    pd.DataFrame(splits).to_csv(data_out / "common_cohort_class_distributions.csv", index=False)
    pd.DataFrame([{"variant": name, **diag, "common_cohort_rows": len(cohort_meta),
                   "common_train_rows": int(cohort_meta.YEAR.isin(TRAIN).sum()),
                   "common_validation_rows": int(cohort_meta.YEAR.isin(VAL).sum()),
                   "common_test_rows": int(cohort_meta.YEAR.isin(TEST).sum()),
                   "legacy_invalid_count": 232 if name == "baseline" else 1551}
                  for name, (_, _, diag) in strict.items()]).to_csv(data_out / "input_integrity_summary.csv", index=False)
    lineage = lineage_report()
    validate_lineage_report(lineage)
    lineage.to_csv(OUT / "forecast_source_lineage.csv", index=False)
    config = {"objective": "forecast future-year violent-crime burden class",
              "thresholds": cuts, "threshold_fit_years": "2001-2010", "train_years": TRAIN,
              "validation_years": VAL, "test_years": TEST, "seed": 42,
              "tabular_features": {k: [f"LAG1_{c}" for c in v] for k, v in APPROVED.items()},
              "sequence_features": APPROVED, "sequence_lookback": 3,
              "common_cohort_rule": "intersection of valid baseline and enhanced sequence YEAR/STATE/UNIT_NAME keys; target labels must agree",
              "source_publication_timing": "unverified; prior observation-year eligibility only"}
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved corrected-v3 matrices, strict shared sequence cohort ({len(cohort_meta)}), lineage, and configuration to {OUT}")


if __name__ == "__main__":
    main()
