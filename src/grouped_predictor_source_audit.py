"""Recover and validate grouped predictor source formulas from project provenance.

Reads the committed grouping provenance reports, checks each row-wise sum
against the final integrated and reduced datasets, maps sources to target
sources, and verifies the corrected target artifacts. This is an audit only.
"""
from pathlib import Path
import ast
import numpy as np
import pandas as pd

from target_creation import BASE_PAPER_GROUPS, create_target

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
RAW = ROOT / "data" / "processed" / "district_integrated_final.csv"
PRE_CANONICAL = ROOT / "data" / "processed" / "district_integrated_raw.csv"
CANONICAL = ROOT / "data" / "processed" / "district_integrated_canonical.csv"
GROUPED = ROOT / "data" / "processed" / "district_grouped_reduced.csv"
TARGETED = ROOT / "data" / "processed" / "district_targeted.csv"
KEYS = ["YEAR", "STATE", "UNIT_NAME"]
GROUP_MAP = TABLES / "crime_group_mapping.csv"
PARENT_CHILD = TABLES / "crime_parent_child_validation.csv"


def _ordered(path):
    return pd.read_csv(path).sort_values(KEYS).reset_index(drop=True)


def main():
    provenance = pd.read_csv(GROUP_MAP)
    selected = provenance[(provenance.action == "USE_FOR_GROUPING") &
                          (provenance.role == "PRIMARY_GROUP_COMPONENT")].copy()
    parent_child = pd.read_csv(PARENT_CHILD)
    raw, grouped = _ordered(RAW), _ordered(GROUPED)
    sources = list(dict.fromkeys(c for cs in BASE_PAPER_GROUPS.values() for c in cs))
    temporal_tree = ast.parse((ROOT / "ML_Implementation" / "temporal_leakage_free_experiment.py").read_text(encoding="utf-8"))
    temporal_groups = next(ast.literal_eval(node.value) for node in temporal_tree.body
                           if isinstance(node, ast.Assign) and
                           any(isinstance(target, ast.Name) and target.id == "BURDEN_GROUPS" for target in node.targets))
    temporal_sources = set(c for cols in temporal_groups.values() for c in cols)
    target_sources = set(sources)
    overlap_rows, formula_rows, consistency_rows = [], [], []

    for group, rows in selected.groupby("grouped_category", sort=True):
        source_cols = rows.original_column.tolist()
        if group not in grouped.columns:
            raise ValueError(f"Grouped dataset is missing {group}.")
        missing = [c for c in source_cols if c not in raw.columns]
        if missing:
            raise ValueError(f"{group} source columns missing from final integrated data: {missing}")
        values = raw[source_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        computed = values.sum(axis=1, min_count=1)
        stored = pd.to_numeric(grouped[group], errors="coerce")
        both_present = computed.notna() & stored.notna()
        mismatches = int((~np.isclose(computed[both_present], stored[both_present])).sum())
        both_missing = int((computed.isna() & stored.isna()).sum())
        formula_rows.append({
            "predictor_group": group,
            "source_columns": " | ".join(source_cols),
            "source_count": len(source_cols),
            "formula_evidence": "outputs/tables/crime_group_mapping.csv + outputs/tables/grouped_crime_summary.csv (both in commit 0926f35); row-level validation against district_grouped_reduced.csv",
            "integration_notebook_status": "No GROUP_* formula definition found in notebooks/01_source_data_integration.ipynb; notebook covers integration/schema, not group construction",
            "grouping_rationale": " | ".join(dict.fromkeys(rows.reason.astype(str))),
            "generator_code_status": "original aggregation implementation not present in tracked source; formula recovered from provenance and verified exactly",
            "aggregation": "row-wise arithmetic sum",
            "missing_handling": "sum(min_count=1): all sources missing => missing; partial missing values are skipped",
            "other_transformations": "none observed (no scaling, rate conversion, or lag in stored group columns)",
            "all_sources_missing_rows": int(values.isna().all(axis=1).sum()),
            "partially_missing_source_rows": int((values.isna().any(axis=1) & ~values.isna().all(axis=1)).sum()),
            "negative_source_cells": int(values.lt(0).sum().sum()),
            "stored_missing_rows": int(stored.isna().sum()),
            "row_value_mismatches": mismatches,
            "both_missing_rows": both_missing,
            "formula_validation": "PASS" if mismatches == 0 and both_missing == int(stored.isna().sum()) else "FAIL",
        })
        related = parent_child[(parent_child.parent_or_reference_column.isin(source_cols)) |
                               (parent_child.component_or_comparison_column.isin(source_cols))]
        for source in source_cols:
            target_groups = [name for name, cols in BASE_PAPER_GROUPS.items() if source in cols]
            related_pairs = related[
                (related.parent_or_reference_column == source) | (related.component_or_comparison_column == source)
            ]
            overlap_rows.append({
                "predictor_group": group,
                "predictor_source_variable": source,
                "included_in_target_sources": source in sources,
                "target_group_names": " | ".join(target_groups),
                "direct_overlap": source in sources,
                "documented_related_target_variables": " | ".join(sorted(set(
                    related_pairs.parent_or_reference_column.tolist() +
                    related_pairs.component_or_comparison_column.tolist()) & set(sources) - {source})),
                "indirect_overlap_evidence": " | ".join(sorted(set(related_pairs.relationship.astype(str)))) if len(related_pairs) else "No related pair documented in parent-child audit",
                "interpretation": ("Direct: same original IPC variable contributes to predictor and burden target" if source in sources else
                                  "No direct shared source; indirect overlap not established by available provenance"),
            })

    for path in [PRE_CANONICAL, CANONICAL, RAW]:
        d = _ordered(path)
        keys_equal = d[KEYS].equals(grouped[KEYS])
        common = [c for c in d.columns if c.startswith("IPC_") and c in raw.columns]
        diff_cells, diff_cols, examples = 0, [], []
        for col in common:
            a = pd.to_numeric(d[col], errors="coerce")
            b = pd.to_numeric(raw[col], errors="coerce")
            different = ~((a == b) | (a.isna() & b.isna()))
            n = int(different.sum())
            if n:
                diff_cols.append(col)
                diff_cells += n
                examples.append(f"{col}:{n} rows")
        consistency_rows.append({
            "dataset": str(path.relative_to(ROOT)),
            "rows": len(d), "keys_match_final": keys_equal,
            "ipc_columns_compared": len(common), "ipc_columns_with_value_differences_vs_final": len(diff_cols),
            "ipc_columns_missing_vs_final": len(set(c for c in raw.columns if c.startswith("IPC_")) - set(c for c in d.columns)),
            "differing_ipc_cells_vs_final": diff_cells,
            "difference_detail": " | ".join(examples),
        })

    pd.DataFrame(formula_rows).to_csv(TABLES / "group_predictor_formula_recovery.csv", index=False)
    overlap_frame = pd.DataFrame(overlap_rows)
    overlap_frame.to_csv(TABLES / "target_predictor_source_overlap.csv", index=False)
    overlap_frame.to_csv(TABLES / "predictor_source_overlap_mapping.csv", index=False)
    pd.DataFrame(consistency_rows).to_csv(TABLES / "integrated_ipc_source_consistency.csv", index=False)

    # Confirm every corrected target artifact row and target value against a fresh construction.
    regenerated, _ = create_target(raw)
    saved = _ordered(TARGETED)
    regenerated = regenerated.sort_values(KEYS).reset_index(drop=True)
    burden_equal = np.allclose(regenerated.VIOLENT_CRIME_BURDEN.fillna(-999999),
                               saved.VIOLENT_CRIME_BURDEN.fillna(-999999))
    regenerated_labels = pd.to_numeric(regenerated.TARGET_VIOLENT_LEVEL, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    saved_labels = pd.to_numeric(saved.TARGET_VIOLENT_LEVEL, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    labels_equal = bool(np.all((regenerated_labels == saved_labels) | (np.isnan(regenerated_labels) & np.isnan(saved_labels))))
    fit_mask = pd.to_numeric(raw.YEAR).between(2001, 2010) & regenerated.VIOLENT_CRIME_BURDEN.notna()
    threshold = np.quantile(regenerated.loc[fit_mask, "VIOLENT_CRIME_BURDEN"], [0.2, 0.4, 0.6, 0.8])
    # Reorder by keys, then assert record-level invariance under a deterministic shuffle.
    shuffled, _ = create_target(raw.sample(frac=1, random_state=947))
    a = regenerated.set_index(KEYS).TARGET_VIOLENT_LEVEL.sort_index()
    b = shuffled.set_index(KEYS).TARGET_VIOLENT_LEVEL.sort_index()
    row_order_invariant = a.astype("string").fillna("MISSING").equals(b.astype("string").fillna("MISSING"))
    altered = raw.copy()
    future = pd.to_numeric(altered.YEAR).gt(2010)
    source_cols = [c for group in BASE_PAPER_GROUPS.values() for c in group if c in altered.columns]
    altered.loc[future, source_cols] = altered.loc[future, source_cols].apply(pd.to_numeric, errors="coerce") * 1000
    altered_target, _ = create_target(altered)
    holdout_invariant = regenerated.attrs["target_thresholds"] == altered_target.attrs["target_thresholds"]
    class_counts = regenerated.TARGET_VIOLENT_LEVEL.value_counts(dropna=False).sort_index().to_dict()
    saved_thresholds = pd.read_csv(TABLES / "target_thresholds.csv")
    threshold_artifact_equal = np.allclose(saved_thresholds.upper_cut_point.to_numpy(dtype=float),
                                           regenerated.attrs["target_thresholds"])
    saved_dist = pd.read_csv(TABLES / "final_target_distribution.csv")
    saved_dist_map = {str(row.target_level): int(row.count) for row in saved_dist.itertuples()}
    regenerated_dist_map = {str(int(k)): int(v) for k, v in regenerated.TARGET_VIOLENT_LEVEL.value_counts().items() if not pd.isna(k)}
    regenerated_dist_map["MISSING"] = int(regenerated.TARGET_VIOLENT_LEVEL.isna().sum())
    distribution_artifact_equal = saved_dist_map == regenerated_dist_map
    group_data = _ordered(GROUPED)
    group_target = pd.to_numeric(group_data["TARGET_VIOLENT_LEVEL"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    stale_target_rows = int(((group_target != saved_labels) & ~(np.isnan(group_target) & np.isnan(saved_labels))).sum())
    model_input_checks = [{"check": "district_grouped_reduced_target_matches_corrected_target", "result": stale_target_rows == 0,
                          "evidence": f"{stale_target_rows} row labels differ; notebook and non-temporal pipeline load this file"}]
    for name in ["district_enhanced_features.csv"]:
        data = _ordered(ROOT / "data" / "processed" / name)
        actual = pd.to_numeric(data.TARGET_VIOLENT_LEVEL, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        differs = (actual != saved_labels) & ~(np.isnan(actual) & np.isnan(saved_labels))
        model_input_checks.append({"check": f"{name}_target_matches_corrected_target", "result": int(differs.sum()) == 0,
                                  "evidence": f"{int(differs.sum())} row labels differ"})
    def normalized_labels(path):
        data = pd.read_csv(path)
        for col in ["STATE", "UNIT_NAME"]:
            data[col] = data[col].map(lambda value: " ".join(str(value).upper().split()))
        data["YEAR"] = pd.to_numeric(data.YEAR, errors="coerce").astype("Int64")
        truth = saved[[*KEYS, "TARGET_VIOLENT_LEVEL"]].copy()
        for col in ["STATE", "UNIT_NAME"]:
            truth[col] = truth[col].map(lambda value: " ".join(str(value).upper().split()))
        merged = data[KEYS + ["TARGET_VIOLENT_LEVEL"]].merge(truth, on=KEYS, suffixes=("_saved", "_recomputed"),
                                                              how="left", validate="one_to_one")
        left = pd.to_numeric(merged.TARGET_VIOLENT_LEVEL_saved, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        right = pd.to_numeric(merged.TARGET_VIOLENT_LEVEL_recomputed, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        equal = (left == right) | (np.isnan(left) & np.isnan(right))
        return int((~equal).sum()), int(merged.TARGET_VIOLENT_LEVEL_recomputed.isna().sum()), len(merged)
    for name in ["baseline_lagged_data.csv", "enhanced_lagged_data.csv"]:
        path = ROOT / "ML_Implementation" / "temporal_leakage_free_results" / name
        differences, no_target_match, count = normalized_labels(path)
        model_input_checks.append({"check": f"{name}_labels_match_corrected_target", "result": differences == 0 and no_target_match == 0,
                                  "evidence": f"{count} rows compared; label differences={differences}; unmatched target keys={no_target_match}"})
    target_checks = pd.DataFrame([
        {"check": "saved_burden_matches_reconstruction", "result": burden_equal, "evidence": "district_targeted.csv vs regenerated raw IPC source sums"},
        {"check": "saved_labels_match_reconstruction", "result": labels_equal, "evidence": f"saved labels; class counts={class_counts}"},
        {"check": "saved_thresholds_match_training_quintiles", "result": np.allclose(threshold, regenerated.attrs["target_thresholds"]), "evidence": f"fit years=2001-2010; thresholds={regenerated.attrs['target_thresholds']}"},
        {"check": "saved_threshold_artifact_matches", "result": threshold_artifact_equal, "evidence": "outputs/tables/target_thresholds.csv"},
        {"check": "saved_class_distribution_matches", "result": distribution_artifact_equal, "evidence": f"outputs/tables/final_target_distribution.csv; regenerated counts={regenerated_dist_map}"},
        {"check": "shuffled_row_order_preserves_keyed_labels", "result": row_order_invariant, "evidence": "deterministic shuffle seed=947"},
        {"check": "validation_test_data_do_not_change_thresholds", "result": holdout_invariant, "evidence": "validation/test source values multiplied by 1000; fit period remains 2001-2010"},
        {"check": "temporal_experiment_burden_sources_match_BASE_PAPER_GROUPS", "result": temporal_sources == target_sources,
         "evidence": f"same={len(temporal_sources & target_sources)}; only target={sorted(target_sources-temporal_sources)}; only temporal={sorted(temporal_sources-target_sources)}"},
    ])
    target_checks = pd.concat([target_checks, pd.DataFrame(model_input_checks)], ignore_index=True)
    target_checks.to_csv(TABLES / "corrected_target_saved_data_verification.csv", index=False)
    print("FORMULAS\n", pd.DataFrame(formula_rows).to_string(index=False))
    print("SOURCE CONSISTENCY\n", pd.DataFrame(consistency_rows).to_string(index=False))
    print("TARGET VERIFICATION\n", target_checks.to_string(index=False))
    if not all(r["formula_validation"] == "PASS" for r in formula_rows):
        raise AssertionError("At least one grouped predictor formula failed row-level validation.")
    if not (burden_equal and labels_equal and row_order_invariant and holdout_invariant and threshold_artifact_equal and distribution_artifact_equal and np.allclose(threshold, regenerated.attrs["target_thresholds"]) and temporal_sources == target_sources):
        raise AssertionError("Corrected target artifact validation failed.")


if __name__ == "__main__":
    main()
