"""Fail-closed structural audit of corrected-v3 forecasting inputs."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "ML_Implementation"
BASE = ML / "corrected_temporal_results/corrected_v3"
sys.path.insert(0, str(ML))
from forecast_feature_contract import (APPROVED, KEYS, TARGET, build_forecast_matrix,
                                      validate_lineage_report, validate_matrix)
from sequence_generation import build_sequences

TRAIN = list(range(2001, 2011)); VAL = [2011, 2012]; TEST = [2013, 2014]


def read_norm(path):
    d = pd.read_csv(path)
    d["YEAR"] = pd.to_numeric(d.YEAR, errors="raise").astype(int)
    for c in KEYS[1:]:
        d[c] = d[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    if d.duplicated(KEYS).any():
        raise AssertionError(f"Duplicate keys in {path}")
    return d


def main():
    target = read_norm(ROOT / "data/processed/district_targeted.csv")
    cuts = pd.read_csv(ROOT / "outputs/tables/target_thresholds.csv").sort_values("boundary_index")
    assert cuts.upper_cut_point.astype(float).tolist() == [107., 237., 362., 563.]
    assert cuts.fit_years.astype(str).unique().tolist() == ["2001-2010"]
    reports = [{"check": "frozen_thresholds", "status": "PASS", "rows": len(target), "details": "107,237,362,563; fitted 2001-2010"}]
    lineage = pd.read_csv(BASE / "forecast_source_lineage.csv")
    validate_lineage_report(lineage)
    reports.append({"check": "approved_feature_source_lineage", "status": "PASS",
                    "rows": int(lineage.experiment.str.startswith("corrected_tabular_").sum()),
                    "details": "all 17 approved tabular predictor rows have original sources, formulas, observation time, overlap status, forecast decision, and evidence"})
    source_paths = {"baseline": ROOT / "data/processed/district_grouped_reduced.csv",
                    "enhanced": ROOT / "data/processed/district_enhanced_features.csv"}
    strict = {}
    forecast_matrices = {}
    for variant, path in source_paths.items():
        source = read_norm(path)
        features = APPROVED[variant]
        saved = read_norm(BASE / "inputs" / f"{variant}_forecast_matrix.csv")
        forecast_matrices[variant] = saved
        cols = [f"LAG1_{f}" for f in features]
        validate_matrix(saved, cols, variant)
        expected, expected_cols = build_forecast_matrix(target, source, features, variant)
        assert cols == expected_cols
        try:
            pd.testing.assert_frame_equal(saved[KEYS + ["SOURCE_YEAR", TARGET] + cols].reset_index(drop=True),
                                          expected[KEYS + ["SOURCE_YEAR", TARGET] + cols].reset_index(drop=True),
                                          check_dtype=False, check_exact=True)
        except AssertionError as exc:
            raise AssertionError(f"{variant}: cached model matrix differs from keyed source reconstruction") from exc
        history = source[KEYS + features].copy(); history["YEAR"] += 1
        history = history.rename(columns={c: f"LAG1_{c}" for c in features})
        check = saved[KEYS + cols].merge(history, on=KEYS, how="left", suffixes=("_saved", "_src"), validate="one_to_one")
        for feature in cols:
            a = pd.to_numeric(check[f"{feature}_saved"], errors="coerce")
            b = pd.to_numeric(check[f"{feature}_src"], errors="coerce")
            assert ((a.eq(b)) | (a.isna() & b.isna())).all(), f"{variant}: lag source mismatch {feature}"
        reports.append({"check": f"{variant}_explicit_feature_matrix", "status": "PASS", "rows": len(saved),
                        "details": f"exact approved columns; keyed source year=target-1; no target/burden column in X; source reconstructs saved matrix"})
        X, meta, diag = build_sequences(target, source, features, 3, TRAIN + VAL + TEST)
        artifact = np.load(BASE / "inputs" / f"{variant}_common_sequences.npz")
        common_meta = pd.read_csv(BASE / "inputs/common_sequence_metadata.csv")
        saved_meta = read_norm(BASE / "inputs/common_sequence_metadata.csv")
        common_X = artifact["X"]
        idx = pd.MultiIndex.from_frame(saved_meta[KEYS])
        generated_idx = pd.MultiIndex.from_frame(meta[KEYS])
        generated_positions = pd.DataFrame({"_generated_position": np.arange(len(meta))})
        for key in KEYS:
            generated_positions[key] = meta[key].to_numpy()
        matched = saved_meta[KEYS].merge(generated_positions, on=KEYS, how="left",
                                         validate="one_to_one", indicator=True)
        if not matched._merge.eq("both").all():
            raise AssertionError(f"{variant}: common sequence key not found in regenerated source sequences")
        positions = matched._generated_position.to_numpy(dtype=int)
        regenerated_meta = meta.iloc[positions].reset_index(drop=True)
        regenerated_X = X[positions]
        if not np.array_equal(regenerated_X, common_X):
            raise AssertionError(f"{variant}: common cached arrays differ from current source reconstruction")
        if not np.array_equal(regenerated_meta[KEYS + [TARGET, "INPUT_YEARS"]].astype(str).to_numpy(),
                              saved_meta[KEYS + [TARGET, "INPUT_YEARS"]].astype(str).to_numpy()):
            raise AssertionError(f"{variant}: common cached metadata differs from source reconstruction")
        if not np.array_equal(artifact["y"], common_meta[TARGET].to_numpy(dtype=int)):
            raise AssertionError(f"{variant}: sequence y array and metadata labels differ")
        if not np.array_equal(artifact["years"], common_meta.YEAR.to_numpy(dtype=int)):
            raise AssertionError(f"{variant}: sequence years and metadata years differ")
        if artifact["feature_names"].tolist() != features or TARGET in artifact["feature_names"].tolist():
            raise AssertionError(f"{variant}: sequence feature list differs from contract or contains target")
        if not np.isfinite(common_X).all():
            raise AssertionError(f"{variant}: sequence array contains nonfinite values")
        reports.append({"check": f"{variant}_common_sequence_integrity", "status": "PASS", "rows": len(common_meta),
                        "details": f"strict source reconstruction agrees; {diag['valid_sequences']} valid variant histories; common cohort arrays/metadata/labels aligned; no target in X"})
        strict[variant] = common_meta
    for key_column in KEYS + [TARGET, "SOURCE_YEAR"]:
        if not forecast_matrices["baseline"][key_column].astype(str).reset_index(drop=True).equals(
                forecast_matrices["enhanced"][key_column].astype(str).reset_index(drop=True)):
            raise AssertionError(f"Tabular baseline/enhanced feature-row alignment differs for {key_column}")
    reports.append({"check": "tabular_full_row_key_and_target_alignment", "status": "PASS",
                    "rows": len(forecast_matrices["baseline"]),
                    "details": "baseline/enhanced matrices share identical target district-year keys, source years, and canonical labels"})
    base = strict["baseline"]; enhanced = strict["enhanced"]
    if not base[KEYS + [TARGET, "INPUT_YEARS"]].equals(enhanced[KEYS + [TARGET, "INPUT_YEARS"]]):
        raise AssertionError("Common baseline/enhanced sequences do not share exact keys, years, and labels")
    ntest = int(base.YEAR.isin(TEST).sum())
    reports.append({"check": "common_cohort_target_keys", "status": "PASS", "rows": len(base),
                    "details": f"baseline/enhanced target keys and labels identical; train={int(base.YEAR.isin(TRAIN).sum())}, validation={int(base.YEAR.isin(VAL).sum())}, test={ntest}"})

    tab_results = pd.read_csv(BASE / "tabular/test_results.csv")
    for variant in ("baseline", "enhanced"):
        expected_cols = "|".join(f"LAG1_{c}" for c in APPROVED[variant])
        rows = tab_results[tab_results.variant.eq(variant)]
        if len(rows) != 2 or not rows.feature_columns.eq(expected_cols).all():
            raise AssertionError(f"{variant}: model-run feature list differs from audited contract")
    tab_keys = []
    for variant in ("baseline", "enhanced"):
        preds = pd.read_csv(BASE / "tabular" / variant / "mlp" / "test_predictions.csv")
        actual = target[KEYS + [TARGET]].merge(preds[KEYS + [TARGET]], on=KEYS,
                                                suffixes=("_canonical", "_pred"), validate="one_to_one")
        if not actual[f"{TARGET}_canonical"].eq(actual[f"{TARGET}_pred"]).all():
            raise AssertionError(f"{variant}: test prediction labels do not match canonical targets")
        tab_keys.append(preds[KEYS].sort_values(KEYS).reset_index(drop=True))
    if not tab_keys[0].equals(tab_keys[1]):
        raise AssertionError("Tabular baseline/enhanced test district-year keys differ")
    reports.append({"check": "trained_tabular_feature_list_and_test_keys", "status": "PASS",
                    "rows": len(tab_keys[0]), "details": "saved result feature strings match approved contract; test keys and labels align with canonical target"})

    seq_results = pd.read_csv(BASE / "common_sequence_models/test_results.csv")
    seq_test_keys = []
    for variant in ("baseline", "enhanced"):
        rows = seq_results[seq_results.variant.eq(variant)]
        if len(rows) != 2 or rows.test_rows.nunique() != 1 or int(rows.test_rows.iloc[0]) != ntest:
            raise AssertionError(f"{variant}: common sequence test size is inconsistent")
        selected_name = rows.loc[~rows.model.eq("Most frequent baseline"), "model"].iloc[0]
        pred_path = BASE / "common_sequence_models" / variant / selected_name.lower() / "test_predictions.csv"
        preds = pd.read_csv(pred_path)
        if not preds[TARGET].eq(pd.to_numeric(preds[TARGET], errors="coerce")).all():
            raise AssertionError(f"{variant}: invalid sequence test labels")
        seq_test_keys.append(preds[KEYS].sort_values(KEYS).reset_index(drop=True))
    if not seq_test_keys[0].equals(seq_test_keys[1]) or not seq_test_keys[0].equals(
            base[base.YEAR.isin(TEST)][KEYS].sort_values(KEYS).reset_index(drop=True)):
        raise AssertionError("Trained sequence predictions do not use exact common test keys")
    reports.append({"check": "trained_common_sequence_test_keys", "status": "PASS",
                    "rows": ntest, "details": "both selected models' test predictions use the same canonical district-year keys"})
    pd.DataFrame(reports).to_csv(BASE / "input_audit_report.csv", index=False)
    print(pd.DataFrame(reports).to_string(index=False))


if __name__ == "__main__":
    main()
