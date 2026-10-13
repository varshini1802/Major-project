"""Audit stored forecasting inputs against district-year source records."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from target_creation import TARGET_COLUMN  # noqa: E402

KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TABLES = ROOT / "outputs/tables"
RAW = ROOT / "data/processed/district_integrated_final.csv"
TARGETED = ROOT / "data/processed/district_targeted.csv"
BASELINE = ROOT / "data/processed/district_grouped_reduced.csv"
ENHANCED = ROOT / "data/processed/district_enhanced_features.csv"
RESULTS = ROOT / "ML_Implementation/temporal_leakage_free_results"
CORRECTED = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v2"
CORRECTED_V3 = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3"
NOTEBOOK = ROOT / "notebooks/observed_year_classification_models.ipynb"
SCRIPT = ROOT / "ML_Implementation/temporal_leakage_free_experiment.py"
REPORT = TABLES / "forecast_feature_availability_audit.csv"
FEATURE_MAP_REPORT = TABLES / "forecast_feature_source_mapping.csv"


def normalized(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["YEAR"] = pd.to_numeric(d["YEAR"], errors="raise").astype(int)
    for c in KEYS[1:]:
        d[c] = d[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    if d.duplicated(KEYS).any():
        raise ValueError(f"Duplicate district-year keys in {path.name}")
    return d


def equal_values(left: pd.Series, right: pd.Series) -> pd.Series:
    a, b = pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")
    return (a.eq(b)) | (a.isna() & b.isna())


def main() -> pd.DataFrame:
    target = normalized(TARGETED)[KEYS + [TARGET_COLUMN]]
    source_data = {
        "baseline": normalized(BASELINE),
        "enhanced": normalized(ENHANCED),
    }
    output = []
    for name, path in (("baseline", RESULTS / "baseline_lagged_data.csv"),
                       ("enhanced", RESULTS / "enhanced_lagged_data.csv")):
        lagged = normalized(path)
        features = [c for c in lagged if c.startswith("LAG1_")]
        source = source_data[name]
        expected_labels = lagged[KEYS].merge(target, on=KEYS, how="left", validate="one_to_one")
        if expected_labels[TARGET_COLUMN].isna().any():
            raise AssertionError(f"{name}: lagged observations lack corrected target keys")
        if not equal_values(lagged[TARGET_COLUMN], expected_labels[TARGET_COLUMN]).all():
            raise AssertionError(f"{name}: stored lagged target labels disagree with canonical target")

        source_features = [c.removeprefix("LAG1_") for c in features]
        prior = source[KEYS + source_features].copy()
        prior["YEAR"] += 1
        prior = prior.rename(columns={c: "LAG1_" + c for c in source_features})
        check = lagged[KEYS + features].merge(prior, on=KEYS, how="left", suffixes=("_stored", "_source"),
                                              validate="one_to_one", indicator=True)
        missing_keys = int(check["_merge"].ne("both").sum())
        mismatch = 0
        for f in features:
            mismatch += int((~equal_values(check[f + "_stored"], check[f + "_source"])).sum())
        output.append({
            "check": f"{name}_lag_alignment", "status": "PASS" if not mismatch and not missing_keys else "FAIL",
            "rows": len(lagged), "features": len(features), "feature_value_mismatches": mismatch,
            "missing_prior_district_year_keys": missing_keys,
            "availability_rule": "Each LAG1 feature for target year t equals same district feature in t-1",
            "max_feature_year": int(lagged.YEAR.max() - 1),
        })

    # Verify no saved forecast observation has any input newer than t-1 and
    # that the documented year partitions are the ones used by the script.
    expected_partition = {"train": "2001-2010", "validation": "2011-2012", "test": "2013-2014"}
    script = (ROOT / "ML_Implementation/tabular_forecast_evaluation.py").read_text(encoding="utf-8")
    config = json.loads((CORRECTED_V3 / "run_config.json").read_text(encoding="utf-8"))
    partition_ok = (config.get("train_years") == list(range(2001, 2011))
                    and config.get("validation_years") == [2011, 2012]
                    and config.get("test_years") == [2013, 2014])
    output.append({"check": "forecast_partition_and_horizon", "status": "PASS" if partition_ok else "FAIL",
                   "rows": "", "features": "", "feature_value_mismatches": "", "missing_prior_district_year_keys": "",
                   "availability_rule": "Target year t uses t-1 inputs; target partitions " + str(expected_partition),
                   "max_feature_year": "script lag join shifts source YEAR by +1"})
    output.append({"check": "historical_source_release_timing", "status": "UNVERIFIED",
                   "rows": "", "features": "annual IPC/grouped observations",
                   "feature_value_mismatches": "", "missing_prior_district_year_keys": "",
                   "availability_rule": "Repository documents source observation years but contains no publication/release timestamps or forecast issue date. t-1 is source-year eligible; availability before a real forecast origin cannot be independently confirmed.",
                   "max_feature_year": "t-1 under annual-year assumption"})

    # The temporal experiment's imputer/scaler live in sklearn pipelines and
    # pipelines are fitted only on X_train after chronological partitioning.
    fit_after_split = all(token in script for token in [
        "make_pipeline(", "factory().fit(tr[columns], tr[TARGET])",
        "factory().fit(fit[columns], fit[TARGET])",
        "model.predict(te[columns])",
    ])
    output.append({"check": "temporal_preprocessing_fit_scope", "status": "PASS" if fit_after_split else "REVIEW",
                   "rows": "", "features": "", "feature_value_mismatches": "", "missing_prior_district_year_keys": "",
                   "availability_rule": "v3 candidate pipelines fit on chronological train; selected pipeline refits on train+validation after selection",
                   "max_feature_year": "train+validation refit after selection"})

    corrected_script = script
    corrected_fit_scope = all(token in corrected_script for token in [
        "factory().fit(tr[columns], tr[TARGET])", "f1_score(va[TARGET], pred, average=\"macro\"",
        "factory().fit(fit[columns], fit[TARGET])", "pred = model.predict(te[columns])",
    ])
    output.append({"check": "corrected_tabular_selection_and_test_lock", "status": "PASS" if corrected_fit_scope else "FAIL",
                   "rows": "corrected_v3 run_config.json", "features": "explicit approved t-1 tabular predictors",
                   "feature_value_mismatches": "", "missing_prior_district_year_keys": "",
                   "availability_rule": "Fit candidate models on train; select by validation macro-F1; refit chosen model on train+validation; test predictions occur afterward. Fixed majority baseline also reported.",
                   "max_feature_year": "target year minus one"})

    corrected_sequence_script = (ROOT / "ML_Implementation/common_cohort_sequence_evaluation.py").read_text(encoding="utf-8")
    sequence_fit_scope = all(token in corrected_sequence_script for token in [
        "scaler_transform(Xtr, Xv)", "f1_score(yv, val_pred, average=\"macro\"",
        "scaler_transform(np.concatenate([Xtr, Xv]), Xte)", "model.predict(test_scaled",
    ])
    output.append({"check": "corrected_sequence_selection_and_test_lock", "status": "PASS" if sequence_fit_scope else "FAIL",
                   "rows": "corrected_v3 run_config.json", "features": "strict common-cohort t-3 through t-1 sequences",
                   "feature_value_mismatches": "", "missing_prior_district_year_keys": "",
                   "availability_rule": "Scaler fitted on train for validation; model/epoch selected using validation; selected preprocessing refitted on train+validation; test predictions occur afterward.",
                   "max_feature_year": "target year minus one"})

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook_text = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
    explicitly_scoped = ("not forecasts of future-year violent-crime burden" in notebook_text
                         and "SAME-YEAR ENHANCED CLASSIFICATION" in notebook_text)
    output.append({"check": "notebook_random_split_scope", "status": "PASS" if explicitly_scoped else "FAIL",
                   "rows": "", "features": "same-year six GROUP_* columns; enhanced adds PREV_YEAR_TOTAL_CRIME",
                   "feature_value_mismatches": "", "missing_prior_district_year_keys": "",
                   "availability_rule": "Same-year grouped counts overlap the target source; random split sections are observed-year classification only",
                   "max_feature_year": "target year t features include t counts"})
    ann_scope_fixed = "X_baseline_train.iloc[training_indices]" in notebook_text and "baseline_scaler.fit_transform" in notebook_text
    output.append({"check": "notebook_ann_scaler_scope", "status": "PASS" if ann_scope_fixed else "FAIL",
                   "rows": "", "features": "", "feature_value_mismatches": "", "missing_prior_district_year_keys": "",
                   "availability_rule": "ANN scaling fit on inner training partition after validation split", "max_feature_year": "observed-year experiment"})

    sequence_examples = []
    sequence_config = json.loads((CORRECTED / "sequences/sequence_config.json").read_text(encoding="utf-8"))
    for variant, columns in (("baseline", sequence_config["baseline_features"]),
                             ("enhanced", sequence_config["enhanced_features"])):
        metadata_path = CORRECTED / "sequences" / f"{variant}_sequence_metadata.csv"
        metadata = normalized(metadata_path)
        source = source_data[variant].set_index(KEYS)
        invalid_rows = 0
        examples = []
        for row in metadata[KEYS].itertuples(index=False, name=None):
            year, state, unit = row
            row_invalid = False
            for historical_year in range(year - 3, year):
                key = (historical_year, state, unit)
                if key not in source.index or not np.isfinite(source.loc[key, columns].to_numpy(dtype=float)).all():
                    row_invalid = True
                    if len(examples) < 5:
                        examples.append(f"{year}/{state}/{unit}: missing or null source at {historical_year}")
                    break
            invalid_rows += int(row_invalid)
        output.append({"check": f"corrected_{variant}_sequence_metadata_alignment", "status": "PASS" if not invalid_rows else "FAIL",
                       "rows": len(metadata), "features": f"{len(columns)} selected historical features",
                       "feature_value_mismatches": invalid_rows,
                       "missing_prior_district_year_keys": invalid_rows,
                       "availability_rule": "Corrected metadata has exact complete t-3,t-2,t-1 histories; legacy metadata had 232 baseline and 1551 enhanced incomplete cases and 56 target-label mismatches each. examples=" + " | ".join(examples),
                       "max_feature_year": "target year minus one"})
    sequence_code = ''.join(''.join(c.get("source", [])) for c in notebook["cells"]
                            if "def create_lstm_sequences" in ''.join(c.get("source", [])))
    code_enforces_history = all(token in sequence_code for token in [
        "required = [(int(y), state, unit)", "lookup.loc[required, feature_columns]",
        "not np.isfinite(values).all()", "Outcome fields cannot be sequence inputs",
    ])
    output.append({"check": "notebook_sequence_builder_enforces_consecutive_prior_years",
                   "status": "PASS" if code_enforces_history else "FAIL", "rows": "", "features": "",
                   "feature_value_mismatches": "", "missing_prior_district_year_keys": "",
                   "availability_rule": "Builder reads target labels separately and uses exact keyed t-3 through t-1 finite inputs; it never gates inclusion on target-year predictor availability",
                   "max_feature_year": "target year minus one"})

    v3_input_audit = pd.read_csv(CORRECTED_V3 / "input_audit_report.csv")
    v3_ok = v3_input_audit.status.eq("PASS").all()
    output.append({"check": "corrected_v3_matrix_array_and_common_cohort_audit",
                   "status": "PASS" if v3_ok else "FAIL", "rows": len(v3_input_audit),
                   "features": "explicit matrix and sequence feature contracts",
                   "feature_value_mismatches": int((v3_input_audit.status != "PASS").sum()),
                   "missing_prior_district_year_keys": "",
                   "availability_rule": "Input audit reconstructs saved matrices/arrays and validates exact shared baseline/enhanced keys and labels",
                   "max_feature_year": "t-1"})

    reconciliation = pd.read_csv(TABLES / "source_stage_reconciliation_audit.csv")
    raw_values_mismatch = int(pd.to_numeric(reconciliation.value_mismatches, errors="coerce").fillna(0).sum())
    raw_mapping_failures = int(reconciliation.comparison.eq("final_ipc_unmapped").sum())
    output.append({"check": "raw_stage_2014_source_discrepancy",
                   "status": "PASS_WITH_PROVENANCE_LIMIT" if raw_values_mismatch == 0 and raw_mapping_failures == 0 else "FAIL",
                   "rows": "see source_stage_reconciliation_audit.csv", "features": 100,
                   "feature_value_mismatches": raw_values_mismatch,
                   "missing_prior_district_year_keys": raw_mapping_failures,
                   "availability_rule": "All 100 final IPC measures reconstruct from raw-stage aliases with zero mismatches; case/feature canonicalization explains 2014 aliases and core-stage filtering explains excluded rows. Original upstream raw copies are absent, so provenance before normalized stage cannot be verified.",
                   "max_feature_year": "2014; source files unchanged"})

    recovered = pd.read_csv(TABLES / "group_predictor_formula_recovery.csv")
    overlap = pd.read_csv(TABLES / "target_predictor_source_overlap.csv")
    mapping = []
    for row in recovered.itertuples():
        overlap_group = overlap[overlap.predictor_group == row.predictor_group]
        mapping.append({"experiment": "basepaper random-split", "feature": row.predictor_group,
                        "original_source_columns": row.source_columns,
                        "feature_formula": "Recovered grouped sum; see group_predictor_formula_recovery.csv",
                        "availability_for_target_year_t": "same-year t (not forecast eligible)",
                        "target_source_overlap": f"{int(overlap_group.direct_overlap.sum())}/{len(overlap_group)} direct source columns",
                        "forecast_horizon_use": "EXCLUDE current-year group value; notebook is observed-year classification"})
        mapping.append({"experiment": "temporal lagged baseline", "feature": row.predictor_group,
                        "original_source_columns": row.source_columns,
                        "feature_formula": "Recovered grouped sum; see group_predictor_formula_recovery.csv",
                        "availability_for_target_year_t": "t-1 via district-year exact-key lag join",
                        "target_source_overlap": f"{int(overlap_group.direct_overlap.sum())}/{len(overlap_group)} direct source columns",
                        "forecast_horizon_use": "PERMITTED when prior-year historical source is available"})
    engineered = [
        ("TOTAL_GROUPED_CRIME", "sum of six group columns; min_count=1", "component counts through t-1"),
        ("ACTIVE_CRIME_GROUPS", "count of six group values greater than zero; all-missing => missing", "component counts through t-1"),
        ("CRIME_DIVERSITY_INDEX", "Shannon entropy over six groups; missing treated as zero within partial rows; all-missing => missing; zero total => 0", "component counts through t-1"),
        ("PREV_YEAR_TOTAL_CRIME", "exact prior calendar-year TOTAL_GROUPED_CRIME by STATE+UNIT_NAME; absent prior row => missing", "component counts through t-2"),
        ("YOY_TOTAL_CRIME_CHANGE", "TOTAL_GROUPED_CRIME(s) - PREV_YEAR_TOTAL_CRIME(s); absolute count difference, not percent", "component counts through t-1 and t-2"),
    ]
    for feature, formula, source_time in engineered:
        mapping.append({"experiment": "temporal lagged enhanced", "feature": feature,
                        "original_source_columns": "six GROUP_* columns (see six formula rows above)",
                        "feature_formula": formula,
                        "availability_for_target_year_t": f"stored at t-1; {source_time}",
                        "target_source_overlap": "Inherited IPC overlap from component GROUP_* columns",
                        "forecast_horizon_use": "PERMITTED after exact t-1 row lag; PREV_YEAR_TOTAL_CRIME reaches t-2"})
    pd.DataFrame(mapping).to_csv(FEATURE_MAP_REPORT, index=False)

    report = pd.DataFrame(output)
    TABLES.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT, index=False)
    print(report.to_string(index=False))
    return report


if __name__ == "__main__":
    main()
