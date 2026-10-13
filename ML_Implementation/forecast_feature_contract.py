"""Approved feature definitions and strict keyed construction for forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TARGET = "TARGET_VIOLENT_LEVEL"
GROUPS = ["GROUP_HOMICIDE", "GROUP_SEXUAL_VIOLENCE", "GROUP_KIDNAPPING_ABDUCTION",
          "GROUP_ROBBERY_DACOITY", "GROUP_PUBLIC_VIOLENCE", "GROUP_PROPERTY_CRIME"]
ENGINEERED = ["TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX",
              "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"]
APPROVED = {"baseline": GROUPS, "enhanced": GROUPS + ENGINEERED}
FORBIDDEN = {TARGET, "VIOLENT_CRIME_BURDEN", "SOURCE_YEAR", "PREVIOUS_YEAR"}


def normalize(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.copy()
    absent = [c for c in KEYS if c not in result]
    if absent:
        raise ValueError(f"{label} lacks keys {absent}")
    result["YEAR"] = pd.to_numeric(result.YEAR, errors="raise").astype(int)
    for col in KEYS[1:]:
        result[col] = (result[col].astype("string").str.strip()
                       .str.replace(r"\s+", " ", regex=True).str.upper())
    if result.duplicated(KEYS).any():
        raise ValueError(f"{label} contains duplicate normalized district-year keys")
    return result


def build_forecast_matrix(targets: pd.DataFrame, source: pd.DataFrame,
                          features: list[str], variant: str) -> tuple[pd.DataFrame, list[str]]:
    """Create target-year rows with feature values sourced only from t-1."""
    if variant not in APPROVED or features != APPROVED[variant]:
        raise ValueError(f"Unapproved {variant} feature list: {features}")
    if any(name in FORBIDDEN or "TARGET" in name or "BURDEN" in name for name in features):
        raise ValueError("Target or post-outcome fields cannot be forecasting predictors")
    target_frame = normalize(targets[KEYS + [TARGET]], "canonical targets")
    target_frame[TARGET] = pd.to_numeric(target_frame[TARGET], errors="coerce")
    target_frame = target_frame.dropna(subset=[TARGET]).copy()
    source_frame = normalize(source[KEYS + features], f"{variant} source")
    for feature in features:
        source_frame[feature] = pd.to_numeric(source_frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    history = source_frame.copy()
    history["SOURCE_YEAR"] = history.YEAR
    history["YEAR"] += 1
    history = history.rename(columns={c: f"LAG1_{c}" for c in features})
    result = target_frame.merge(history[KEYS + ["SOURCE_YEAR"] + [f"LAG1_{c}" for c in features]],
                                on=KEYS, how="inner", validate="one_to_one")
    result = result[result.YEAR.eq(result.SOURCE_YEAR + 1)]
    selected = [f"LAG1_{c}" for c in features]
    result = result.dropna(subset=selected, how="all").copy()
    result[TARGET] = pd.to_numeric(result[TARGET], errors="raise").astype(int)
    if len(result) != len(result.drop_duplicates(KEYS)):
        raise AssertionError("Forecast matrix contains repeated target keys")
    return result, selected


def validate_matrix(matrix: pd.DataFrame, feature_columns: list[str], variant: str) -> None:
    expected = [f"LAG1_{c}" for c in APPROVED[variant]]
    if feature_columns != expected:
        raise AssertionError(f"{variant}: feature matrix does not match approved contract")
    if set(feature_columns) & FORBIDDEN or any("TARGET" in c or "BURDEN" in c for c in feature_columns):
        raise AssertionError("Target-derived field found in forecasting feature matrix")
    if not matrix.YEAR.eq(matrix.SOURCE_YEAR + 1).all():
        raise AssertionError("A forecast input is not sourced from target year minus one")
    if matrix.duplicated(KEYS).any():
        raise AssertionError("Forecast matrix has duplicate district-year keys")


def validate_lineage_report(report: pd.DataFrame) -> None:
    """Fail if any approved forecasting feature lacks source/formula/time evidence."""
    expected_experiments = {
        **{f"corrected_tabular_{v}": names for v, names in APPROVED.items()},
        **{f"corrected_sequence_{v}": names for v, names in APPROVED.items()},
        "retired_temporal_leakage_free_runner": GROUPS + ENGINEERED,
        "retired_combined_temporal_sequence_runner": GROUPS + ENGINEERED,
        "retired_temporal_deep_learning_sequence_runner": GROUPS + ENGINEERED,
        "notebook_observed_year_classification": GROUPS + ["PREV_YEAR_TOTAL_CRIME"],
        "notebook_lstm_sequence_code": GROUPS + ["PREV_YEAR_TOTAL_CRIME"],
        "same_year_feature_candidates_excluded": GROUPS + ENGINEERED,
    }
    for experiment, names in expected_experiments.items():
        subset = report[report.experiment.eq(experiment)]
        if subset.predictor.tolist() != names:
            raise AssertionError(f"Lineage list for {experiment} differs from approved features")
        for column in ["original_source_columns", "aggregation_or_transformation",
                       "source_observation_year", "target_year", "available_by_forecast_origin",
                       "decision", "evidence"]:
            if subset[column].isna().any() or subset[column].astype(str).str.strip().eq("").any():
                raise AssertionError(f"{experiment} has undocumented lineage field {column}")
        if experiment.startswith("corrected_tabular_") and subset.source_observation_year.str.contains(
                r"=t(?![-0-9])|target year t", case=False, regex=True).any():
            raise AssertionError(f"{experiment} contains a same-year forecasting source")

