"""Strict district-keyed sequence construction for chronological forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TARGET = "TARGET_VIOLENT_LEVEL"


def normalize_keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    missing = [column for column in KEYS if column not in result]
    if missing:
        raise ValueError(f"{name} missing keys: {missing}")
    result["YEAR"] = pd.to_numeric(result["YEAR"], errors="raise").astype(int)
    for column in KEYS[1:]:
        result[column] = (result[column].astype("string").str.strip()
                          .str.replace(r"\s+", " ", regex=True).str.upper())
    if result.duplicated(KEYS).any():
        raise ValueError(f"{name} contains duplicate normalized district-year keys")
    return result


def build_sequences(targets: pd.DataFrame, features: pd.DataFrame,
                    feature_columns: list[str], lookback: int = 3,
                    target_years: list[int] | None = None
                    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Build samples for year t from complete feature records t-L..t-1.

    A sequence is rejected if any required district-year key is absent or any
    selected feature is nonnumeric, infinite, or missing on a required row.
    No padding or imputation occurs here. Any later learned preprocessing is
    fit only on training sequences.
    """
    if lookback < 1:
        raise ValueError("lookback must be positive")
    forbidden = [c for c in feature_columns
                 if c in {TARGET, "VIOLENT_CRIME_BURDEN"}
                 or "TARGET" in c.upper() or "BURDEN" in c.upper()]
    if forbidden:
        raise ValueError(f"Target or outcome-derived fields cannot be sequence inputs: {forbidden}")
    missing_features = [c for c in feature_columns if c not in features]
    if missing_features:
        raise ValueError(f"Feature data missing columns: {missing_features}")
    if TARGET not in targets:
        raise ValueError(f"Targets missing {TARGET}")

    target_frame = normalize_keys(targets[KEYS + [TARGET]], "targets")
    feature_frame = normalize_keys(features[KEYS + feature_columns], "features")
    target_frame[TARGET] = pd.to_numeric(target_frame[TARGET], errors="coerce")
    numeric = feature_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    feature_frame[feature_columns] = numeric
    lookup = feature_frame.set_index(KEYS)

    candidate = target_frame.dropna(subset=[TARGET]).copy()
    candidate = candidate[candidate[TARGET].isin([1, 2, 3, 4, 5])]
    if target_years is not None:
        candidate = candidate[candidate.YEAR.isin(target_years)]

    sequences, metadata = [], []
    missing_key_targets = []
    incomplete_feature_targets = []
    for row in candidate.itertuples(index=False, name=None):
        year, state, unit, label = row
        required_keys = [(int(y), state, unit) for y in range(year - lookback, year)]
        absent = [key for key in required_keys if key not in lookup.index]
        if absent:
            missing_key_targets.append((int(year), state, unit, absent))
            continue
        values = lookup.loc[required_keys, feature_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            incomplete_feature_targets.append((int(year), state, unit))
            continue
        sequences.append(values.astype(np.float32))
        metadata.append({
            "YEAR": int(year), "STATE": state, "UNIT_NAME": unit,
            TARGET: int(label),
            "INPUT_YEARS": "|".join(str(key[0]) for key in required_keys),
        })

    shape = (0, lookback, len(feature_columns))
    X = np.stack(sequences).astype(np.float32) if sequences else np.empty(shape, dtype=np.float32)
    meta = pd.DataFrame(metadata, columns=KEYS + [TARGET, "INPUT_YEARS"])
    diagnostics = {
        "target_observations_considered": int(len(candidate)),
        "valid_sequences": int(len(meta)),
        "rejected_missing_district_year_records": int(len(missing_key_targets)),
        "rejected_incomplete_feature_records": int(len(incomplete_feature_targets)),
        "rejected_total": int(len(missing_key_targets) + len(incomplete_feature_targets)),
        "lookback": int(lookback),
        "feature_columns": list(feature_columns),
    }
    return X, meta, diagnostics
