"""Keyed common-cohort construction for fair sequence comparisons."""
import numpy as np
import pandas as pd

from sequence_generation import KEYS, TARGET, normalize_keys


def common_sequence_cohort(base_X: np.ndarray, base_meta: pd.DataFrame,
                           enhanced_X: np.ndarray, enhanced_meta: pd.DataFrame
                           ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Return baseline/enhanced arrays on identical valid target district-years."""
    if len(base_X) != len(base_meta) or len(enhanced_X) != len(enhanced_meta):
        raise ValueError("Sequence arrays and metadata lengths differ")
    if base_X.ndim != 3 or enhanced_X.ndim != 3 or base_X.shape[1] != enhanced_X.shape[1]:
        raise ValueError("Sequence arrays must be rank 3 with the same lookback")
    b = normalize_keys(base_meta, "baseline metadata").reset_index(drop=True)
    e = normalize_keys(enhanced_meta, "enhanced metadata").reset_index(drop=True)
    if b.duplicated(KEYS).any() or e.duplicated(KEYS).any():
        raise ValueError("Sequence metadata contains duplicate target keys")
    if not np.isfinite(base_X).all() or not np.isfinite(enhanced_X).all():
        raise ValueError("Input sequence arrays contain nonfinite values")
    for frame in (b, e):
        for year, years in zip(frame.YEAR, frame.INPUT_YEARS):
            wanted = list(range(int(year) - 3, int(year)))
            actual = [int(value) for value in str(years).split("|")]
            if actual != wanted or max(actual) >= int(year):
                raise ValueError("Metadata contains incorrect or future input years")
    b[TARGET] = pd.to_numeric(b[TARGET], errors="raise").astype(int)
    e[TARGET] = pd.to_numeric(e[TARGET], errors="raise").astype(int)
    b_index = b.reset_index(names="_baseline_index")
    e_index = e.reset_index(names="_enhanced_index")
    common = b_index.merge(e_index[KEYS + [TARGET, "_enhanced_index"]], on=KEYS,
                           how="inner", suffixes=("_base", "_enhanced"), validate="one_to_one")
    if not common[f"{TARGET}_base"].eq(common[f"{TARGET}_enhanced"]).all():
        raise ValueError("Baseline and enhanced labels disagree on common target keys")
    common = common.sort_values(["YEAR", "STATE", "UNIT_NAME"], kind="stable").reset_index(drop=True)
    bi = common["_baseline_index"].to_numpy(dtype=int)
    ei = common["_enhanced_index"].to_numpy(dtype=int)
    base_common = b.iloc[bi].reset_index(drop=True)
    enhanced_common = e.iloc[ei].reset_index(drop=True)
    if not base_common[KEYS + [TARGET, "INPUT_YEARS"]].equals(enhanced_common[KEYS + [TARGET, "INPUT_YEARS"]]):
        raise AssertionError("Common sequence metadata failed exact alignment")
    return base_X[bi], base_common, enhanced_X[ei], enhanced_common
