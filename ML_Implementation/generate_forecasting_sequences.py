"""Generate strict 3-year sequence inputs and metadata without training models."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent
ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(ROOT / "src"))
from sequence_generation import build_sequences, normalize_keys, KEYS, TARGET  # noqa: E402

OUT = ML_DIR / "corrected_temporal_results" / "corrected_v2" / "sequences"
TRAIN_YEARS = list(range(2001, 2011))
VALIDATION_YEARS = [2011, 2012]
TEST_YEARS = [2013, 2014]
GROUPS = [
    "GROUP_HOMICIDE", "GROUP_SEXUAL_VIOLENCE", "GROUP_KIDNAPPING_ABDUCTION",
    "GROUP_ROBBERY_DACOITY", "GROUP_PUBLIC_VIOLENCE", "GROUP_PROPERTY_CRIME",
]


def legacy_diagnostic(path: Path, source: pd.DataFrame, features: list[str], target: pd.DataFrame,
                      row_details: list[dict]) -> dict:
    old = normalize_keys(pd.read_csv(path), path.name)
    d = normalize_keys(source[KEYS + features], "feature source").set_index(KEYS)
    t = normalize_keys(target[KEYS + [TARGET]], "canonical target").set_index(KEYS)[TARGET]
    invalid_missing_keys = 0
    invalid_incomplete_values = 0
    label_mismatches = 0
    for year, state, unit, saved_label in old[KEYS + [TARGET]].itertuples(index=False, name=None):
        key = (int(year), state, unit)
        label_mismatch = key not in t.index or pd.isna(t.loc[key]) or int(saved_label) != int(t.loc[key])
        if label_mismatch:
            label_mismatches += 1
        required = [(y, state, unit) for y in range(year - 3, year)]
        absent = [k[0] for k in required if k not in d.index]
        incomplete = []
        if absent:
            invalid_missing_keys += 1
        else:
            vals = d.loc[required, features].to_numpy(dtype=float)
            incomplete = [f"{required[i][0]}:{features[j]}" for i, j in zip(*np.where(~np.isfinite(vals)))]
        if not absent and incomplete:
            invalid_incomplete_values += 1
        row_details.append({"dataset": path.stem.replace("_sequence_metadata", ""),
                            "YEAR": int(year), "STATE": state, "UNIT_NAME": unit,
                            "legacy_label": saved_label,
                            "canonical_label": t.loc[key] if key in t.index else np.nan,
                            "stale_target_label": bool(label_mismatch),
                            "missing_historical_years": "|".join(map(str, absent)),
                            "missing_or_nonfinite_features": "|".join(incomplete),
                            "sequence_rejection_reason": ("missing district-year record" if absent else
                                "missing/nonfinite selected feature" if incomplete else "valid complete history")})
    return {
        "legacy_metadata_file": path.name,
        "legacy_rows": len(old),
        "legacy_target_label_mismatches": label_mismatches,
        "legacy_valid_complete_sequences": len(old) - invalid_missing_keys - invalid_incomplete_values,
        "legacy_invalid_missing_historical_key": invalid_missing_keys,
        "legacy_invalid_missing_or_nonfinite_features": invalid_incomplete_values,
    }


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite prior sequence artifacts: {OUT}")
    grouped = pd.read_csv(ROOT / "data/processed/district_grouped_reduced.csv")
    enhanced = pd.read_csv(ROOT / "data/processed/district_enhanced_features.csv")
    target = pd.read_csv(ROOT / "data/processed/district_targeted.csv")[KEYS + [TARGET]]
    target_norm = normalize_keys(target, "target")
    for name, source in (("baseline", grouped), ("enhanced", enhanced)):
        source_norm = normalize_keys(source, name)
        merged = source_norm[KEYS].merge(target_norm[KEYS + [TARGET]], on=KEYS,
                                         how="left", validate="one_to_one", indicator=True)
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"{name} has district-year keys without canonical targets")
        expected = pd.to_numeric(merged[TARGET], errors="coerce")
        actual = pd.to_numeric(source_norm[TARGET], errors="coerce")
        if not ((actual.eq(expected)) | (actual.isna() & expected.isna())).all():
            raise ValueError(f"{name} labels do not match canonical target")

    enhanced_features = GROUPS + [c for c in enhanced.columns
                                  if c not in KEYS + GROUPS + [TARGET]
                                  and pd.api.types.is_numeric_dtype(enhanced[c])]
    feature_sets = {"baseline": (grouped, GROUPS), "enhanced": (enhanced, enhanced_features)}
    OUT.mkdir(parents=True)
    reports = []
    splits = []
    legacy_rows = []
    for name, (source, features) in feature_sets.items():
        legacy_path = ML_DIR / "temporal_deep_learning_results" / f"{name}_sequence_metadata.csv"
        before = legacy_diagnostic(legacy_path, source, features, target, legacy_rows)
        X, meta, diag = build_sequences(
            target, source, features, lookback=3,
            target_years=TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS,
        )
        if not np.isfinite(X).all():
            raise AssertionError(f"{name}: generated sequence contains nonfinite values")
        if not (meta.INPUT_YEARS.str.split("|").map(lambda ys: len(ys) == 3).all()
                and meta.apply(lambda r: max(map(int, r.INPUT_YEARS.split("|"))) < r.YEAR, axis=1).all()):
            raise AssertionError(f"{name}: future or incomplete sequence input years detected")
        target_by_key = target_norm.set_index(KEYS)[TARGET]
        aligned = [int(target_by_key.loc[(r.YEAR, r.STATE, r.UNIT_NAME)]) for r in meta.itertuples()]
        if aligned != meta[TARGET].astype(int).tolist():
            raise AssertionError(f"{name}: sequence targets do not align to target-year keys")

        d = {**before, **diag, "dataset": name, "post_refresh_sequence_rows": len(meta),
             "sequence_input_shape": str(list(X.shape)), "target_key_alignment_mismatches": 0,
             "post_refresh_target_label_mismatches": 0,
             "post_refresh_incomplete_sequences": 0,
             "future_input_rows": 0}
        reports.append(d)
        meta.to_csv(OUT / f"{name}_sequence_metadata.csv", index=False)
        np.savez_compressed(
            OUT / f"{name}_sequences.npz", X=X,
            y=meta[TARGET].to_numpy(dtype=np.int64),
            years=meta.YEAR.to_numpy(dtype=np.int64),
            states=meta.STATE.to_numpy(dtype=str),
            units=meta.UNIT_NAME.to_numpy(dtype=str),
            feature_names=np.asarray(features, dtype=str),
            input_years=meta.INPUT_YEARS.to_numpy(dtype=str),
        )
        for split_name, years in (("train", TRAIN_YEARS), ("validation", VALIDATION_YEARS), ("test", TEST_YEARS)):
            subset = meta[meta.YEAR.isin(years)]
            for cls in range(1, 6):
                splits.append({"dataset": name, "partition": split_name,
                               "target_years": f"{min(years)}-{max(years)}", "class": cls,
                               "count": int((subset[TARGET] == cls).sum()), "rows": len(subset)})

    pd.DataFrame(reports).to_csv(OUT / "sequence_refresh_report.csv", index=False)
    pd.DataFrame(legacy_rows).to_csv(OUT / "legacy_sequence_row_audit.csv", index=False)
    pd.DataFrame(splits).to_csv(OUT / "sequence_class_distributions.csv", index=False)
    config = {
        "target": TARGET, "target_source": "data/processed/district_targeted.csv",
        "thresholds": [107.0, 237.0, 362.0, 563.0], "threshold_fit_years": "2001-2010",
        "train_years": TRAIN_YEARS, "validation_years": VALIDATION_YEARS,
        "test_years": TEST_YEARS, "lookback": 3,
        "sequence_rule": "For target year t, require exact same-district feature rows t-3,t-2,t-1 and finite values for every selected feature. Reject incomplete sequences; do not pad or impute during generation.",
        "random_seed": 42, "baseline_features": GROUPS,
        "enhanced_features": enhanced_features,
        "generation_script": "ML_Implementation/generate_forecasting_sequences.py",
    }
    (OUT / "sequence_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(reports).to_string(index=False))
    print(f"Corrected sequence arrays and metadata saved under {OUT}")


if __name__ == "__main__":
    main()
