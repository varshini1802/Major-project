"""Explain every stale legacy sequence label and incomplete historical record."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from sequence_generation import normalize_keys, KEYS, TARGET  # noqa: E402

GROUPS = ["GROUP_HOMICIDE", "GROUP_SEXUAL_VIOLENCE", "GROUP_KIDNAPPING_ABDUCTION",
          "GROUP_ROBBERY_DACOITY", "GROUP_PUBLIC_VIOLENCE", "GROUP_PROPERTY_CRIME"]
LEGACY = HERE / "temporal_deep_learning_results"
OUT = HERE / "corrected_temporal_results/corrected_v2/sequences/legacy_sequence_row_audit.csv"


def audit(name, source, features, targets):
    metadata = normalize_keys(pd.read_csv(LEGACY / f"{name}_sequence_metadata.csv"), "legacy metadata")
    source = normalize_keys(source[KEYS + features], "feature source").set_index(KEYS)
    targets = normalize_keys(targets[KEYS + [TARGET]], "canonical target").set_index(KEYS)[TARGET]
    rows = []
    for year, state, unit, old_label in metadata[KEYS + [TARGET]].itertuples(index=False, name=None):
        key = (int(year), state, unit)
        new_label = targets.loc[key] if key in targets.index else np.nan
        stale = pd.isna(new_label) or int(old_label) != int(new_label)
        required = [(y, state, unit) for y in range(int(year) - 3, int(year))]
        absent = [y for y, s, u in required if (y, s, u) not in source.index]
        missing = []
        if not absent:
            values = source.loc[required, features].to_numpy(dtype=float)
            missing = [f"{required[i][0]}:{features[j]}"
                       for i, j in zip(*np.where(~np.isfinite(values)))]
        rows.append({"dataset": name, "YEAR": int(year), "STATE": state, "UNIT_NAME": unit,
                     "legacy_label": old_label, "canonical_label": new_label,
                     "stale_target_label": bool(stale),
                     "missing_historical_years": "|".join(map(str, absent)),
                     "missing_or_nonfinite_features": "|".join(missing),
                     "sequence_rejection_reason": "missing district-year record" if absent else
                         "missing/nonfinite selected feature" if missing else "valid complete history"})
    return rows


def main():
    grouped = pd.read_csv(ROOT / "data/processed/district_grouped_reduced.csv")
    enhanced = pd.read_csv(ROOT / "data/processed/district_enhanced_features.csv")
    targets = pd.read_csv(ROOT / "data/processed/district_targeted.csv")
    extra = [c for c in enhanced if c not in KEYS + GROUPS + [TARGET]
             and pd.api.types.is_numeric_dtype(enhanced[c])]
    rows = audit("baseline", grouped, GROUPS, targets)
    rows += audit("enhanced", enhanced, GROUPS + extra, targets)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    report = pd.DataFrame(rows)
    print(report.groupby(["dataset", "sequence_rejection_reason", "stale_target_label"])
          .size().rename("rows").to_string())
    print(f"Detailed legacy row audit saved to {OUT}")


if __name__ == "__main__":
    main()
