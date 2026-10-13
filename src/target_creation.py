"""Reproducible construction of the project-defined violent burden target.

The base-paper target rule is not available in the repository. This module
therefore documents a project adaptation: IPC count sources are summed, and
four quintile cut points are fitted on 2001--2010 training observations then
frozen for all years. Ties are assigned as a unit with right-closed cuts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "district_integrated_final.csv"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
TARGET_COLUMN = "TARGET_VIOLENT_LEVEL"
BURDEN_COLUMN = "VIOLENT_CRIME_BURDEN"
DEFAULT_FIT_YEARS = tuple(range(2001, 2011))
CLASS_LABELS = {
    1: "Lowest violent-crime burden",
    2: "Low-to-moderate violent-crime burden",
    3: "Moderate violent-crime burden",
    4: "Moderate-to-high violent-crime burden",
    5: "Highest violent-crime burden",
}

BASE_PAPER_GROUPS = {
    "CHNOT_AMOUNTING_TO_MURDER": ["IPC_CULPABLE_HOMICIDE_NOT_AMOUNTING_TO_MURDER"],
    "RIOTS": ["IPC_RIOTS", "IPC_RIOTS_COMMUNAL", "IPC_RIOTS_INDUSTRIAL", "IPC_RIOTS_POLITICAL", "IPC_RIOTS_CASTE_CONFLICT", "IPC_RIOTS_AGRARIAN", "IPC_RIOTS_STUDENTS", "IPC_RIOTS_SECTARIAN"],
    "RAPE": ["IPC_RAPE", "IPC_OTHER_RAPE", "IPC_CUSTODIAL_RAPE", "IPC_CUSTODIAL_GANG_RAPE", "IPC_RAPE_GANG_RAPE", "IPC_RAPE_OTHERS"],
    "MURDER": ["IPC_MURDER"],
    "PREPARATION_AND_ASSEMBLY_FOR_DACOITY": ["IPC_PREPARATION_AND_ASSEMBLY_FOR_DACOITY"],
    "DOWRY_DEATH": ["IPC_DOWRY_DEATHS"],
    "ROBBERY": ["IPC_ROBBERY"],
    "DACOITY": ["IPC_DACOITY", "IPC_OTHER_DACOITY", "IPC_DACOITY_WITH_MURDER"],
    "KIDNAPPING_ABDUCTION": ["IPC_KIDNAPPING_AND_ABDUCTION_OF_WOMEN_AND_GIRLS", "IPC_KIDNAPPING_AND_ABDUCTION_OF_OTHERS", "IPC_KIDNAPPING_ABDUCTION", "IPC_KIDNAPPING_FOR_RANSOM"],
    "ARSON": ["IPC_ARSON"],
    "ATTEMPT_TO_MURDER": ["IPC_ATTEMPT_TO_MURDER"],
}


def fit_thresholds(burden: pd.Series, years: pd.Series | None = None,
                   fit_years=DEFAULT_FIT_YEARS) -> np.ndarray:
    """Fit and validate four training-only quintile boundaries."""
    valid = burden.notna() & np.isfinite(pd.to_numeric(burden, errors="coerce"))
    if years is not None:
        valid &= pd.to_numeric(years, errors="coerce").isin(fit_years)
    fitting = pd.to_numeric(burden.loc[valid], errors="coerce")
    if fitting.empty:
        raise ValueError(f"No valid burden observations in threshold fitting years {list(fit_years)}.")
    boundaries = np.quantile(fitting, [0.2, 0.4, 0.6, 0.8])
    boundaries = np.unique(boundaries.astype(float))
    if len(boundaries) != 4:
        raise ValueError(f"Five classes require four distinct training cut points; found {len(boundaries)}. Inspect training burden distribution.")
    return boundaries


def create_target(df: pd.DataFrame, fit_years=DEFAULT_FIT_YEARS,
                  thresholds=None) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy with burden and labels, plus original IPC source names.

    Each group's available numeric IPC columns are summed with min_count=1;
    group totals are then summed with min_count=1. Nonnumeric and nonfinite
    values are treated as missing; any observed negative source invalidates
    that row because counts cannot be negative. If YEAR is absent, thresholds
    are fitted on the supplied data (observed-year classification use).
    """
    result = df.copy()
    group_totals, sources, invalid_negative = [], [], pd.Series(False, index=df.index)
    for columns in BASE_PAPER_GROUPS.values():
        present = [c for c in columns if c in result.columns]
        sources.extend(present)
        if not present:
            continue
        values = result[present].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        invalid_negative |= values.lt(0).any(axis=1)
        group_totals.append(values.sum(axis=1, min_count=1))
    if not group_totals:
        raise ValueError("No verified violent-crime IPC source columns were found.")
    burden = pd.concat(group_totals, axis=1).sum(axis=1, min_count=1)
    burden = burden.mask(invalid_negative)
    if not burden.notna().any():
        raise ValueError("No valid violent-crime burden values exist after excluding missing or negative counts.")
    years = result["YEAR"] if "YEAR" in result.columns else None
    if thresholds is None:
        boundaries = fit_thresholds(burden, years, fit_years if years is not None else burden.index)
    else:
        boundaries = np.asarray(thresholds, dtype=float)
        if (boundaries.shape != (4,) or not np.isfinite(boundaries).all()
                or not np.all(np.diff(boundaries) > 0)):
            raise ValueError("Frozen thresholds must contain four finite, strictly increasing cut points.")
    labels = pd.Series(pd.NA, index=result.index, dtype="Int64")
    valid = burden.notna()
    labels.loc[valid] = np.searchsorted(boundaries, burden.loc[valid].to_numpy(), side="right") + 1
    if labels.loc[valid].nunique() != 5:
        raise ValueError("Frozen training thresholds do not produce all five classes on the supplied data.")
    result[BURDEN_COLUMN] = burden
    result[TARGET_COLUMN] = labels
    result.attrs["target_thresholds"] = boundaries.tolist()
    result.attrs["threshold_fit_years"] = (list(fit_years) if years is not None else "entire supplied dataset") if thresholds is None else "externally supplied frozen thresholds"
    return result, list(dict.fromkeys(sources))


def create_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    return pd.DataFrame([{"target_level": "MISSING" if pd.isna(level) else int(level),
                          "count": int(count), "percentage": round(count / len(df) * 100, 2)}
                         for level, count in counts.items()])


if __name__ == "__main__":
    frame = pd.read_csv(DATA_PATH)
    target_df, target_sources = create_target(frame)
    boundaries = target_df.attrs["target_thresholds"]
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    distribution = create_target_distribution(target_df)
    distribution.to_csv(OUTPUT_TABLES_DIR / "final_target_distribution.csv", index=False)
    pd.DataFrame({"target": TARGET_COLUMN, "source_column": target_sources,
                  "role": "target_construction_source"}).to_csv(OUTPUT_TABLES_DIR / "target_source_column_mapping.csv", index=False)
    pd.DataFrame([{"boundary_index": i + 1, "upper_cut_point": v,
                   "rule": "class = searchsorted(cut_points, burden, side='right') + 1",
                   "fit_years": "2001-2010", "method": "linear empirical quintiles"}
                  for i, v in enumerate(boundaries)]).to_csv(OUTPUT_TABLES_DIR / "target_thresholds.csv", index=False)
    target_df.to_csv(PROJECT_ROOT / "data" / "processed" / "district_targeted.csv", index=False)
