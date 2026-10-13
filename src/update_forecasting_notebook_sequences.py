"""Apply keyed-history and common-cohort corrections to notebook sequence cells."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/observed_year_classification_models.ipynb"

SEQUENCE_FUNCTION = '''def create_lstm_sequences(
    df,
    feature_columns,
    target_column,
    sequence_length=3
):
    """Build strict same-district histories without using target-year features."""
    key_columns = ["YEAR", "STATE", "UNIT_NAME"]
    forbidden = [c for c in feature_columns
                 if c == target_column or "TARGET" in c.upper() or "BURDEN" in c.upper()]
    if forbidden:
        raise ValueError(f"Outcome fields cannot be sequence inputs: {forbidden}")
    target_rows = df[key_columns + [target_column]].copy()
    source_rows = df[key_columns + feature_columns].copy()
    for frame in (target_rows, source_rows):
        frame["YEAR"] = pd.to_numeric(frame["YEAR"], errors="raise").astype(int)
        for col in ["STATE", "UNIT_NAME"]:
            frame[col] = frame[col].astype("string").str.strip().str.replace(r"\\s+", " ", regex=True).str.upper()
        if frame.duplicated(key_columns).any():
            raise ValueError("Duplicate district-year keys in sequence source")
    target_rows[target_column] = pd.to_numeric(target_rows[target_column], errors="coerce")
    source_rows[feature_columns] = source_rows[feature_columns].apply(pd.to_numeric, errors="coerce")
    source_rows[feature_columns] = source_rows[feature_columns].replace([np.inf, -np.inf], np.nan)
    lookup = source_rows.set_index(key_columns)
    sequences, labels, metadata = [], [], []
    eligible = target_rows.dropna(subset=[target_column]).sort_values(key_columns)
    for year, state, unit, label in eligible.itertuples(index=False, name=None):
        required = [(int(y), state, unit) for y in range(int(year)-sequence_length, int(year))]
        if any(key not in lookup.index for key in required):
            continue
        values = lookup.loc[required, feature_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            continue
        sequences.append(values.astype(np.float32))
        labels.append(int(label))
        metadata.append({"YEAR": int(year), "STATE": state, "UNIT_NAME": unit,
                         target_column: int(label),
                         "INPUT_YEARS": "|".join(str(key[0]) for key in required)})
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64), pd.DataFrame(metadata)
'''

COMMON_ALIGNMENT = '''# One stable sorted intersection is used for both arrays and labels.
common_key_list = sorted(common_keys.tolist(), key=lambda key: (int(key[0]), str(key[1]), str(key[2])))
baseline_lookup = {key: idx for idx, key in enumerate(baseline_key_index)}
enhanced_lookup = {key: idx for idx, key in enumerate(enhanced_key_index)}
baseline_indices = [baseline_lookup[key] for key in common_key_list]
enhanced_indices = [enhanced_lookup[key] for key in common_key_list]
assert len(baseline_indices) == len(enhanced_indices) == len(common_key_list)
'''

FINAL_ALIGNMENT = '''baseline_common_keys = baseline_metadata_common[KEY_COLUMNS].reset_index(drop=True)
enhanced_common_keys = enhanced_metadata_common[KEY_COLUMNS].reset_index(drop=True)
assert baseline_common_keys.equals(enhanced_common_keys)
assert np.array_equal(y_lstm_baseline_common, y_lstm_enhanced_common)
assert np.array_equal(baseline_metadata_common["INPUT_YEARS"], enhanced_metadata_common["INPUT_YEARS"])
assert np.array_equal(baseline_metadata_common[TARGET_COLUMN], y_lstm_baseline_common)
assert np.array_equal(enhanced_metadata_common[TARGET_COLUMN], y_lstm_enhanced_common)
'''


def main():
    notebook = json.loads(PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    function_index = next(i for i, cell in enumerate(cells)
                          if "def create_lstm_sequences" in "".join(cell.get("source", [])))
    common_index = next(i for i, cell in enumerate(cells)
                        if "baseline_lookup = {key: idx for idx, key in enumerate(baseline_key_index)}" in "".join(cell.get("source", [])))
    final_index = next(i for i, cell in enumerate(cells)
                       if "baseline_common_keys = baseline_metadata_common" in "".join(cell.get("source", [])))
    cells[function_index]["source"] = SEQUENCE_FUNCTION.splitlines(keepends=True)
    cells[common_index]["source"] = COMMON_ALIGNMENT.splitlines(keepends=True)
    cells[final_index]["source"] = FINAL_ALIGNMENT.splitlines(keepends=True)
    note = {
        "cell_type": "markdown", "metadata": {},
        "source": [
            "### Temporal sequence section: corrected code, legacy cached outputs\n",
            "\n",
            "The builder below uses only exact same-district t−3, t−2, t−1 feature rows; it does not use target-year predictor values to decide whether a sample is present. The baseline and enhanced arrays are restricted to the same sorted district-year key intersection and their target labels are asserted equal. Cached outputs from earlier notebook executions are retained as legacy records; corrected-v3 saved artifacts and results are authoritative until this section is rerun. Annual source publication timing remains unverified.\n",
        ],
    }
    if not any(c.get("cell_type") == "markdown" and "Temporal sequence section: corrected code" in "".join(c.get("source", [])) for c in cells):
        cells.insert(70, note)
    PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Updated temporal sequence integrity cells in {PATH}")


if __name__ == "__main__":
    main()
