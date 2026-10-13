import sys
from pathlib import Path
import unittest
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ML_Implementation"))
from sequence_generation import build_sequences
from sequence_cohort import common_sequence_cohort


class SequenceGenerationTests(unittest.TestCase):
    def data(self):
        features, targets = [], []
        for unit, offset in (("A", 0), ("B", 100)):
            for year in range(2000, 2007):
                features.append({"YEAR": year, "STATE": "S", "UNIT_NAME": unit,
                                 "F1": year + offset, "F2": 2 * year + offset})
                targets.append({"YEAR": year, "STATE": "S", "UNIT_NAME": unit,
                                "TARGET_VIOLENT_LEVEL": year % 5 + 1})
        return pd.DataFrame(targets), pd.DataFrame(features)

    def test_district_boundaries_and_target_alignment(self):
        targets, features = self.data()
        X, meta, _ = build_sequences(targets, features, ["F1", "F2"], target_years=[2003])
        self.assertEqual(len(meta), 2)
        for i, row in meta.reset_index(drop=True).iterrows():
            offset = 0 if row.UNIT_NAME == "A" else 100
            np.testing.assert_array_equal(X[i, :, 0], np.array([2000, 2001, 2002]) + offset)
            self.assertEqual(row.TARGET_VIOLENT_LEVEL, 2003 % 5 + 1)
            self.assertEqual(row.INPUT_YEARS, "2000|2001|2002")

    def test_missing_history_and_missing_feature_are_rejected(self):
        targets, features = self.data()
        features = features[~((features.UNIT_NAME == "A") & (features.YEAR == 2001))].copy()
        features.loc[(features.UNIT_NAME == "B") & (features.YEAR == 2001), "F2"] = np.nan
        _, meta, diag = build_sequences(targets, features, ["F1", "F2"], target_years=[2004])
        self.assertNotIn("A", meta.UNIT_NAME.tolist())  # t-3=2001 key absent
        self.assertNotIn("B", meta.UNIT_NAME.tolist())  # t-3 feature incomplete
        self.assertEqual(diag["rejected_missing_district_year_records"], 1)
        self.assertEqual(diag["rejected_incomplete_feature_records"], 1)

    def test_future_feature_changes_do_not_change_past_sequence(self):
        targets, features = self.data()
        base_X, base_meta, _ = build_sequences(targets, features, ["F1", "F2"], target_years=[2004])
        altered = features.copy()
        altered.loc[altered.YEAR >= 2004, ["F1", "F2"]] *= 10000
        changed_X, changed_meta, _ = build_sequences(targets, altered, ["F1", "F2"], target_years=[2004])
        pd.testing.assert_frame_equal(base_meta, changed_meta)
        np.testing.assert_array_equal(base_X, changed_X)
        self.assertTrue(all(max(map(int, years.split("|"))) < year
                             for year, years in zip(base_meta.YEAR, base_meta.INPUT_YEARS)))

    def test_duplicate_district_years_are_rejected(self):
        targets, features = self.data()
        features = pd.concat([features, features.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate normalized district-year"):
            build_sequences(targets, features, ["F1", "F2"], target_years=[2004])

    def test_target_or_burden_cannot_enter_sequence_features(self):
        targets, features = self.data()
        features["TARGET_VIOLENT_LEVEL"] = 1
        with self.assertRaisesRegex(ValueError, "Target or outcome-derived"):
            build_sequences(targets, features, ["F1", "TARGET_VIOLENT_LEVEL"], target_years=[2004])

    def test_generation_is_reproducible_and_rejects_infinity(self):
        targets, features = self.data()
        features["F1"] = features["F1"].astype(float)
        features.loc[(features.UNIT_NAME == "A") & (features.YEAR == 2001), "F1"] = np.inf
        x1, m1, d1 = build_sequences(targets, features, ["F1", "F2"], target_years=[2004])
        x2, m2, d2 = build_sequences(targets, features, ["F1", "F2"], target_years=[2004])
        np.testing.assert_array_equal(x1, x2)
        pd.testing.assert_frame_equal(m1, m2)
        self.assertEqual(d1, d2)
        self.assertNotIn("A", m1.UNIT_NAME.tolist())

    def test_common_cohort_has_identical_keys_labels_and_years(self):
        targets, features = self.data()
        xb, mb, _ = build_sequences(targets, features, ["F1", "F2"], target_years=[2003, 2004])
        xe, me, _ = build_sequences(targets, features, ["F1"], target_years=[2004])
        bc, bm, ec, em = common_sequence_cohort(xb, mb, xe, me)
        self.assertEqual(len(bm), 2)
        pd.testing.assert_frame_equal(bm, em)
        self.assertEqual(len(bc), len(ec))
        self.assertEqual(bm.YEAR.tolist(), [2004, 2004])

    def test_common_cohort_rejects_label_disagreement(self):
        targets, features = self.data()
        xb, mb, _ = build_sequences(targets, features, ["F1", "F2"], target_years=[2004])
        xe, me, _ = build_sequences(targets, features, ["F1"], target_years=[2004])
        me.loc[0, "TARGET_VIOLENT_LEVEL"] = me.loc[0, "TARGET_VIOLENT_LEVEL"] % 5 + 1
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            common_sequence_cohort(xb, mb, xe, me)

    def test_notebook_sequence_code_does_not_gate_on_target_year_features(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/observed_year_classification_models.ipynb").read_text(encoding="utf-8"))
        source = next("".join(cell.get("source", [])) for cell in notebook["cells"]
                      if "def create_lstm_sequences" in "".join(cell.get("source", [])))
        compile(source, "notebook_create_lstm_sequences", "exec")
        self.assertIn("target_rows = df[key_columns + [target_column]]", source)
        self.assertIn("source_rows = df[key_columns + feature_columns]", source)
        self.assertIn("lookup.loc[required, feature_columns]", source)
        self.assertNotIn("dropna(\n        subset=[target_column] + feature_columns", source)


if __name__ == "__main__":
    unittest.main()
