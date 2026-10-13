import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from feature_engineering import (  # noqa: E402
    GROUP_COLUMNS,
    engineer_enhanced_features,
)


class EngineeredFeatureAvailabilityTests(unittest.TestCase):
    def frame(self):
        rows = []
        # A has a missing calendar year 2002; B has two true zero-count years.
        for year, values in (
            (2000, [1, 2, 0, 0, 0, 0]),
            (2001, [2, np.nan, 0, 0, 0, 0]),
            (2003, [4, 0, 0, 0, 0, 0]),
        ):
            rows.append({"YEAR": year, "STATE": "S", "UNIT_NAME": "A",
                         **dict(zip(GROUP_COLUMNS, values)),
                         "TARGET_VIOLENT_LEVEL": 1})
        for year in (2000, 2001):
            rows.append({"YEAR": year, "STATE": "S", "UNIT_NAME": "B",
                         **dict.fromkeys(GROUP_COLUMNS, 0),
                         "TARGET_VIOLENT_LEVEL": 1})
        rows.append({"YEAR": 2000, "STATE": "S", "UNIT_NAME": "C",
                     **dict.fromkeys(GROUP_COLUMNS, np.nan),
                     "TARGET_VIOLENT_LEVEL": pd.NA})
        return pd.DataFrame(rows)

    def test_formulas_partial_missing_all_missing_and_zero_total(self):
        result = engineer_enhanced_features(self.frame()).set_index(["STATE", "UNIT_NAME", "YEAR"])
        a2001 = result.loc[("S", "A", 2001)]
        self.assertEqual(a2001.TOTAL_GROUPED_CRIME, 2)
        self.assertEqual(a2001.ACTIVE_CRIME_GROUPS, 1)
        self.assertEqual(a2001.PREV_YEAR_TOTAL_CRIME, 3)
        self.assertEqual(a2001.YOY_TOTAL_CRIME_CHANGE, -1)
        b2001 = result.loc[("S", "B", 2001)]
        self.assertEqual(b2001.TOTAL_GROUPED_CRIME, 0)
        self.assertEqual(b2001.CRIME_DIVERSITY_INDEX, 0)
        self.assertEqual(b2001.YOY_TOTAL_CRIME_CHANGE, 0)  # absolute difference; no denominator
        c2000 = result.loc[("S", "C", 2000)]
        self.assertTrue(pd.isna(c2000.TOTAL_GROUPED_CRIME))
        self.assertTrue(pd.isna(c2000.ACTIVE_CRIME_GROUPS))
        self.assertTrue(pd.isna(c2000.CRIME_DIVERSITY_INDEX))

    def test_missing_prior_calendar_year_stays_missing(self):
        result = engineer_enhanced_features(self.frame()).set_index(["STATE", "UNIT_NAME", "YEAR"])
        a2003 = result.loc[("S", "A", 2003)]
        self.assertTrue(pd.isna(a2003.PREV_YEAR_TOTAL_CRIME))
        self.assertTrue(pd.isna(a2003.YOY_TOTAL_CRIME_CHANGE))

    def test_keyed_alignment_is_invariant_to_input_row_order(self):
        frame = self.frame()
        original = engineer_enhanced_features(frame).sort_values(["STATE", "UNIT_NAME", "YEAR"]).reset_index(drop=True)
        shuffled = engineer_enhanced_features(frame.sample(frac=1, random_state=71)).sort_values(
            ["STATE", "UNIT_NAME", "YEAR"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(original, shuffled)

    def test_duplicate_district_year_is_rejected_by_validated_lookup(self):
        frame = self.frame()
        duplicated = pd.concat([frame, frame.loc[(frame.UNIT_NAME == "A") & (frame.YEAR == 2000)]],
                               ignore_index=True)
        with self.assertRaises(pd.errors.MergeError):
            engineer_enhanced_features(duplicated)


if __name__ == "__main__":
    unittest.main()
