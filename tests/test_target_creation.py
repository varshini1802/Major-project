import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from target_creation import create_target, fit_thresholds


class TargetCreationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"YEAR": [2001] * 10 + [2011, 2013],
                             "IPC_MURDER": list(range(1, 11)) + [3, 8]})

    def test_ties_and_row_order_are_stable(self):
        frame = pd.DataFrame({"YEAR": [2001] * 15,
                              "IPC_MURDER": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]})
        result, _ = create_target(frame)
        self.assertEqual(result.groupby("VIOLENT_CRIME_BURDEN")["TARGET_VIOLENT_LEVEL"].nunique().max(), 1)
        shuffled = frame.sample(frac=1, random_state=22)
        shuffled_result, _ = create_target(shuffled)
        a = result.set_index("IPC_MURDER").TARGET_VIOLENT_LEVEL
        b = shuffled_result.set_index("IPC_MURDER").TARGET_VIOLENT_LEVEL
        self.assertTrue(a.sort_index().equals(b.sort_index()))

    def test_class_domain_and_missing_source(self):
        frame = self.frame().iloc[:10].copy()
        frame.loc[0, "IPC_MURDER"] = None
        result, _ = create_target(frame)
        self.assertTrue(set(result.TARGET_VIOLENT_LEVEL.dropna().astype(int)) <= {1, 2, 3, 4, 5})
        self.assertTrue(pd.isna(result.loc[0, "TARGET_VIOLENT_LEVEL"]))

    def test_thresholds_are_reproducible_and_fit_training_years_only(self):
        frame = self.frame()
        baseline, _ = create_target(frame)
        changed_holdout = frame.copy()
        changed_holdout.loc[changed_holdout.YEAR > 2010, "IPC_MURDER"] *= 1000
        changed, _ = create_target(changed_holdout)
        self.assertEqual(baseline.attrs["target_thresholds"], changed.attrs["target_thresholds"])
        import numpy as np
        np.testing.assert_array_equal(
            fit_thresholds(pd.Series(range(1, 11)), pd.Series([2001] * 10)),
            fit_thresholds(pd.Series(range(1, 11)), pd.Series([2001] * 10)))

    def test_insufficient_distinct_cut_points_errors(self):
        with self.assertRaisesRegex(ValueError, "four distinct training cut points"):
            fit_thresholds(pd.Series([1] * 20), pd.Series([2001] * 20))

    def test_expected_interface(self):
        result, sources = create_target(self.frame())
        self.assertIn("TARGET_VIOLENT_LEVEL", result)
        self.assertIn("IPC_MURDER", sources)

    def test_frozen_thresholds_are_applied_without_refitting(self):
        thresholds = [107.0, 237.0, 362.0, 563.0]
        frame = pd.DataFrame({
            "YEAR": [2010, 2011, 2012, 2013, 2014],
            "IPC_MURDER": [10, 107, 237, 362, 563],
        })
        result, _ = create_target(frame, thresholds=thresholds)
        self.assertEqual(result.TARGET_VIOLENT_LEVEL.astype(int).tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(result.attrs["target_thresholds"], thresholds)

    def test_invalid_frozen_thresholds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "four finite, strictly increasing"):
            create_target(self.frame(), thresholds=[107, 237, 237, 563])


if __name__ == "__main__":
    unittest.main()
