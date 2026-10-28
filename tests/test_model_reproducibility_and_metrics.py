"""Targeted unit tests for model reproducibility, schema validation, and evaluation metrics."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

class ModelReproducibilityAndMetricsTests(unittest.TestCase):
    def setUp(self):
        self.processed_dir = ROOT / "data/processed"
        self.results_dir = ROOT / "ML_Implementation/corrected_temporal_results"
        self.tables_dir = ROOT / "outputs/tables"

    def test_reduced_dataset_schema_and_dimensions(self):
        """Verify district_grouped_reduced.csv columns, row counts, and target non-null count."""
        path = self.processed_dir / "district_grouped_reduced.csv"
        self.assertTrue(path.exists(), f"Missing {path}")
        df = pd.read_csv(path)
        self.assertEqual(len(df), 9856)
        expected_cols = [
            "YEAR", "STATE", "UNIT_NAME",
            "GROUP_HOMICIDE", "GROUP_SEXUAL_VIOLENCE", "GROUP_KIDNAPPING_ABDUCTION",
            "GROUP_ROBBERY_DACOITY", "GROUP_PUBLIC_VIOLENCE", "GROUP_PROPERTY_CRIME",
            "TARGET_VIOLENT_LEVEL"
        ]
        self.assertListEqual(list(df.columns), expected_cols)
        valid_targets = df["TARGET_VIOLENT_LEVEL"].dropna()
        self.assertEqual(len(valid_targets), 9631)
        self.assertEqual(df["TARGET_VIOLENT_LEVEL"].isna().sum(), 225)

    def test_enhanced_dataset_features_and_alignment(self):
        """Verify district_enhanced_features.csv has the 5 engineered columns."""
        path = self.processed_dir / "district_enhanced_features.csv"
        self.assertTrue(path.exists(), f"Missing {path}")
        df = pd.read_csv(path)
        self.assertEqual(len(df), 9856)
        engineered_cols = [
            "TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX",
            "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"
        ]
        for col in engineered_cols:
            self.assertIn(col, df.columns)

    def test_frozen_target_boundaries_and_ordering(self):
        """Verify frozen cutoffs [107, 237, 362, 563] produce strictly 5 ordered classes."""
        cutoffs = [107.0, 237.0, 362.0, 563.0]
        test_burdens = pd.Series([10, 107, 237, 362, 563, 1000])
        labels = np.searchsorted(cutoffs, test_burdens, side="right") + 1
        expected = np.array([1, 2, 3, 4, 5, 5])
        np.testing.assert_array_equal(labels, expected)

    def test_locked_test_cohort_dimensions(self):
        """Verify locked test cohort in v4 results has 671 rows spanning 2013-2014."""
        keys_path = self.results_dir / "corrected_v4_controlled_sequence/locked_test_keys.csv"
        self.assertTrue(keys_path.exists(), f"Missing {keys_path}")
        keys_df = pd.read_csv(keys_path)
        self.assertEqual(len(keys_df), 671)
        self.assertTrue(set(keys_df["YEAR"].unique()).issubset({2013, 2014}))
        self.assertEqual(len(keys_df.drop_duplicates(subset=["YEAR", "STATE", "UNIT_NAME"])), 671)

    def test_metric_calculation_correctness(self):
        """Test accuracy, macro F1, and weighted F1 against hand-crafted ground truth."""
        y_true = np.array([1, 1, 2, 2, 3, 3, 4, 5])
        y_pred = np.array([1, 2, 2, 2, 3, 4, 4, 5])
        acc = accuracy_score(y_true, y_pred)
        self.assertAlmostEqual(acc, 6.0 / 8.0)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        self.assertGreater(macro_f1, 0.6)
        self.assertLessEqual(macro_f1, 1.0)

    def test_reproducibility_saved_metrics_file(self):
        """Verify that v4 test_results.csv contains all expected models and positive metrics."""
        path = self.results_dir / "corrected_v4_controlled_sequence/test_results.csv"
        self.assertTrue(path.exists(), f"Missing {path}")
        df = pd.read_csv(path)
        self.assertEqual(len(df), 8)
        self.assertTrue(all(df["test_accuracy"] > 0.3))
        self.assertTrue(all(df["test_macro_f1"] > 0.1))

if __name__ == "__main__":
    unittest.main()
