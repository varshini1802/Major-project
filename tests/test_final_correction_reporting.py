import unittest

import numpy as np
import pandas as pd

from src.target_creation import CLASS_LABELS
from src.data_quality_pipeline_audit import missingness_rows, GROUPS
from ML_Implementation.controlled_sequence_forecast_evaluation import score_probabilities


class FinalCorrectionReportingTests(unittest.TestCase):
    def test_ordered_class_names_are_burden_levels(self):
        self.assertEqual(list(CLASS_LABELS), [1, 2, 3, 4, 5])
        self.assertEqual(CLASS_LABELS[1], "Lowest violent-crime burden")
        self.assertEqual(CLASS_LABELS[5], "Highest violent-crime burden")
        self.assertNotIn("Homicide", " ".join(CLASS_LABELS.values()))

    def test_missingness_audit_keeps_zero_distinct_from_missing(self):
        values = {name: [0.0, np.nan, 3.0] for name in GROUPS}
        rows = missingness_rows(pd.DataFrame(values), GROUPS, [])
        first = rows.iloc[0]
        self.assertEqual(first.observed_zero, 1)
        self.assertEqual(first.missing_or_unavailable, 1)
        self.assertEqual(first.observed_positive, 1)
        self.assertFalse(first.missing_reason_identifiable_from_available_columns)

    def test_multiclass_auc_requires_all_test_classes(self):
        y = np.array([1, 2, 3, 4, 5] * 2)
        p = np.eye(5)[y - 1]
        metrics = score_probabilities(y, p)
        self.assertEqual(metrics["macro_ovr_roc_auc"], 1.0)
        with self.assertRaises(ValueError):
            score_probabilities(np.array([1, 2, 3, 4]), np.eye(5)[:4])


if __name__ == "__main__":
    unittest.main()
