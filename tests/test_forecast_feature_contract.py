import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ML_Implementation"))
from forecast_feature_contract import (APPROVED, GROUPS, TARGET, build_forecast_matrix,
                                       validate_lineage_report, validate_matrix)


class ForecastFeatureContractTests(unittest.TestCase):
    def frame(self, year, values, label=None):
        d = {"YEAR": year, "STATE": " S ", "UNIT_NAME": " A  DISTRICT "}
        d.update(values)
        if label is not None:
            d[TARGET] = label
        return d

    def test_forecast_matrix_uses_only_t_minus_one_and_approved_features(self):
        values = {name: float(i + 10) for i, name in enumerate(GROUPS)}
        source = pd.DataFrame([self.frame(2000, values), self.frame(2001, {k: v + 1000 for k, v in values.items()})])
        targets = pd.DataFrame([self.frame(2001, {}, 1), self.frame(2002, {}, 2)])
        matrix, columns = build_forecast_matrix(targets, source, APPROVED["baseline"], "baseline")
        validate_matrix(matrix, columns, "baseline")
        row = matrix.loc[matrix.YEAR.eq(2001)].iloc[0]
        self.assertEqual(row.SOURCE_YEAR, 2000)
        self.assertEqual(row.LAG1_GROUP_HOMICIDE, values[GROUPS[0]])
        self.assertNotIn(TARGET, columns)
        self.assertNotIn("VIOLENT_CRIME_BURDEN", columns)

    def test_unapproved_or_target_feature_is_rejected(self):
        source = pd.DataFrame([self.frame(2000, {**{g: 1 for g in GROUPS}, TARGET: 2})])
        targets = pd.DataFrame([self.frame(2001, {}, 1)])
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            build_forecast_matrix(targets, source, GROUPS + [TARGET], "baseline")

    def test_duplicate_source_district_year_is_rejected(self):
        row = self.frame(2000, {g: 1 for g in GROUPS})
        source = pd.DataFrame([row, row])
        targets = pd.DataFrame([self.frame(2001, {}, 1)])
        with self.assertRaisesRegex(ValueError, "duplicate normalized"):
            build_forecast_matrix(targets, source, GROUPS, "baseline")

    def test_cross_district_prior_record_cannot_fill_forecast_row(self):
        values = {name: float(i + 1) for i, name in enumerate(GROUPS)}
        source = pd.DataFrame([{"YEAR": 2000, "STATE": "S", "UNIT_NAME": "OTHER", **values}])
        targets = pd.DataFrame([{"YEAR": 2001, "STATE": "S", "UNIT_NAME": "A",
                                 TARGET: 1}])
        matrix, _ = build_forecast_matrix(targets, source, GROUPS, "baseline")
        self.assertTrue(matrix.empty)

    def test_wrong_year_shift_is_rejected(self):
        values = {name: float(i + 10) for i, name in enumerate(GROUPS)}
        source = pd.DataFrame([self.frame(2000, values)])
        targets = pd.DataFrame([self.frame(2001, {}, 1)])
        matrix, columns = build_forecast_matrix(targets, source, GROUPS, "baseline")
        matrix.loc[:, "SOURCE_YEAR"] = matrix.YEAR  # controlled same-year violation
        with self.assertRaisesRegex(AssertionError, "minus one"):
            validate_matrix(matrix, columns, "baseline")

    def test_every_approved_predictor_requires_documented_source_lineage(self):
        rows = []
        expected = {**{f"corrected_tabular_{v}": f for v, f in APPROVED.items()},
                    **{f"corrected_sequence_{v}": f for v, f in APPROVED.items()},
                    "retired_temporal_leakage_free_runner": GROUPS + ["TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX", "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"],
                    "retired_combined_temporal_sequence_runner": GROUPS + ["TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX", "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"],
                    "retired_temporal_deep_learning_sequence_runner": GROUPS + ["TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX", "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"],
                    "notebook_observed_year_classification": GROUPS + ["PREV_YEAR_TOTAL_CRIME"],
                    "notebook_lstm_sequence_code": GROUPS + ["PREV_YEAR_TOTAL_CRIME"],
                    "same_year_feature_candidates_excluded": GROUPS + ["TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX", "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"]}
        for experiment, features in expected.items():
            for feature in features:
                rows.append({"experiment": experiment, "predictor": feature,
                             "original_source_columns": "IPC_SOURCE", "aggregation_or_transformation": "formula",
                             "source_observation_year": "s=t-1", "target_year": "t",
                             "available_by_forecast_origin": "timing caveat", "decision": "retained", "evidence": "test"})
        validate_lineage_report(pd.DataFrame(rows))
        incomplete = pd.DataFrame(rows[:-1])
        with self.assertRaisesRegex(AssertionError, "differs from approved"):
            validate_lineage_report(incomplete)

    def test_lineage_rejects_same_year_forecasting_source(self):
        rows = []
        for variant, features in APPROVED.items():
            for feature in features:
                rows.append({"experiment": f"corrected_tabular_{variant}", "predictor": feature,
                             "original_source_columns": "IPC_SOURCE", "aggregation_or_transformation": "formula",
                             "source_observation_year": "source at target year t", "target_year": "t",
                             "available_by_forecast_origin": "unknown", "decision": "retained", "evidence": "test"})
        for experiment, features in (("corrected_sequence_baseline", GROUPS),
                                     ("corrected_sequence_enhanced", GROUPS + ["TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX", "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"]),
                                     ("retired_temporal_leakage_free_runner", GROUPS + ["TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX", "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"]),
                                     ("retired_combined_temporal_sequence_runner", GROUPS + ["TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX", "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"]),
                                     ("retired_temporal_deep_learning_sequence_runner", GROUPS + ["TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX", "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"]),
                                     ("notebook_observed_year_classification", GROUPS + ["PREV_YEAR_TOTAL_CRIME"]),
                                     ("notebook_lstm_sequence_code", GROUPS + ["PREV_YEAR_TOTAL_CRIME"]),
                                     ("same_year_feature_candidates_excluded", GROUPS + ["TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX", "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"])):
            for feature in features:
                rows.append({"experiment": experiment, "predictor": feature,
                             "original_source_columns": "IPC_SOURCE", "aggregation_or_transformation": "formula",
                             "source_observation_year": "t-3,t-2,t-1", "target_year": "t",
                             "available_by_forecast_origin": "timing caveat", "decision": "retained", "evidence": "test"})
        with self.assertRaisesRegex(AssertionError, "same-year"):
            validate_lineage_report(pd.DataFrame(rows))


if __name__ == "__main__":
    unittest.main()
