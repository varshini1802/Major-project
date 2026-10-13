import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evidence_closure_audit import (
    KNOWN_DIRECT,
    KEYS,
    OUT,
    ROOT,
    calculate_overlap_reconciliation,
    classify_availability,
    current_protected_hashes,
    overlap_matrix,
)


class FinalEvidenceClosureTests(unittest.TestCase):
    def test_all_known_direct_sources_are_audited(self):
        matrix = overlap_matrix()
        observed = set(matrix.loc[matrix.overlap_type.eq("DIRECT_SHARED_COMPONENT"), "source_column"])
        self.assertEqual(observed, KNOWN_DIRECT)
        self.assertEqual(matrix.loc[matrix.overlap_type.eq("DIRECT_SHARED_COMPONENT"), "source_column"].nunique(), 9)

    def test_possible_relationships_are_not_called_independent(self):
        matrix = overlap_matrix()
        non_direct = matrix[~matrix.direct_overlap]
        self.assertNotIn("NO_IDENTIFIED_OVERLAP", set(non_direct.overlap_type))
        self.assertTrue(non_direct.resolution_status.str.contains("UNRESOLVED").any())

    def test_availability_unknown_and_post_origin_are_strict(self):
        self.assertEqual(classify_availability(None, None), "SCENARIO_UNDEFINED")
        self.assertEqual(classify_availability(None, "2015-01-01"), "UNKNOWN")
        self.assertEqual(classify_availability("2015-02-01", "2015-01-01"), "UNAVAILABLE_AFTER_ORIGIN")
        self.assertEqual(classify_availability("2014-12-31", "2015-01-01"), "AVAILABLE_BY_ORIGIN")

    def test_denominator_eligibility_and_real_values_reproduce(self):
        data = pd.read_csv(ROOT / "data/processed/district_targeted.csv", low_memory=False)
        sources = sorted(KNOWN_DIRECT)
        metrics = calculate_overlap_reconciliation(data, sources)
        self.assertEqual(metrics["valid_rows"], 9631)
        self.assertEqual(metrics["eligible_rows"], 9552)
        self.assertEqual(metrics["valid_rows_without_overlap_values"], 0)
        self.assertEqual(metrics["observed_overlap_rows"], 9631)
        self.assertEqual(metrics["zero_burden_valid_rows"], 79)
        self.assertEqual(metrics["zero_burden_direct_sum"], 0)
        self.assertEqual(metrics["coverage"], 1.0)
        self.assertTrue(metrics["all_valid_components_observed"])
        self.assertAlmostEqual(metrics["numerator"], 3228843.0)
        self.assertAlmostEqual(metrics["denominator"], 4215620.0)
        self.assertAlmostEqual(metrics["ratio"], 0.7659236363808882)
        shuffled = data.sample(frac=1, random_state=713).reset_index(drop=True)
        shuffled_metrics = calculate_overlap_reconciliation(shuffled, sources)
        self.assertEqual(shuffled_metrics["eligible_rows"], metrics["eligible_rows"])
        self.assertAlmostEqual(shuffled_metrics["numerator"], metrics["numerator"])
        self.assertAlmostEqual(shuffled_metrics["denominator"], metrics["denominator"])

    def test_missing_sources_are_not_zero_and_zero_denominator_is_safe(self):
        frame = pd.DataFrame({"TARGET_VIOLENT_LEVEL": [1, 1, 1], "VIOLENT_CRIME_BURDEN": [0, 5, 4],
                              "A": [0, np.nan, np.nan], "B": [0, np.nan, 0]})
        result = calculate_overlap_reconciliation(frame, ["A", "B"])
        self.assertEqual(result["valid_rows_without_overlap_values"], 1)
        self.assertEqual(result["eligible_rows"], 1)
        self.assertEqual(result["observed_overlap_rows"], 2)
        self.assertEqual(result["numerator"], 0)
        self.assertEqual(result["denominator"], 4)
        self.assertFalse(np.isnan(result["ratio"]))
        all_missing = frame.iloc[[1]].copy()
        empty_result = calculate_overlap_reconciliation(all_missing, ["A", "B"])
        self.assertTrue(np.isnan(empty_result["ratio"]))
        self.assertTrue(np.isnan(empty_result["all_valid_ratio"]))
        zero_only = pd.DataFrame({"TARGET_VIOLENT_LEVEL": [1], "VIOLENT_CRIME_BURDEN": [0], "A": [0], "B": [0]})
        zero_denominator = calculate_overlap_reconciliation(zero_only, ["A", "B"])
        self.assertTrue(np.isnan(zero_denominator["ratio"]))
        self.assertTrue(np.isnan(zero_denominator["all_valid_ratio"]))

    def test_duplicate_groups_are_preserved_and_conflicts_not_collapsed(self):
        duplicates = pd.read_csv(OUT / "duplicate_key_resolution_audit.csv")
        self.assertEqual(len(duplicates), 7)
        self.assertEqual(int(duplicates.copy_count.sum()), 14)
        conflicts = duplicates[duplicates.classification.eq("CONFLICTING_OFFENCE_PAYLOAD")]
        self.assertEqual(len(conflicts), 2)
        self.assertTrue(conflicts.analytical_treatment.str.contains("excluded").all())

    def test_missing_values_and_all_missing_sum_semantics(self):
        values = pd.DataFrame({"x": [0.0, np.nan, np.nan], "y": [np.nan, 3.0, np.nan]})
        sums = values.sum(axis=1, min_count=1)
        self.assertEqual(sums.iloc[0], 0.0)
        self.assertEqual(sums.iloc[1], 3.0)
        self.assertTrue(pd.isna(sums.iloc[2]))

    def test_six_group_formula_paths_match_saved_group_values(self):
        formulas = pd.read_csv(ROOT / "outputs/audits/final_corrections_v5/group_predictor_formula_recovery.csv")
        canonical = pd.read_csv(ROOT / "data/processed/district_integrated_canonical.csv", low_memory=False)
        grouped = pd.read_csv(ROOT / "data/processed/district_grouped_reduced.csv", low_memory=False)
        canonical_i = canonical.set_index(KEYS)
        stored_i = grouped.set_index(KEYS)
        for row in formulas.to_dict("records"):
            columns = [c.strip() for c in row["source_columns"].split("|")]
            expected = canonical_i[columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
            actual = pd.to_numeric(stored_i[row["predictor_group"]], errors="coerce")
            paired = expected.rename("expected").to_frame().join(actual.rename("stored").to_frame(), how="outer", validate="one_to_one")
            self.assertEqual(len(paired), len(canonical_i))
            self.assertTrue(np.allclose(paired.expected.to_numpy(), paired.stored.to_numpy(), equal_nan=True), row["predictor_group"])
            self.assertEqual(row["aggregation"], "row-wise arithmetic sum")
            self.assertIn("min_count=1", row["missing_handling"])

    def test_missingness_audit_keeps_schema_gaps_explicit(self):
        audit = pd.read_csv(OUT / "missingness_semantics_audit.csv")
        self.assertIn("UNKNOWN_CAUSE", set(audit.missing_reason.astype(str)))
        self.assertIn("UNMAPPED_CATEGORY_OR_SCHEMA_MISMATCH", set(audit.missing_reason.astype(str)))
        self.assertIn("DUPLICATE_DISTRICT_YEAR_EXCLUDED", set(audit.missing_reason.astype(str)))

    def test_thresholds_split_and_locked_671_key_artifact(self):
        thresholds = pd.read_csv(ROOT / "outputs/tables/target_thresholds.csv")
        self.assertEqual(thresholds.upper_cut_point.astype(float).tolist(), [107, 237, 362, 563])
        config = json.loads((ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/run_config.json").read_text())
        self.assertEqual(config["train_years"], list(range(2001, 2011)))
        self.assertEqual(config["validation_years"], [2011, 2012])
        self.assertEqual(config["locked_test_years"], [2013, 2014])
        keys = pd.read_csv(ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/locked_test_keys.csv")
        self.assertEqual(len(keys), 671)

    def test_protected_data_target_backup_and_saved_result_hashes_unchanged(self):
        snapshot = json.loads((OUT / "protected_input_hashes_before.json").read_text())
        self.assertEqual(snapshot, current_protected_hashes())


if __name__ == "__main__":
    unittest.main()
