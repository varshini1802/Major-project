"""Validate and summarize saved v5 five-feature, five-seed ablations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v5_feature_ablation_multiseed"
V4 = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence"
V5 = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v5_multiseed"
EXPECTED_SEEDS = {11, 23, 37, 53, 71}
EXPECTED_FEATURES = {"TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX",
                     "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"}
EXPECTED_MODELS = {"GRU", "LSTM"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    status = pd.read_csv(OUT / "run_status.csv")
    metrics = pd.read_csv(OUT / "per_seed_ablation_metrics.csv")
    expected = {(s, f, m) for s in EXPECTED_SEEDS for f in EXPECTED_FEATURES for m in EXPECTED_MODELS}
    actual = set(map(tuple, metrics[["seed", "feature_added", "model"]].itertuples(index=False, name=None)))
    if actual != expected or len(metrics) != 50:
        missing = sorted(expected - actual); extra = sorted(actual - expected)
        raise AssertionError(f"Expected 50 distinct ablation rows; got {len(metrics)}; missing={missing}; extra={extra}")
    if not status.status.eq("COMPLETE").all() or len(status) != 50:
        raise AssertionError(f"Ablation status contains incomplete runs: {status.status.value_counts().to_dict()}")
    locked = pd.read_csv(V4 / "locked_test_keys.csv")
    checked = []
    for row in metrics.itertuples(index=False):
        prediction = OUT / f"seed_{row.seed}" / row.feature_added.lower() / row.model.lower() / "test_predictions.csv"
        frame = pd.read_csv(prediction)
        pd.testing.assert_frame_equal(frame[locked.columns], locked, check_dtype=False)
        if len(frame) != 671:
            raise AssertionError(f"Unexpected test rows: {prediction}")
        v5metrics = pd.read_csv(V5 / f"seed_{row.seed}" / "test_results.csv")
        baseline = v5metrics.loc[(v5metrics.variant == "baseline") & (v5metrics.model == row.model), "test_macro_f1"]
        if len(baseline) != 1 or not np.isclose(float(baseline.iloc[0]), row.baseline_macro_f1_same_seed, atol=1e-12):
            raise AssertionError(f"Paired baseline mismatch for seed={row.seed}, model={row.model}")
        checked.append({"seed": row.seed, "feature_added": row.feature_added, "model": row.model,
                        "test_rows": len(frame), "test_keys_match_locked_v4": True,
                        "baseline_matches_same_seed_v5": True, "test_used_for_selection": False})
    pd.DataFrame(checked).to_csv(OUT / "integrity_checks.csv", index=False)
    summary = metrics.groupby(["feature_added", "model"]).paired_macro_f1_difference.agg(
        seeds="count", mean_paired_difference="mean", sd_across_seeds="std").reset_index()
    direction = metrics.assign(direction=np.sign(metrics.paired_macro_f1_difference)).groupby(
        ["feature_added", "model"]).direction.agg(
            higher=lambda x: int((x > 0).sum()), equal=lambda x: int((x == 0).sum()), lower=lambda x: int((x < 0).sum())).reset_index()
    summary = summary.merge(direction, on=["feature_added", "model"], validate="one_to_one")
    summary["interpretation_limit"] = "fixed-test seed variability only; not sampling uncertainty, statistical significance, or causal feature importance"
    summary.to_csv(OUT / "paired_difference_summary.csv", index=False)
    cfg = json.loads((OUT / "run_config.json").read_text(encoding="utf-8"))
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(OUT.glob("seed_*/**/test_predictions.csv"))}
    (OUT / "input_output_hashes.json").write_text(json.dumps({"locked_test_keys_sha256": sha256(V4 / "locked_test_keys.csv"),
        "prediction_file_sha256": hashes, "runner_config": cfg}, indent=2) + "\n", encoding="utf-8")
    lines = ["# Five-seed one-feature ablation report", "",
             "All 50 prespecified feature/model/seed runs completed. Each row adds exactly one engineered feature to the six grouped predictors, and is paired against the same-seed, same-architecture baseline. Validation macro-F1 selects epoch; test keys are identical to the locked v4 cohort (671 rows).", "",
             "Positive paired differences favor the one-feature input relative to its paired six-feature baseline. Mean and SD describe seed variability on this fixed cohort; they are not inferential intervals or causal effects.", "",
             "| Feature added | Model | Mean paired macro-F1 | SD across seeds | Higher / equal / lower seeds |", "|---|---|---:|---:|---:|"]
    for r in summary.itertuples(index=False):
        lines.append(f"| {r.feature_added} | {r.model} | {r.mean_paired_difference:+.5f} | {r.sd_across_seeds:.5f} | {r.higher}/{r.equal}/{r.lower} |")
    lines += ["", "Saved exact per-seed metrics, validation selections, test predictions, model files, integrity checks, run config, run status and SHA-256 manifest are in this directory.",
              "No p-values or confidence intervals are provided. These are one-feature-at-a-time descriptive comparisons, not causal feature importance.", ""]
    (OUT / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print("PASS: all 50 runs and locked test cohorts verified")


if __name__ == "__main__":
    main()
