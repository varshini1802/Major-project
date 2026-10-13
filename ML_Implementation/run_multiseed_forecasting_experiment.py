"""Run a predeclared multi-seed robustness extension without touching v4.

Usage: python ML_Implementation/run_multiseed_forecasting_experiment.py
Default seeds are fixed before evaluation. Each seed writes to a new directory.
This runner reuses the exact v4 experiment procedure and common cohort.
"""
from __future__ import annotations

import importlib.util
import contextlib
import json
import sys
import traceback
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 23, 37, 53, 71)
OUT = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v5_multiseed"
SOURCE = ROOT / "ML_Implementation/controlled_sequence_forecast_evaluation.py"


def load_experiment():
    spec = importlib.util.spec_from_file_location("corrected_v4_runner_for_v5", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize() -> None:
    rows = []
    metric_rows = []
    for seed in SEEDS:
        run = OUT / f"seed_{seed}"
        record_path = OUT / "run_status.csv"
        if record_path.exists():
            status = pd.read_csv(record_path)
            success = status[(status.seed == seed) & status.status.eq("COMPLETE")]
            if not success.empty:
                run = ROOT / success.iloc[-1].output_directory
        path = run / "test_results.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for model in ("GRU", "LSTM", "LogisticRegression"):
            pair = frame[frame.model.eq(model)].set_index("variant")
            if set(pair.index) != {"baseline", "enhanced"}:
                raise AssertionError(f"Seed {seed}/{model}: expected paired variants")
            for metric in ("test_accuracy", "test_macro_f1", "test_weighted_f1",
                           "macro_ovr_roc_auc", "macro_ovr_average_precision"):
                for variant in ("baseline", "enhanced"):
                    metric_rows.append({"seed": seed, "model": model, "variant": variant,
                                        "metric": metric, "value": pair.loc[variant, metric]})
                rows.append({"seed": seed, "model": model, "metric": metric,
                             "baseline": pair.loc["baseline", metric],
                             "enhanced": pair.loc["enhanced", metric],
                             "enhanced_minus_baseline": pair.loc["enhanced", metric] - pair.loc["baseline", metric],
                             "paired_test_rows": int(pair.loc["baseline", "test_rows"]),
                             "same_test_rows": pair.loc["baseline", "test_rows"] == pair.loc["enhanced", "test_rows"]})
    individual = pd.DataFrame(rows)
    if individual.empty:
        return
    individual.to_csv(OUT / "per_seed_paired_metrics.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUT / "per_seed_model_metrics.csv", index=False)
    score_summary = pd.DataFrame(metric_rows).groupby(["model", "variant", "metric"]).value.agg(
        seeds="count", mean="mean", standard_deviation="std").reset_index()
    score_summary.to_csv(OUT / "model_metric_summary.csv", index=False)
    summary = individual.groupby(["model", "metric"]).enhanced_minus_baseline.agg(
        seeds="count", mean_paired_difference="mean", sd_across_seeds="std").reset_index()
    signs = individual.assign(direction=np.sign(individual.enhanced_minus_baseline)).groupby(
        ["model", "metric"]).direction.agg(lambda x: f"enhanced higher: {(x > 0).sum()}; tie: {(x == 0).sum()}; lower: {(x < 0).sum()}").reset_index(name="seed_direction_counts")
    summary = summary.merge(signs, on=["model", "metric"], validate="one_to_one")
    summary["uncertainty_interval"] = "not supplied: seed repetitions measure optimizer/random initialization variability; districts and adjacent years in each test remain dependent"
    summary.to_csv(OUT / "paired_difference_summary.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_experiment()
    status_rows = []
    status_path = OUT / "run_status.csv"
    if status_path.exists():
        status_rows = pd.read_csv(status_path).to_dict("records")
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    for seed in SEEDS:
        complete = [row for row in status_rows if row["seed"] == seed and row["status"] == "COMPLETE"]
        if complete and (ROOT / complete[-1]["output_directory"] / "test_results.csv").exists():
            print(f"Seed {seed} already complete; preserving {complete[-1]['output_directory']}", flush=True)
            continue
        attempt = 1 + sum(1 for row in status_rows if row["seed"] == seed)
        target = OUT / (f"seed_{seed}" if attempt == 1 else f"seed_{seed}_attempt_{attempt:02d}")
        module.SEED = seed
        module.OUT = target
        log_path = logs / f"seed_{seed}_attempt_{attempt:02d}.log"
        print(f"Starting predeclared seed {seed} ({attempt}/...); log: {log_path}", flush=True)
        try:
            with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                module.main()
            result_path = target / "test_results.csv"
            results = pd.read_csv(result_path)
            if len(results) != 8 or set(results.test_rows.astype(int)) != {671}:
                raise AssertionError("Expected eight model rows on 671 test cases")
            keys = pd.read_csv(target / "locked_test_keys.csv")
            v4_keys = pd.read_csv(ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/locked_test_keys.csv")
            pd.testing.assert_frame_equal(keys, v4_keys, check_dtype=False)
            env = {}
            for name in ("numpy", "pandas", "scikit-learn", "tensorflow"):
                try:
                    env[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    env[name] = "not installed"
            (target / "environment_versions.json").write_text(json.dumps({"python": sys.version,
                "packages": env, "seed": seed, "seed_controls": "Python random, NumPy, TensorFlow set_seed"}, indent=2) + "\n", encoding="utf-8")
            status_rows.append({"seed": seed, "attempt": attempt, "status": "COMPLETE",
                                "output_directory": str(target.relative_to(ROOT)), "log": str(log_path.relative_to(ROOT)),
                                "detail": "all paired models completed and locked test keys match v4"})
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as log:
                log.write("\nRUNNER FAILURE\n" + traceback.format_exc())
            status_rows.append({"seed": seed, "attempt": attempt, "status": "FAILED_OR_INCOMPLETE",
                                "output_directory": str(target.relative_to(ROOT)), "log": str(log_path.relative_to(ROOT)),
                                "detail": f"{type(exc).__name__}: {exc}"})
            print(f"Seed {seed} did not complete: {type(exc).__name__}: {exc}", flush=True)
        pd.DataFrame(status_rows).to_csv(status_path, index=False)
    summarize()
    statuses = pd.DataFrame(status_rows)
    config = {"status": "seed outcomes are recorded in run_status.csv; only complete runs enter metric summaries",
              "predeclared_seeds": list(SEEDS), "source_runner": str(SOURCE.relative_to(ROOT)),
              "procedure": "corrected-v4 runner; same strict common cohort, architecture pairs, chronological years, validation selection and locked test",
              "randomness_controls": "Python random, NumPy and TensorFlow seed via tf.keras.utils.set_random_seed; deterministic kernels/platform behavior is not guaranteed",
              "test_uncertainty": "no row bootstrap supplied; seeds do not create independent districts/years; district/temporal dependence remains",
              "ablation_repeats": "completed separately: five-seed one-feature/model ablation; see corrected_v5_feature_ablation_multiseed",
              "complete_seed_count": int(statuses.status.eq("COMPLETE").sum()) if not statuses.empty else 0,
              "failed_or_incomplete_seed_count": int(statuses.status.eq("FAILED_OR_INCOMPLETE").sum()) if not statuses.empty else 0}
    (OUT / "runner_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Multi-seed outputs: {OUT}")
    print(statuses.to_string(index=False) if not statuses.empty else "No seed runs recorded")


if __name__ == "__main__":
    main()
