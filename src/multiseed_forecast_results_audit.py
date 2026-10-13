
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audits/final_corrections_v5"
RUNS = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v5_multiseed"
V4 = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence"
SEEDS = [11, 23, 37, 53, 71]
KEYS = ["YEAR", "STATE", "UNIT_NAME"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm(frame):
    d = frame.copy()
    d.YEAR = pd.to_numeric(d.YEAR, errors="raise").astype(int)
    for c in KEYS[1:]:
        d[c] = d[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    return d


def main():
    status = pd.read_csv(RUNS / "run_status.csv")
    v4keys = norm(pd.read_csv(V4 / "locked_test_keys.csv"))[KEYS + ["TARGET_VIOLENT_LEVEL", "INPUT_YEARS"]]
    checks, metrics = [], []
    for seed in SEEDS:
        status_rows = status[(status.seed == seed) & status.status.eq("COMPLETE")]
        if len(status_rows) != 1:
            checks.append({"seed": seed, "check": "seed_complete", "result": "FAIL",
                           "detail": f"completed status records={len(status_rows)}"})
            continue
        folder = ROOT / status_rows.iloc[0].output_directory
        config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
        cfg_ok = (config["seed"] == seed and config["thresholds"] == [107, 237, 362, 563]
                  and config["train_years"] == list(range(2001, 2011))
                  and config["validation_years"] == [2011, 2012]
                  and config["locked_test_years"] == [2013, 2014]
                  and config["test_used_for_selection"] is False)
        checks.append({"seed": seed, "check": "frozen_config_and_test_lock", "result": "PASS" if cfg_ok else "FAIL",
                       "detail": json.dumps({"seed": config.get("seed"), "thresholds": config.get("thresholds"),
                                             "train": config.get("train_years"), "validation": config.get("validation_years"),
                                             "test": config.get("locked_test_years"),
                                             "test_used_for_selection": config.get("test_used_for_selection")})})
        keys_ok = True
        for variant in ("baseline", "enhanced"):
            for model in ("gru", "lstm", "logistic_regression", "majority_baseline"):
                pred_path = folder / variant / model / "test_predictions.csv"
                prob_path = folder / variant / model / "test_probabilities.csv"
                pred = norm(pd.read_csv(pred_path))
                actual_keys = pred[KEYS + ["TARGET_VIOLENT_LEVEL", "INPUT_YEARS"]].sort_values(KEYS).reset_index(drop=True)
                if not actual_keys.equals(v4keys.sort_values(KEYS).reset_index(drop=True)):
                    keys_ok = False
                probs = pd.read_csv(prob_path)
                if len(probs) != 671 or len(actual_keys) != 671 or not np.isfinite(
                        probs[[f"PROB_CLASS_{c}" for c in range(1, 6)]].to_numpy(dtype=float)).all():
                    keys_ok = False
                p_matrix = probs[[f"PROB_CLASS_{c}" for c in range(1, 6)]].to_numpy(dtype=float)
                if np.max(np.abs(p_matrix.sum(axis=1) - 1.0)) > 1e-5:
                    keys_ok = False
                if model != "majority_baseline":
                    labels = pred.TARGET_VIOLENT_LEVEL.to_numpy(dtype=int)
                    hard = pred.PREDICTED_TARGET.to_numpy(dtype=int)
                    p = probs[[f"PROB_CLASS_{c}" for c in range(1, 6)]].to_numpy(dtype=float)
                    if not np.array_equal(hard, np.argmax(p, axis=1) + 1):
                        raise AssertionError(f"Seed {seed} {variant}/{model}: prediction/probability class mismatch")
                result = pd.read_csv(folder / "test_results.csv")
                row = result[(result.variant == variant) & (result.model.str.lower() == model.replace("_", ""))]
                if model == "logistic_regression":
                    row = result[(result.variant == variant) & result.model.eq("LogisticRegression")]
                elif model == "majority_baseline":
                    row = result[(result.variant == variant) & result.model.eq("Majority baseline")]
                if len(row) != 1 or int(row.iloc[0].test_rows) != 671:
                    raise AssertionError(f"Seed {seed} {variant}/{model}: metric row/test size invalid")
        checks.append({"seed": seed, "check": "all_model_test_keys_labels_histories_match_v4",
                       "result": "PASS" if keys_ok else "FAIL", "detail": "8 model outputs checked against frozen 671 keys/labels/input-years"})
        result = pd.read_csv(folder / "test_results.csv")
        for model in ("GRU", "LSTM", "LogisticRegression"):
            pair = result[result.model.eq(model)].set_index("variant")
            for metric in ("test_accuracy", "test_macro_f1", "test_weighted_f1", "macro_ovr_roc_auc", "macro_ovr_average_precision"):
                metrics.append({"seed": seed, "model": model, "metric": metric,
                                "baseline": pair.loc["baseline", metric], "enhanced": pair.loc["enhanced", metric],
                                "enhanced_minus_baseline": pair.loc["enhanced", metric] - pair.loc["baseline", metric]})
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(OUT / "multiseed_integrity_checks.csv", index=False)
    if not check_frame.result.eq("PASS").all():
        raise AssertionError("One or more multi-seed integrity checks failed")
    individual = pd.DataFrame(metrics)
    individual.to_csv(OUT / "multiseed_per_seed_metrics.csv", index=False)
    value_rows = []
    for row in individual.itertuples(index=False):
        for variant in ("baseline", "enhanced"):
            value_rows.append({"seed": row.seed, "model": row.model, "variant": variant,
                               "metric": row.metric, "value": getattr(row, variant)})
    long = pd.DataFrame(value_rows)
    per_variant = long.groupby(["model", "variant", "metric"]).value.agg(
        seeds="count", mean="mean", standard_deviation="std").reset_index()
    paired = individual.groupby(["model", "metric"]).enhanced_minus_baseline.agg(
        seeds="count", mean_paired_difference="mean", standard_deviation="std").reset_index()
    def signs(s):
        return f"higher={int((s>0).sum())}; equal={int((s==0).sum())}; lower={int((s<0).sum())}"
    sign_counts = individual.groupby(["model", "metric"]).enhanced_minus_baseline.apply(signs).rename("direction_counts").reset_index()
    paired = paired.merge(sign_counts, on=["model", "metric"], validate="one_to_one")
    paired["interval"] = "Not estimated: seed variability is not sampling uncertainty for the fixed dependent district-year test cohort."
    per_variant.to_csv(OUT / "multiseed_metric_summary.csv", index=False)
    paired.to_csv(OUT / "multiseed_paired_difference_summary.csv", index=False)

    details = ["# Corrected v5 multi-seed robustness results", "",
               "Five predeclared seeds (11, 23, 37, 53, 71) completed. Each run reused the corrected-v4 training procedure, frozen thresholds, 2001–2010 train / 2011–2012 validation / 2013–2014 test split, and identical locked 671-key test cohort. Each paired GRU/LSTM baseline and enhanced model used validation macro-F1 for epoch selection; logistic C selection was also validation-only. No test result selected a model.", "",
               "## Paired differences by seed", "",
               "Positive values favor enhanced features. These are fixed-test descriptive differences, not causal effects.", "",
               "```text", individual.to_string(index=False, float_format=lambda x: f"{x:.6f}"), "```", "",
               "## Mean and standard deviation of paired differences", "",
               "```text", paired.to_string(index=False, float_format=lambda x: f"{x:.6f}"), "```", "",
               "## Stability interpretation", "",
               "GRU enhanced macro-F1 was higher in 4/5 seeds (mean paired difference +0.00785, seed SD 0.00772), while GRU macro ROC-AUC was lower in 4/5 (mean −0.00099) and AP direction was mixed (3/5 higher). LSTM enhanced macro-F1 was higher in all five seeds (mean +0.01248, seed SD 0.00768), and accuracy and weighted-F1 were higher in all five. LSTM ROC-AUC direction was mixed (3/5 higher), and AP was higher in 3/5. This supports seed-stable LSTM macro-F1 improvement on this locked cohort but does not establish a universal gain across model families or metrics. Logistic regression differences are exactly repeatable here because its fixed procedure is deterministic for this data/configuration.", "",
               "No confidence interval or p-value is reported. Seeds vary initialization/training randomness; they do not resample districts or years. Test rows from a district and adjacent years can be dependent, so ordinary row-wise intervals would not address that dependence. District-cluster or temporal-block inference was not attempted.", "",
               "The preserved seed-42 v4 ablation remains separate. The one-feature-at-a-time analyses were also rerun in the distinct v5 five-seed ablation directory (50/50 fits); see its ablation report and integrity audit. Saved v4 outputs remain unchanged.", ""]
    (OUT / "multiseed_robustness_report.md").write_text("\n".join(details), encoding="utf-8")

    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "tensorflow"):
        packages[name] = importlib.metadata.version(name)
    inputs = [ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3/inputs/common_sequence_metadata.csv",
              ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3/inputs/baseline_common_sequences.npz",
              ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3/inputs/enhanced_common_sequences.npz",
              ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/run_config.json"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    manifest = {"environment_python": platform.python_version(), "packages": packages,
                "predeclared_seeds": SEEDS, "completed_seeds": sorted(individual.seed.unique().astype(int).tolist()),
                "input_hashes": hashes, "runner": "ML_Implementation/run_multiseed_forecasting_experiment.py",
                "model_source": "ML_Implementation/controlled_sequence_forecast_evaluation.py",
                "configuration": "each seed folder/run_config.json; thresholds 107/237/362/563; train 2001-2010; validation 2011-2012; test 2013-2014",
                "test_lock": "identical v4 671 district-year keys, target labels and input years checked for every model/seed",
                "selection_scope": "validation macro-F1 for NN epochs and logistic C; test used once for reporting",
                "uncertainty": "mean and standard deviation across random seeds reported; no inferential confidence interval because seed runs do not resample independent districts/years",
                "ablation_scope": "v4 seed-42 ablation preserved separately; v5 five-seed feature ablations are stored in corrected_v5_feature_ablation_multiseed"}
    (OUT / "multiseed_reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Integrity checks:")
    print(check_frame.to_string(index=False))
    print("Paired metric summary:")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
