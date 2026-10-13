"""Train/evaluate explicit-contract tabular forecasts from saved v3 matrices."""
from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ML_Implementation"))
from forecast_feature_contract import APPROVED, KEYS, TARGET, validate_matrix

BASE = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3"
IN = BASE / "inputs"
OUT = BASE / "tabular"
TRAIN = list(range(2001, 2011)); VAL = [2011, 2012]; TEST = [2013, 2014]
SEED = 42


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir()
    result_rows = []; distributions = []
    for variant in ("baseline", "enhanced"):
        matrix = pd.read_csv(IN / f"{variant}_forecast_matrix.csv")
        columns = [f"LAG1_{c}" for c in APPROVED[variant]]
        validate_matrix(matrix, columns, variant)
        tr, va, te = (matrix[matrix.YEAR.isin(ys)].copy() for ys in (TRAIN, VAL, TEST))
        if any(frame.empty for frame in (tr, va, te)):
            raise ValueError(f"{variant} has an empty temporal split")
        for split, frame in (("train", tr), ("validation", va), ("test", te)):
            for cls in range(1, 6):
                distributions.append({"variant": variant, "partition": split, "class": cls,
                                      "count": int(frame[TARGET].eq(cls).sum()), "rows": len(frame)})
        factories = {
            "Most frequent baseline": lambda: DummyClassifier(strategy="most_frequent"),
            "GaussianNB": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), GaussianNB()),
            "ExtraTrees": lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(
                n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)),
            "RandomForest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)),
            "MLP": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=SEED)),
        }
        selection = []
        for name, factory in factories.items():
            model = factory().fit(tr[columns], tr[TARGET])
            pred = model.predict(va[columns])
            selection.append((name, factory, f1_score(va[TARGET], pred, average="macro", zero_division=0)))
        selection.sort(key=lambda x: (-x[2], x[0]))
        pd.DataFrame([{"model": n, "validation_macro_f1": s} for n, _, s in selection]).to_csv(
            OUT / f"{variant}_validation_selection.csv", index=False)
        # Freeze validation selection before reading test labels or predictions.
        fit = pd.concat([tr, va], ignore_index=True)
        for name, factory, score in (selection[0], next(x for x in selection if x[0] == "Most frequent baseline")):
            model = factory().fit(fit[columns], fit[TARGET])
            pred = model.predict(te[columns])
            folder = OUT / variant / name.lower().replace(" ", "_")
            folder.mkdir(parents=True)
            joblib.dump(model, folder / "fitted_pipeline.joblib")
            pd.DataFrame(classification_report(te[TARGET], pred, labels=[1, 2, 3, 4, 5],
                                               output_dict=True, zero_division=0)).T.to_csv(folder / "test_classification_report.csv")
            pd.DataFrame(confusion_matrix(te[TARGET], pred, labels=[1, 2, 3, 4, 5]),
                         index=[f"actual_{i}" for i in range(1, 6)],
                         columns=[f"predicted_{i}" for i in range(1, 6)]).to_csv(folder / "test_confusion_matrix.csv")
            te[KEYS + ["SOURCE_YEAR", TARGET]].assign(PREDICTED_TARGET=pred).to_csv(folder / "test_predictions.csv", index=False)
            result_rows.append({"variant": variant, "model": name, "validation_macro_f1": score,
                                "train_rows": len(tr), "validation_rows": len(va), "test_rows": len(te),
                                "test_accuracy": accuracy_score(te[TARGET], pred),
                                "test_macro_f1": f1_score(te[TARGET], pred, average="macro", zero_division=0),
                                "test_weighted_f1": f1_score(te[TARGET], pred, average="weighted", zero_division=0),
                                "feature_columns": "|".join(columns)})
    pd.DataFrame(result_rows).to_csv(OUT / "test_results.csv", index=False)
    pd.DataFrame(distributions).to_csv(OUT / "class_distributions.csv", index=False)
    config = {"objective": "future-year violent-crime burden forecasting", "target": TARGET,
              "thresholds": [107, 237, 362, 563], "threshold_fit_years": "2001-2010",
              "train_years": TRAIN, "validation_years": VAL, "locked_test_years": TEST,
              "feature_contract": {k: [f"LAG1_{c}" for c in v] for k, v in APPROVED.items()},
              "selection": "validation macro-F1; test evaluated only for selected model and prespecified majority baseline",
              "preprocessing": "pipeline operations fit on training rows for selection; final pipelines fit on train+validation",
              "seed": SEED}
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(result_rows).to_string(index=False))


if __name__ == "__main__":
    main()
