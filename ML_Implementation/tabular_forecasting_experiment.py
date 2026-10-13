"""Corrected future-year tabular forecasting run (versioned, test locked)."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v2/tabular"
KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TARGET = "TARGET_VIOLENT_LEVEL"
TRAIN = list(range(2001, 2011))
VAL = [2011, 2012]
TEST = [2013, 2014]
GROUPS = ["GROUP_HOMICIDE", "GROUP_SEXUAL_VIOLENCE", "GROUP_KIDNAPPING_ABDUCTION",
          "GROUP_ROBBERY_DACOITY", "GROUP_PUBLIC_VIOLENCE", "GROUP_PROPERTY_CRIME"]
SEED = 42


def load(path):
    d = pd.read_csv(path)
    d["YEAR"] = pd.to_numeric(d.YEAR, errors="raise").astype(int)
    for c in KEYS[1:]:
        d[c] = d[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    if d.duplicated(KEYS).any():
        raise ValueError(f"Duplicate keys in {path}")
    return d


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite corrected run: {OUT}")
    OUT.mkdir(parents=True)
    target = load(ROOT / "data/processed/district_targeted.csv")
    if target[TARGET].isna().any():
        target = target.dropna(subset=[TARGET])
    target[TARGET] = target[TARGET].astype(int)
    thresholds = pd.read_csv(ROOT / "outputs/tables/target_thresholds.csv").sort_values("boundary_index")
    cuts = thresholds.upper_cut_point.astype(float).tolist()
    if cuts != [107., 237., 362., 563.] or thresholds.fit_years.astype(str).unique().tolist() != ["2001-2010"]:
        raise ValueError("Frozen training-only target boundary artifact mismatch")

    variants = {
        "baseline": (load(ROOT / "data/processed/district_grouped_reduced.csv"), GROUPS),
        "enhanced": (load(ROOT / "data/processed/district_enhanced_features.csv"), None),
    }
    summaries, distributions = [], []
    for name, (source, features) in variants.items():
        if features is None:
            extras = [c for c in source if c not in KEYS + [TARGET] + GROUPS
                      and pd.api.types.is_numeric_dtype(source[c])]
            features = GROUPS + extras
        # Bind each target year t only to feature row t-1 by normalized district keys.
        hist = source[KEYS + features].copy()
        hist["YEAR"] += 1
        hist = hist.rename(columns={c: f"LAG1_{c}" for c in features})
        data = target[KEYS + [TARGET]].merge(hist, on=KEYS, how="left", validate="one_to_one")
        xcols = [f"LAG1_{c}" for c in features]
        data = data[data.YEAR.isin(TRAIN + VAL + TEST)].dropna(subset=xcols, how="all").copy()
        # Do not retain rows with no observed historical predictor; remaining feature-level nulls
        # are handled by an imputer fitted only on permitted fit years.
        tr, va, te = (data[data.YEAR.isin(ys)].copy() for ys in (TRAIN, VAL, TEST))
        if min(len(tr), len(va), len(te)) == 0:
            raise ValueError(f"Empty chronological split for {name}")
        for part, years in (("train", TRAIN), ("validation", VAL), ("test", TEST)):
            sub = data[data.YEAR.isin(years)]
            for cls in range(1, 6):
                distributions.append({"dataset": name, "partition": part, "class": cls,
                                      "count": int((sub[TARGET] == cls).sum()), "rows": len(sub)})
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
        fitted = []
        for model_name, factory in factories.items():
            model = factory()
            model.fit(tr[xcols], tr[TARGET])
            vp = model.predict(va[xcols])
            fitted.append((model_name, factory, f1_score(va[TARGET], vp, average="macro", zero_division=0)))
        # Freeze selection using validation macro-F1. Test remains unread until selected model is refit.
        fitted.sort(key=lambda x: (-x[2], x[0]))
        selected = fitted[0]
        model_name, factory, val_f1 = selected
        fit = pd.concat([tr, va], ignore_index=True)
        model = factory()
        model.fit(fit[xcols], fit[TARGET])
        yp = model.predict(te[xcols])
        report = classification_report(te[TARGET], yp, labels=[1, 2, 3, 4, 5], output_dict=True, zero_division=0)
        cm = confusion_matrix(te[TARGET], yp, labels=[1, 2, 3, 4, 5])
        model_dir = OUT / name
        model_dir.mkdir()
        pd.DataFrame(report).T.to_csv(model_dir / "test_classification_report.csv")
        pd.DataFrame(cm, index=[f"actual_{i}" for i in range(1, 6)],
                     columns=[f"predicted_{i}" for i in range(1, 6)]).to_csv(model_dir / "test_confusion_matrix.csv")
        pd.DataFrame({"YEAR": te.YEAR, "STATE": te.STATE, "UNIT_NAME": te.UNIT_NAME,
                      "actual": te[TARGET], "predicted": yp}).to_csv(model_dir / "test_predictions.csv", index=False)
        summaries.append({"dataset": name, "selected_model": model_name,
                          "validation_macro_f1": val_f1, "train_rows": len(tr), "validation_rows": len(va),
                          "test_rows": len(te), "test_accuracy": accuracy_score(te[TARGET], yp),
                          "test_macro_f1": f1_score(te[TARGET], yp, average="macro", zero_division=0),
                          "test_weighted_f1": f1_score(te[TARGET], yp, average="weighted", zero_division=0),
                          "feature_columns": "|".join(xcols)})
        # Report a prespecified most-frequent-class baseline on the same locked test.
        base_factory = factories["Most frequent baseline"]
        base_model = base_factory().fit(fit[xcols], fit[TARGET])
        base_pred = base_model.predict(te[xcols])
        base_dir = OUT / name / "most_frequent_baseline"
        base_dir.mkdir(parents=True)
        pd.DataFrame(classification_report(te[TARGET], base_pred, labels=[1, 2, 3, 4, 5],
                                           output_dict=True, zero_division=0)).T.to_csv(base_dir / "test_classification_report.csv")
        pd.DataFrame(confusion_matrix(te[TARGET], base_pred, labels=[1, 2, 3, 4, 5]),
                     index=[f"actual_{i}" for i in range(1, 6)],
                     columns=[f"predicted_{i}" for i in range(1, 6)]).to_csv(base_dir / "test_confusion_matrix.csv")
        pd.DataFrame({"YEAR": te.YEAR, "STATE": te.STATE, "UNIT_NAME": te.UNIT_NAME,
                      "actual": te[TARGET], "predicted": base_pred}).to_csv(base_dir / "test_predictions.csv", index=False)
        base_score = next(score for n, _, score in fitted if n == "Most frequent baseline")
        summaries.append({"dataset": name, "selected_model": "Most frequent baseline",
                          "validation_macro_f1": base_score, "train_rows": len(tr), "validation_rows": len(va),
                          "test_rows": len(te), "test_accuracy": accuracy_score(te[TARGET], base_pred),
                          "test_macro_f1": f1_score(te[TARGET], base_pred, average="macro", zero_division=0),
                          "test_weighted_f1": f1_score(te[TARGET], base_pred, average="weighted", zero_division=0),
                          "feature_columns": "|".join(xcols)})
        # Validation selection table contains no test scores.
        pd.DataFrame([{"model": n, "validation_macro_f1": s} for n, _, s in fitted]).to_csv(
            model_dir / "validation_model_selection.csv", index=False)

    pd.DataFrame(summaries).to_csv(OUT / "test_results.csv", index=False)
    pd.DataFrame(distributions).to_csv(OUT / "class_distributions.csv", index=False)
    config = {"objective": "forecast future-year violent-crime burden", "target": TARGET,
              "thresholds": cuts, "threshold_fit_years": "2001-2010", "train_years": TRAIN,
              "validation_years": VAL, "locked_test_years": TEST, "features": "district keyed t-1 rows",
              "selection": "validation macro-F1; refit selected model on train+validation; evaluate locked test once",
              "preprocessing": "median imputation and scaling are within sklearn pipeline fitted on train, then refitted on train+validation after selection",
              "seed": SEED}
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
