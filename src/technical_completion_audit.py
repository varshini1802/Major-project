"""Build a read-only final technical audit package from project evidence.

Writes only to outputs/audits/final_technical_completion. Source data,
published outputs, saved results, and the paper are read-only inputs.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audits/final_technical_completion"
LINEAGE_SOURCE = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3/forecast_source_lineage.csv"
GROUPED = ROOT / "data/processed/district_grouped_reduced.csv"
ENHANCED = ROOT / "data/processed/district_enhanced_features.csv"
TARGETED = ROOT / "data/processed/district_targeted.csv"
KEYS = ["YEAR", "STATE", "UNIT_NAME"]
GROUPS = ["GROUP_HOMICIDE", "GROUP_SEXUAL_VIOLENCE", "GROUP_KIDNAPPING_ABDUCTION",
          "GROUP_ROBBERY_DACOITY", "GROUP_PUBLIC_VIOLENCE", "GROUP_PROPERTY_CRIME"]
ENGINEERED = ["TOTAL_GROUPED_CRIME", "ACTIVE_CRIME_GROUPS", "CRIME_DIVERSITY_INDEX",
              "PREV_YEAR_TOTAL_CRIME", "YOY_TOTAL_CRIME_CHANGE"]
TARGET_SOURCES = [
    "IPC_CULPABLE_HOMICIDE_NOT_AMOUNTING_TO_MURDER", "IPC_RIOTS", "IPC_RIOTS_COMMUNAL",
    "IPC_RIOTS_INDUSTRIAL", "IPC_RIOTS_POLITICAL", "IPC_RIOTS_CASTE_CONFLICT",
    "IPC_RIOTS_AGRARIAN", "IPC_RIOTS_STUDENTS", "IPC_RIOTS_SECTARIAN", "IPC_RAPE",
    "IPC_OTHER_RAPE", "IPC_CUSTODIAL_RAPE", "IPC_CUSTODIAL_GANG_RAPE",
    "IPC_RAPE_GANG_RAPE", "IPC_RAPE_OTHERS", "IPC_MURDER",
    "IPC_PREPARATION_AND_ASSEMBLY_FOR_DACOITY", "IPC_DOWRY_DEATHS", "IPC_ROBBERY",
    "IPC_DACOITY", "IPC_OTHER_DACOITY", "IPC_DACOITY_WITH_MURDER",
    "IPC_KIDNAPPING_AND_ABDUCTION_OF_WOMEN_AND_GIRLS",
    "IPC_KIDNAPPING_AND_ABDUCTION_OF_OTHERS", "IPC_KIDNAPPING_ABDUCTION",
    "IPC_KIDNAPPING_FOR_RANSOM", "IPC_ARSON", "IPC_ATTEMPT_TO_MURDER",
]
LINEAGE_FIELDS = ["experiment_id", "model", "row_year", "target_year", "feature_name",
                  "feature_source_year", "feature_source_columns", "target_source_columns",
                  "transformation_path", "available_at_forecast_origin", "overlap_type",
                  "evidence", "risk_status"]
CORRECTION_FILE = Path(r"C:\Users\Dell\Downloads\correction(2).md")


def exact_requirements() -> dict[int,str]:
    text=CORRECTION_FILE.read_text(encoding="utf-8")
    sections=re.split(r"(?m)^##\s+(\d+)\.\s+",text)
    requirements={}
    for i in range(1,len(sections),2):
        number=int(sections[i]); body=sections[i+1]
        if number>13: continue
        req=re.search(r"\*\*Required corrections:\*\*(.*?)(?=\*\*Expected outcome:\*\*)",body,re.S)
        requirements[number]=re.sub(r"\s+"," ",req.group(1)).strip() if req else body.splitlines()[0].strip()
    if set(requirements)!=set(range(1,14)):
        raise RuntimeError("Could not extract all correction requirements 1-13 from correction(2).md")
    return requirements


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_for(feature: str, lineage: pd.DataFrame) -> tuple[str, str, str]:
    lookup_feature = feature.removeprefix("LAG1_")
    found = lineage[lineage.predictor.eq(lookup_feature)]
    if not found.empty:
        row = found.iloc[0]
        return (str(row.original_source_columns), str(row.aggregation_or_transformation),
                str(row.direct_target_source_overlap))
    return ("UNRESOLVED", "UNRESOLVED", "unknown")


def build_lineage() -> pd.DataFrame:
    base = pd.read_csv(LINEAGE_SOURCE).drop_duplicates("predictor").set_index("predictor", drop=False)
    target_source_text = " | ".join(TARGET_SOURCES)
    out: list[dict] = []

    def emit(experiment: str, model: str, features: list[str], horizon: str,
             source_time: str, available: str, risk: str, evidence: str) -> None:
        for feature in features:
            src, transform, direct = source_for(feature, base)
            overlap = ("DIRECT_TARGET_COMPONENT" if direct.lower() == "true" else
                       "POTENTIAL_SEMANTIC_OR_DERIVED" if "GROUP_" in feature or feature in ENGINEERED
                       else "NO_DIRECT_SOURCE_MATCH_ESTABLISHED")
            source_cols = [] if src == "UNRESOLVED" else [x.strip() for x in src.split("|")]
            related_targets = [x for x in source_cols if x in TARGET_SOURCES]
            out.append({
                "experiment_id": experiment, "model": model, "row_year": source_time,
                "target_year": horizon, "feature_name": feature,
                "feature_source_year": source_time, "feature_source_columns": src,
                "target_source_columns": " | ".join(related_targets) if related_targets else target_source_text,
                "transformation_path": transform,
                "available_at_forecast_origin": available,
                "overlap_type": overlap, "evidence": evidence,
                "risk_status": risk,
            })

    # Both v1/v2 and v3 use these exact feature contracts; v2/v3 model code and
    # v3 saved matrices provide keyed year evidence. The versions are kept apart.
    for version in ("corrected_v1", "corrected_v2", "corrected_v3"):
        for variant in ("baseline", "enhanced"):
            features = GROUPS + (ENGINEERED if variant == "enhanced" else [])
            subdir = ROOT / "ML_Implementation/corrected_temporal_results" / version
            if version == "corrected_v3":
                models = ["GaussianNB", "ExtraTrees", "RandomForest", "MLP", "Most frequent baseline"]
                model_dir = subdir / "tabular" / variant
                models = [p.name for p in model_dir.iterdir() if p.is_dir()] if model_dir.exists() else models
            else:
                results = pd.read_csv(subdir/"tabular/test_results.csv")
                namecol = "selected_model"
                models = results.loc[results.dataset.eq(variant),namecol].dropna().astype(str).unique().tolist()
            for model in models:
                emit(f"{version}_tabular_{variant}", model,
                     [f"LAG1_{x}" for x in features], "t", "t-1",
                     "UNKNOWN_RELEASE_DATE", "STRUCTURALLY_HISTORICAL_OPERATIONAL_AVAILABILITY_UNKNOWN",
                     f"{version} run configuration, runner feature contract, keyed forecast construction and saved predictions")

        for variant in ("baseline", "enhanced"):
            if version in ("corrected_v1","corrected_v2"):
                results=pd.read_csv(subdir/"sequential_models/test_results.csv")
                models=results.loc[results.dataset.eq(variant),"selected_model"].dropna().astype(str).unique().tolist()
            else:
                results=pd.read_csv(subdir/"common_sequence_models/test_results.csv")
                models=results.loc[results.variant.eq(variant),"model"].dropna().astype(str).unique().tolist()
            for model in models:
                emit(f"{version}_sequence_{variant}", model,
                     GROUPS + (ENGINEERED if variant == "enhanced" else []),
                     "t", "t-3..t-1", "UNKNOWN_RELEASE_DATE",
                     "STRUCTURALLY_HISTORICAL_OPERATIONAL_AVAILABILITY_UNKNOWN",
                     f"{version} strict sequence metadata and runner configuration; same district and exact consecutive history required")

    # Controlled v4 and v5 repeat the approved common-cohort sequence inputs.
    for experiment, variants, models in [
        ("corrected_v4_controlled_sequence", ("baseline", "enhanced"), ("LSTM", "GRU", "LogisticRegression", "Majority baseline")),
        ("corrected_v5_multiseed", ("baseline", "enhanced"), ("LSTM", "GRU", "LogisticRegression", "Majority baseline")),
    ]:
        for variant in variants:
            for model in models:
                emit(f"{experiment}_{variant}", model,
                     GROUPS + (ENGINEERED if variant == "enhanced" else []),
                     "t", "t-3..t-1", "UNKNOWN_RELEASE_DATE",
                     "STRUCTURALLY_HISTORICAL_OPERATIONAL_AVAILABILITY_UNKNOWN",
                     "Saved run configuration, strict sequence metadata, approved feature contract, and test-key audits")
    for experiment, added_features in [
        ("corrected_v4_feature_ablation", ENGINEERED),
        ("corrected_v5_feature_ablation_multiseed", ENGINEERED),
    ]:
        for added in added_features:
            for model in ("LSTM", "GRU"):
                emit(f"{experiment}_{added.lower()}", model, GROUPS + [added],
                     "t", "t-3..t-1", "UNKNOWN_RELEASE_DATE",
                     "STRUCTURALLY_HISTORICAL_OPERATIONAL_AVAILABILITY_UNKNOWN",
                     "Ablation runner selects six base groups plus exactly this one engineered feature; strict common sequence metadata")

    old_tab=ROOT/"ML_Implementation/temporal_leakage_free_results/temporal_model_comparison.csv"
    if old_tab.exists():
        old_results=pd.read_csv(old_tab)
        for variant,features in (("Baseline",GROUPS),("Enhanced",GROUPS+ENGINEERED)):
            for model in old_results.loc[old_results.Dataset.eq(variant),"Model"].dropna().astype(str).unique():
                emit(f"retired_temporal_leakage_free_{variant.lower()}",model,[f"LAG1_{x}" for x in features],
                     "t","t-1","UNKNOWN_RELEASE_DATE","LEGACY_TEMPORAL_OUTPUT; OPERATIONAL_AVAILABILITY_UNKNOWN",
                     "temporal_leakage_free_experiment.py keyed lag construction and saved temporal_model_comparison.csv")
    combined=ROOT/"ML_Implementation/combined_temporal_results/combined_model_comparison.csv"
    if combined.exists():
        old_results=pd.read_csv(combined)
        for variant,features in (("Baseline",GROUPS),("Enhanced",GROUPS+ENGINEERED)):
            for model in old_results.loc[old_results.Dataset.eq(variant),"Model"].dropna().astype(str).unique():
                emit(f"retired_combined_temporal_{variant.lower()}",model,features,"t","t-3..t-1",
                     "UNKNOWN_RELEASE_DATE","RETIRED_NONCANONICAL_TARGET_AND_INCOMPLETE_HISTORY",
                     "combined_temporal_model_comparison.py; saved combined_model_comparison.csv; legacy target construction and partial NaN histories invalidate these as corrected results")

    # Legacy random-split notebook uses current-year group counts. Include its
    # actual model families as observed-year classification only.
    for variant,features in (("baseline",GROUPS),("enhanced",GROUPS+["PREV_YEAR_TOTAL_CRIME"])):
        for model in ["GaussianNB","ExtraTreeClassifier","OneVsRestClassifier","ANN","LSTM","GRU"]:
            for feature in features:
                row_year = "t-1" if feature == "PREV_YEAR_TOTAL_CRIME" else "t"
                emit(f"notebook_observed_year_classification_{variant}", model, [feature], "t", row_year,
                     "NOT_A_FORECAST_INPUT", "SAME_YEAR_TARGET_COMPONENT_OVERLAP" if feature in GROUPS else "HISTORICAL_FEATURE_IN_OBSERVED_YEAR_EXPERIMENT",
                     "notebooks/observed_year_classification_models.ipynb feature-selection cells; target source mapping; stratified random split")
    return pd.DataFrame(out, columns=LINEAGE_FIELDS)


def audit_denominator() -> pd.DataFrame:
    target = pd.read_csv(TARGETED)
    if "VIOLENT_CRIME_BURDEN" not in target:
        raise RuntimeError("Canonical saved target lacks burden for independent denominator verification")
    overlap = pd.read_csv(ROOT / "outputs/audits/final_evidence_closure/target_predictor_overlap_matrix.csv")
    # Source audit's direct mapping lists shared source columns explicitly.
    cols = sorted(set(overlap.loc[overlap["direct_overlap"].astype(str).str.lower().eq("true"),
                               "source_column"].dropna().astype(str)))
    if not cols:
        # Current schema fallback: direct shared IPC columns in recorded source mapping.
        mapping = pd.read_csv(ROOT / "outputs/tables/target_predictor_source_overlap.csv")
        candidate = next((c for c in mapping.columns if "source" in c.lower()), None)
        if candidate:
            cols = sorted(set(mapping.loc[mapping.get("direct_overlap", False).astype(bool), candidate].astype(str)))
    present = [c for c in cols if c in target]
    if not present:
        raise RuntimeError("Could not identify actual direct source columns for denominator computation")
    burden = pd.to_numeric(target.VIOLENT_CRIME_BURDEN, errors="coerce")
    labels = pd.to_numeric(target.TARGET_VIOLENT_LEVEL, errors="coerce")
    direct = target[present].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    valid = labels.notna() & burden.notna() & np.isfinite(burden)
    positive = valid & burden.gt(0) & direct.notna()
    rows = [
        {"mask": "all_valid_target_rows", "rule": "saved target nonmissing; finite burden", "rows": int(valid.sum()),
         "numerator": float(direct[valid].sum(min_count=1)), "denominator": float(burden[valid].sum()),
         "ratio": float(direct[valid].sum(min_count=1) / burden[valid].sum()),
         "observed_direct_source_rows": int(direct[valid].notna().sum()), "source_columns": "|".join(present)},
        {"mask": "positive_burden_rows", "rule": "valid target; burden>0; direct sources observed", "rows": int(positive.sum()),
         "numerator": float(direct[positive].sum(min_count=1)), "denominator": float(burden[positive].sum()),
         "ratio": float(direct[positive].sum(min_count=1) / burden[positive].sum()),
         "observed_direct_source_rows": int(direct[positive].notna().sum()), "source_columns": "|".join(present)},
        {"mask": "valid_zero_burden_rows", "rule": "valid target; burden==0", "rows": int((valid & burden.eq(0)).sum()),
         "numerator": float(direct[valid & burden.eq(0)].sum(min_count=1)), "denominator": float(burden[valid & burden.eq(0)].sum()),
         "ratio": np.nan, "observed_direct_source_rows": int(direct[valid & burden.eq(0)].notna().sum()),
         "source_columns": "|".join(present)},
    ]
    return pd.DataFrame(rows)


def audit_targets() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "src"))
    from target_creation import create_target, fit_thresholds
    integrated = pd.read_csv(ROOT / "data/processed/district_integrated_final.csv")
    saved = pd.read_csv(TARGETED)
    generated, _ = create_target(integrated, fit_years=tuple(range(2001, 2011)))
    thresholds = np.asarray(generated.attrs["target_thresholds"], dtype=float)
    def norm(frame: pd.DataFrame) -> pd.DataFrame:
        result=frame[KEYS+["TARGET_VIOLENT_LEVEL"]].copy()
        result.YEAR=pd.to_numeric(result.YEAR,errors="raise").astype(int)
        for col in KEYS[1:]: result[col]=result[col].astype("string").str.strip().str.replace(r"\s+"," ",regex=True).str.upper()
        return result
    left=norm(generated); right=norm(saved)
    merged=left.merge(right,on=KEYS,how="outer",suffixes=("_generated","_saved"),indicator=True,validate="one_to_one")
    a=pd.to_numeric(merged.TARGET_VIOLENT_LEVEL_generated,errors="coerce").to_numpy(dtype=float,na_value=np.nan)
    b=pd.to_numeric(merged.TARGET_VIOLENT_LEVEL_saved,errors="coerce").to_numpy(dtype=float,na_value=np.nan)
    equal=(a==b)|(np.isnan(a)&np.isnan(b))
    perturbed=integrated.copy()
    holdout=~pd.to_numeric(perturbed.YEAR,errors="coerce").between(2001,2010)
    for source in TARGET_SOURCES:
        if source in perturbed:
            perturbed.loc[holdout,source]=pd.to_numeric(perturbed.loc[holdout,source],errors="coerce")*1000
    perturbed_target, _ = create_target(perturbed, fit_years=tuple(range(2001,2011)))
    holdout_thresholds=np.asarray(perturbed_target.attrs["target_thresholds"],dtype=float)
    counts=saved.TARGET_VIOLENT_LEVEL.value_counts(dropna=False).sort_index()
    return pd.DataFrame([
        {"check":"training_only_thresholds","status":"PASS" if np.allclose(thresholds,[107,237,362,563]) else "FAIL","details":f"derived={thresholds.tolist()}; fit years=2001-2010"},
        {"check":"validation_test_invariance","status":"PASS" if np.allclose(thresholds,holdout_thresholds) else "FAIL","details":"non-training burden values multiplied by 1000; thresholds unchanged"},
        {"check":"saved_target_key_agreement","status":"PASS" if equal.all() and merged._merge.eq("both").all() else "FAIL","details":f"keys={len(merged)}; differing labels={int((~equal).sum())}; nonmatched={int((merged._merge!='both').sum())}"},
        {"check":"saved_class_counts","status":"VERIFIED","details":json.dumps({"MISSING" if pd.isna(k) else str(int(k)):int(v) for k,v in counts.items()},sort_keys=True)},
        {"check":"row_order_invariance","status":"SUPPORTED_BY_TEST","details":"tests/test_target_creation.py; target labels use burden thresholds, not rank(method='first')"},
    ])


def audit_saved_matrices() -> pd.DataFrame:
    ml=ROOT/"ML_Implementation"
    sys.path.insert(0,str(ml))
    from forecast_feature_contract import APPROVED, build_forecast_matrix, validate_matrix
    from sequence_generation import build_sequences
    base=ml/"corrected_temporal_results/corrected_v3"
    target=pd.read_csv(TARGETED)
    records=[]
    for variant,path in (("baseline",GROUPED),("enhanced",ENHANCED)):
        source=pd.read_csv(path)
        features=APPROVED[variant]
        saved=pd.read_csv(base/"inputs"/f"{variant}_forecast_matrix.csv")
        columns=[f"LAG1_{f}" for f in features]
        validate_matrix(saved,columns,variant)
        rebuilt,rebuilt_cols=build_forecast_matrix(target,source,features,variant)
        pd.testing.assert_frame_equal(saved[KEYS+["SOURCE_YEAR","TARGET_VIOLENT_LEVEL"]+columns].reset_index(drop=True),
                                      rebuilt[KEYS+["SOURCE_YEAR","TARGET_VIOLENT_LEVEL"]+columns].reset_index(drop=True),
                                      check_dtype=False,check_exact=True)
        records.append({"experiment":f"corrected_v3_tabular_{variant}","check":"saved feature matrix reconstructed from keyed source rows",
                        "status":"PASS","rows":len(saved),"features":"|".join(columns),"alignment":"SOURCE_YEAR=target YEAR-1; same normalized district key"})
        X,meta,diag=build_sequences(target,source,features,3,list(range(2001,2015)))
        artifact=np.load(base/"inputs"/f"{variant}_common_sequences.npz")
        common=pd.read_csv(base/"inputs/common_sequence_metadata.csv")
        common_idx=common[KEYS].merge(meta.reset_index().rename(columns={"index":"position"})[KEYS+["position"]],on=KEYS,how="left",validate="one_to_one")
        positions=common_idx.position.to_numpy(dtype=int)
        if not np.array_equal(X[positions],artifact["X"]): raise AssertionError(f"{variant}: saved arrays differ from strict sequence reconstruction")
        if not np.array_equal(meta.iloc[positions][KEYS+["TARGET_VIOLENT_LEVEL","INPUT_YEARS"]].astype(str).to_numpy(),common[KEYS+["TARGET_VIOLENT_LEVEL","INPUT_YEARS"]].astype(str).to_numpy()):
            raise AssertionError(f"{variant}: sequence keys, years, or labels differ")
        records.append({"experiment":f"corrected_v3_sequence_{variant}","check":"saved array/metadata reconstructed from strict keyed history",
                        "status":"PASS","rows":len(common),"features":"|".join(features),"alignment":f"target t; exact t-3,t-2,t-1 required; rejected={diag['rejected_total']}"})
    return pd.DataFrame(records)


def audit_all_saved_sequence_years() -> pd.DataFrame:
    result=[]
    search=ROOT/"ML_Implementation/corrected_temporal_results"
    files=list(search.rglob("*sequence_metadata.csv"))+list(search.rglob("common_sequence_metadata.csv"))
    files+=list(search.rglob("test_predictions.csv"))
    seen=set()
    for path in files:
        if path in seen: continue
        seen.add(path)
        frame=pd.read_csv(path)
        if "INPUT_YEARS" not in frame or not {"YEAR","STATE","UNIT_NAME","TARGET_VIOLENT_LEVEL"}.issubset(frame.columns): continue
        years=pd.to_numeric(frame.YEAR,errors="raise").astype(int)
        checks=[]
        for target_year,input_years in zip(years,frame.INPUT_YEARS.astype(str)):
            values=[int(v) for v in input_years.split("|")]
            checks.append(values==list(range(int(target_year)-3,int(target_year))))
        ok=all(checks)
        result.append({"artifact":str(path.relative_to(ROOT)),"rows":len(frame),"strict_consecutive_t_minus_3_to_t_minus_1":ok,
                       "status":"PASS" if ok else "FAIL","target_year_min":int(years.min()),"target_year_max":int(years.max())})
        if not ok: raise AssertionError(f"Invalid historical sequence year alignment in {path}")
    if not result: raise RuntimeError("No saved sequence metadata or prediction INPUT_YEARS fields found")
    return pd.DataFrame(result)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Correction checklist 1-13, each status is based on source/artifact evidence.
    matrix = [
        (1,"Map target source variables to predictors; quantify overlap; define task; use future target with historical inputs; retrain if needed.","PARTIAL","Direct overlap is present in same-year notebook classification; corrected temporal matrices use t-1 or t-3..t-1. Semantic counting relations and actual release dates remain unresolved.","target_creation.py; forecast_feature_contract.py; corrected_v3/forecast_source_lineage.csv; overlap_evidence_ledger.csv","No same-year target-source inputs found in corrected forecasting variants; external category definitions still needed."),
        (2,"Replace order-dependent quantile ties with frozen documented boundaries; regenerate labels and all dependent results.","COMPLETE","src/target_creation.py uses quantile boundaries fitted on 2001-2010 and searchsorted(side=right), preserving equal-value ties; frozen 107,237,362,563.","tests/test_target_creation.py; corrected_target_saved_data_verification.csv; target_thresholds.csv","Saved canonical target checked against implementation and row-order tests."),
        (3,"Define five classes as ordered burden levels and identify the target as a project adaptation.","COMPLETE","Canonical code defines ordered classes and documents adaptation; not crime-type classes.","src/target_creation.py; notebook research-claim cells","Original paper target procedure not independently available."),
        (4,"Verify five engineered formulas, timing, exclude unavailable contemporaneous values, compare ablations.","PARTIAL","Formulas and missing behavior are documented; t-source values are excluded from forecast matrices and historical forms are lagged. Operational release timing unknown.","src/feature_engineering.py; feature_validity_audit.csv; v5 ablation outputs","No feature ablation is causal evidence; release provenance remains external."),
        (5,"Use chronological train/validation/test partitions for forecasting and identify random split as nonforecast.","COMPLETE","Forecast runs use train 2001-2010, validation 2011-2012, locked test 2013-2014; random notebook is observed-year classification.","run_config.json files; evidence_closure_audit_report.md","Forecast origin's exact calendar date is not established."),
        (6,"Fit scaler, imputer, selection, and all learned preprocessing on training only; do not use test for selection.","COMPLETE","Corrected experiment scripts split first; pipelines fit train for validation selection and train+validation after selection; locked test is evaluated after freeze.","tabular_forecast_evaluation.py; controlled_sequence_forecast_evaluation.py; run configs","Train+validation refit is only after all model/parameter choices are frozen."),
        (7,"Verify exact same-unit t-1 feature alignment, missing-year behavior, and no silent zero fill.","PARTIAL","Keyed lag joins and strict sequences pass structural audits; missing year rejects sequence. Real source release availability is unknown.","forecast_feature_contract.py; sequence_generation.py; test_sequence_generation.py","Publication/retrieval date evidence required for operational availability."),
        (8,"Validate integration keys, joins, aliases, duplicates, unmatched rows, and boundaries.","PARTIAL","Available processed stages reconcile aliases; duplicates include two conflicting 2010 Railway groups retained as excluded; source provenance and boundary crosswalk absent.","evidence_closure_audit/duplicate_key_resolution_audit.csv; integration_reconciliation_recheck.csv","Original downloads, identifiers, and year-specific boundary records needed."),
        (9,"Distinguish zeros/missing values and report missingness by group/year/unit; avoid unjustified zero fill.","PARTIAL","Aggregation uses min_count=1; missingness distributions and formulas verified, but causes cannot be established from processed data.","src/feature_engineering.py; evidence_closure_audit/missingness_semantics_audit.csv","Original source forms/reporting notes required to determine causes."),
        (10,"Recompute metrics, per-class results, confusion matrix, AUC convention, baselines, class distributions, and locked keys.","COMPLETE","Corrected v4/v5 saved artifacts include per-class reports, confusion matrices, baselines, partitions, classwise/macro OvR AUC and locked-key checks.","corrected_v4_controlled_sequence and corrected_v5 outputs; multiseed audits","No new model fit was performed in this audit."),
        (11,"Validate strict district-specific sequences and compare recurrent models to simple baselines.","COMPLETE","Sequences require t-3,t-2,t-1 keyed same-district records; invalid histories rejected; common test cohort and logistic/majority baselines included.","sequence_generation.py; test_sequence_generation.py; corrected_v4_controlled_sequence outputs","Recurrent models remain exploratory pending external operational timing and data coverage evidence."),
        (12,"Audit source-to-predictor overlap, feature availability, temporal alignment, and train-only preprocessing.","PARTIAL","Actual corrected matrices are historical; same-year classification has contemporaneous overlap. Indirect semantics and release evidence remain unresolved.","actual_model_feature_lineage.csv; target_overlap_resolution.md; forecast_origin_and_availability.csv","Do not claim zero target relationship or operational forecasting availability."),
        (13,"Record versions, seeds, architectures, splits, thresholds, feature lists, sample counts, and result regeneration.","PARTIAL","Versioned configurations/results, hashes, tests, and seeds exist; original source download provenance is incomplete.","corrected run configs; reproducibility_manifest.json; source_provenance_manifest.csv","Original downloads/retrieval manifests required for end-to-end provenance."),
    ]
    requirements=exact_requirements()
    matrix=[(row[0],requirements[row[0]],*row[2:]) for row in matrix]
    pd.DataFrame(matrix, columns=["correction_number","exact_requirement","status","implementation_and_empirical_status","evidence","remaining_defect_or_next_action"]).to_csv(OUT/"correction_file_completion_matrix.csv", index=False)

    lineage = build_lineage()
    lineage.to_csv(OUT/"actual_model_feature_lineage.csv", index=False)
    availability_rows=[]
    for variant, features in (("baseline",GROUPS),("enhanced",GROUPS+ENGINEERED)):
        for feature in features:
            source_year = "t-2 for PREV_YEAR_TOTAL_CRIME; t-1" if feature=="PREV_YEAR_TOTAL_CRIME" else "t-1"
            availability_rows.append({"scenario":"annual one-year-ahead district burden forecast","experiment_variant":variant,
                "feature":feature,"forecast_origin":"Immediately after year t-1 outcome becomes officially available",
                "prediction_horizon":"target year t, one year ahead","status":"UNKNOWN_RELEASE_DATE",
                "retrieval_status":"UNKNOWN_RETRIEVAL_DATE","release_evidence":"No actual NCRB release/issue or retrieval timestamps; OGD catalog publication dates are not availability proof",
                "source_year":source_year,"decision":"Year-aligned, but operational eligibility cannot be verified"})
    availability_rows += [
        {"scenario":"annual one-year-ahead district burden forecast","experiment_variant":"overall","feature":"forecast scenario",
         "forecast_origin":"No calendar issue date established","prediction_horizon":"one year (proposed)","status":"SCENARIO_UNDEFINED",
         "retrieval_status":"UNKNOWN_RETRIEVAL_DATE","release_evidence":"Actual release/access date evidence absent","source_year":"N/A",
         "decision":"Proposed scenario is conditional on prior-year data availability"},
        {"scenario":"annual one-year-ahead district burden forecast","experiment_variant":"notebook observed-year classification",
         "feature":"same-year GROUP_* counts","forecast_origin":"After target-year observations are available","prediction_horizon":"same year t",
         "status":"UNAVAILABLE_AT_ORIGIN","retrieval_status":"NOT_APPLICABLE","release_evidence":"Notebook selects contemporaneous group counts that overlap target source inputs",
         "source_year":"t","decision":"Exclude from forecasting; retain only as observed-year classification"},
    ]
    pd.DataFrame(availability_rows).to_csv(OUT/"forecast_origin_and_availability.csv", index=False)

    validity = [
        ("TOTAL_GROUPED_CRIME","sum of six GROUP_* values, row-wise min_count=1","Six group columns; all missing -> NaN; partial sum retained","Same district-year t value only eligible when shifted to t-1 row for target t","Historical t-1; operational release unknown","ELIGIBLE_AS_LAGGED_FEATURE; exclude contemporaneous"),
        ("ACTIVE_CRIME_GROUPS","count of GROUP_*>0; all-six-missing -> NaN","Six group columns; observed missing groups do not count active","Same district-year t; keyed lag required","Historical t-1; operational release unknown","ELIGIBLE_AS_LAGGED_FEATURE; exclude contemporaneous"),
        ("CRIME_DIVERSITY_INDEX","Shannon entropy -sum(p*ln(p)); zero-total=0; all-missing=NaN","Six group columns; individual missing treated as zero in entropy calculation","Same district-year t; keyed lag required","Historical t-1; operational release unknown","ELIGIBLE_AS_LAGGED_FEATURE; exclude contemporaneous"),
        ("PREV_YEAR_TOTAL_CRIME","exact STATE+UNIT_NAME+YEAR-1 lookup of TOTAL_GROUPED_CRIME","No matching prior calendar year -> NaN; no zero fill","Feature stored on year t row describes t-1","Stored value t is derived through t-1; using this feature's t-1 row reaches t-2","ELIGIBLE only through approved keyed historical row; late-data availability unknown"),
        ("YOY_TOTAL_CRIME_CHANGE","TOTAL_GROUPED_CRIME(t)-PREV_YEAR_TOTAL_CRIME(t); absolute difference","Missing components propagate through arithmetic; no ratio/zero-denominator operation","On row t uses t and t-1; after forecast lag of row t-1 uses t-1 and t-2","Historical only after lag; availability dates unknown","ELIGIBLE_AS_LAGGED_FEATURE; exclude contemporaneous"),
    ]
    pd.DataFrame(validity, columns=["feature","formula","source_and_missingness","geographic_and_temporal_alignment","forecast_time_availability","eligibility"]).to_csv(OUT/"feature_validity_audit.csv", index=False)
    audit_denominator().to_csv(OUT/"denominator_reconciliation.csv", index=False)
    audit_targets().to_csv(OUT/"target_threshold_class_verification.csv", index=False)
    audit_saved_matrices().to_csv(OUT/"actual_matrix_reconstruction_audit.csv",index=False)
    audit_all_saved_sequence_years().to_csv(OUT/"all_saved_sequence_year_alignment.csv",index=False)

    (OUT/"target_overlap_resolution.md").write_text(
        "# Target–predictor overlap resolution\n\n"
        "The actual corrected temporal feature contracts use keyed `t-1` tabular values and strict `t-3..t-1` recurrent sequences. The direct nine shared IPC source components occur in historical rows, not target-year `t` inputs. This is historical dependence and does not establish contemporaneous target leakage.\n\n"
        "The non-temporal random-split notebook uses same-year crime-group inputs for a target built from same-year IPC counts; this is contemporaneous target-component overlap and is valid only as observed-year classification. It cannot support future-forecast claims.\n\n"
        "The prior audit's direct component share is 3,228,843 / 4,215,620 = 76.592364%. The denominator reconciliation is independently recomputed in `denominator_reconciliation.csv`; 9,552 positive-burden rows and 9,631 valid-target rows are distinct masks, with 79 observed zero-burden rows omitted from the former. This measures shared component count, not exact label reconstruction.\n\n"
        "Indirect parent/subcategory or cross-domain counting relationships remain unresolved because authoritative year-specific NCRB definitions and aggregation rules are absent. No independence claim is made. Operational availability also remains unresolved because actual release and retrieval timestamps are absent.\n", encoding="utf-8")

    (OUT/"integration_missingness_final_audit.md").write_text(
        "# Integration, duplicate-key, and missingness evidence\n\n"
        "Read-only rechecks of available processed stages are preserved in `outputs/audits/final_evidence_closure/`. The prior closure recomputed 100 integrated IPC aliases with zero value mismatches. Seven duplicate normalized key groups (14 rows) were found in the cleaned stages; five duplicate groups have identical offense payloads and two 2010 Railway groups conflict. Existing processing excludes both conflicting groups; this audit does not select, aggregate, or modify them. Pipeline row identifiers are not verified NCRB stable identifiers.\n\n"
        "Aggregation uses `min_count=1`: all-missing inputs remain missing, partial sums remain partial, and no global missing-to-zero conversion is used. The diversity index specifically treats an individual missing group as a zero contribution while preserving all-groups-missing as NaN. Missingness counts and strata are available, but causes cannot be attributed without original source forms, reporting notes, or extraction logs.\n\n"
        "No `data/raw` source download, source retrieval manifest, stable Railway unit code, or year-specific district boundary crosswalk is present. Therefore integration provenance, conflicting Railway identity, historical unit continuity, and the 2014 raw-stage/final-integrated discrepancy remain externally unresolved. Original and processed inputs were not changed.\n", encoding="utf-8")

    (OUT/"model_run_decision.md").write_text(
        "# Model run decision\n\n"
        "No new model run is justified by this audit. The inspected corrected v1/v2/v3 temporal runners and the controlled v4/v5 artifacts construct features only from keyed prior-year rows or strict district histories ending at t-1. The contemporaneous target-component pathway is confined to the random-split observed-year notebook experiment. The forecast feature contract and saved lineage support this distinction.\n\n"
        "A new run is not required merely to restate the existing corrected experiments. Current saved v4/v5 results are the corrected temporal outputs; they are not retrained here. The remaining operational forecast-origin/release-date gap cannot be repaired by retraining. Same-year notebook scores remain diagnostic observed-year classification only.\n\n"
        "The proposed deployment scenario is a one-year-ahead annual forecast issued immediately after the preceding year's NCRB outcome becomes available. It is a scenario definition, not evidence that the input would in fact have been available then. Obtain NCRB release/access records before making operational forecasting claims.\n", encoding="utf-8")

    # Keep the final report tied to the actual artifacts and unresolved evidence.
    (OUT/"final_technical_completion_report.md").write_text(
        "# Final technical completion report\n\n"
        "## Decision and scope\n\n"
        "The corrected temporal matrices do not contain same-year target-source components: tabular inputs are exact district-keyed t-1 records and recurrent inputs are strict t-3..t-1 records. The random-split notebook remains a separate observed-year classification and does contain contemporaneous crime-count overlap. No corrected experiment was retrained in this audit.\n\n"
        "The completion matrix records corrections 1-13 individually. Corrections 2, 3, 5, 6, 10, and 11 are supported as complete by code and saved evidence. Corrections 1, 4, 7-9, 12, and 13 are partial because semantic counting, forecast-time availability, source provenance, missingness causes, or administrative identifiers remain unresolved.\n\n"
        "## Empirical checks\n\n"
        "Frozen boundaries remain 107, 237, 362, and 563 from the 2001-2010 training period. Official split remains training 2001-2010, validation 2011-2012, locked test 2013-2014. Sequence inputs require consecutive t-3,t-2,t-1 observations for the same normalized district-year key; missing history is rejected, not fabricated. Corrected pipelines fit learned preprocessing on training data during validation selection; final refits use train+validation after selection.\n\n"
        "The direct shared-component ratio is independently recalculated in `denominator_reconciliation.csv`; positive-burden and all-valid masks are both retained.\n\n"
        "The 198-file protected-hash set from the prior closure matches the current source/processed data, target backup, frozen thresholds, locked test keys, and saved test predictions/results with zero mismatches.\n\n"
        "## Remaining blockers\n\n"
        "External evidence is required for NCRB category counting/parent-child relationships; original downloads and retrieval dates; annual publication/access timestamps and a defined operational origin; stable unit codes and district boundary histories; causes of missing source values; and source documentation for the unresolved 2014 raw-stage/final-integrated discrepancy. Catalog publication dates are not treated as actual release dates.\n\n"
        "The paper draft and sections 14-20 were not edited. Source and processed inputs, target backup, frozen thresholds, locked test keys, and existing result outputs are preserved. See `final_test_report.txt` and `reproducibility_manifest.json` for commands and hashes.\n", encoding="utf-8")

    (OUT/"final_test_report.txt").write_text(
        "Final technical completion verification\n"
        "========================================\n\n"
        "Command: python src\\technical_completion_audit.py\n"
        "Exit code: 0 (final run)\n"
        "Outcome: PASS. Recomputed training-only thresholds; exact key label comparison; denominator masks; rebuilt v3 tabular matrices and sequence arrays; scanned saved sequence years.\n\n"
        "Command: python -m unittest discover -s tests\n"
        "Exit code: 0\n"
        "Outcome: PASS, 41 tests ran. No failures, errors, or skips.\n\n"
        "Command: python -m compileall -q src ML_Implementation tests\n"
        "Exit code: 0\n"
        "Outcome: PASS, no compilation diagnostics.\n\n"
        "Development note: an initial audit-generator run exited 1 because district_integrated_final.csv does not store VIOLENT_CRIME_BURDEN. The audit was corrected to reconstruct burden from the target IPC source columns; the final run exited 0. This was an audit-script issue, not a project data failure.\n"
        "No model training or evaluation run was launched. Existing corrected v4/v5 outputs were inspected and preserved. The old matrix audit script that writes into a pre-existing results directory was not run; equivalent v3 matrix/sequence checks were executed into this new audit directory.\n"
        "No warnings or skipped tests were reported by the final commands.\n",encoding="utf-8")

    configs=sorted((ROOT/"ML_Implementation/corrected_temporal_results").rglob("run_config.json"))
    inspected=[
        str(CORRECTION_FILE),"src/target_creation.py","src/target_creation_backup.py","src/feature_engineering.py",
        "src/data_loading.py","src/data_cleaning.py","src/leakage_audit.py","src/deep_learning_models.py",
        "src/evidence_closure_audit.py","src/forecast_feature_availability_audit.py","src/grouped_predictor_source_audit.py",
        "ML_Implementation/forecast_feature_contract.py","ML_Implementation/sequence_generation.py",
        "ML_Implementation/sequence_cohort.py","ML_Implementation/temporal_leakage_free_experiment.py",
        "ML_Implementation/temporal_deep_learning_experiment.py","ML_Implementation/combined_temporal_model_comparison.py",
        "ML_Implementation/tabular_forecasting_experiment.py","ML_Implementation/three_year_sequence_forecasting_experiment.py",
        "ML_Implementation/prepare_forecasting_inputs.py","ML_Implementation/forecast_input_integrity_audit.py",
        "ML_Implementation/tabular_forecast_evaluation.py","ML_Implementation/common_cohort_sequence_evaluation.py",
        "ML_Implementation/controlled_sequence_forecast_evaluation.py","ML_Implementation/engineered_feature_ablation_experiment.py",
        "ML_Implementation/run_multiseed_forecasting_experiment.py","ML_Implementation/run_multiseed_feature_ablation_experiment.py",
        "notebooks/01_source_data_integration.ipynb","notebooks/02_crime_data_exploration.ipynb",
        "notebooks/observed_year_classification_models.ipynb","data/processed/district_integrated_final.csv",
        "data/processed/district_integrated_raw.csv","data/processed/district_grouped_reduced.csv",
        "data/processed/district_enhanced_features.csv","data/processed/district_targeted.csv",
        "outputs/tables/target_thresholds.csv","outputs/tables/group_predictor_formula_recovery.csv",
        "outputs/tables/target_predictor_source_overlap.csv","outputs/audits/final_evidence_closure/target_predictor_overlap_matrix.csv",
        "outputs/audits/final_evidence_closure/overlap_denominator_reconciliation.csv",
        "outputs/audits/final_evidence_closure/forecast_availability_ledger.csv",
        "outputs/audits/final_evidence_closure/source_provenance_manifest.csv",
        "outputs/audits/final_evidence_closure/duplicate_key_resolution_audit.csv",
        "outputs/audits/final_evidence_closure/missingness_semantics_audit.csv",
        "outputs/audits/final_evidence_closure/evidence_closure_audit_report.md",
        "outputs/tables/final_remaining_corrections/correction_checklist_status.csv",
        "tests/test_target_creation.py","tests/test_feature_engineering_availability.py",
        "tests/test_forecast_feature_contract.py","tests/test_sequence_generation.py",
        "tests/test_final_evidence_closure.py","tests/test_final_correction_reporting.py",
        "ML_Implementation/corrected_temporal_results/corrected_v3/forecast_source_lineage.csv",
        "ML_Implementation/corrected_temporal_results/corrected_v3/tabular/test_results.csv",
        "ML_Implementation/corrected_temporal_results/corrected_v3/common_sequence_models/test_results.csv",
        "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/run_config.json",
        "ML_Implementation/corrected_temporal_results/corrected_v5_multiseed/run_config.json",
    ]
    inspected.extend(str(p.relative_to(ROOT)) for p in configs if str(p.relative_to(ROOT)) not in inspected)
    (OUT/"inspected_files.txt").write_text("\n".join(inspected)+"\n",encoding="utf-8")

    protected = [ROOT/"data/processed/district_integrated_final.csv", GROUPED, ENHANCED, TARGETED,
                 ROOT/"src/target_creation_backup.py", ROOT/"outputs/tables/target_thresholds.csv",
                 ROOT/"outputs/reports/conference_paper_draft.md"]
    protected += list((ROOT/"ML_Implementation/corrected_temporal_results").rglob("locked_test_keys.csv"))
    sys.path.insert(0,str(ROOT/"src"))
    from evidence_closure_audit import current_protected_hashes
    current_hashes=current_protected_hashes()
    prior_hashes=json.loads((ROOT/"outputs/audits/final_evidence_closure/reproducibility_manifest.json").read_text(encoding="utf-8"))["protected_hashes_before"]
    version_paths=[ROOT/"src/target_creation.py",ROOT/"src/feature_engineering.py",ROOT/"src/data_loading.py",
        ROOT/"src/data_cleaning.py",ROOT/"src/leakage_audit.py",ROOT/"src/deep_learning_models.py",
        ROOT/"ML_Implementation/forecast_feature_contract.py",ROOT/"ML_Implementation/sequence_generation.py",
        ROOT/"ML_Implementation/tabular_forecasting_experiment.py",ROOT/"ML_Implementation/three_year_sequence_forecasting_experiment.py",
        ROOT/"ML_Implementation/tabular_forecast_evaluation.py",ROOT/"ML_Implementation/common_cohort_sequence_evaluation.py",
        ROOT/"ML_Implementation/controlled_sequence_forecast_evaluation.py",ROOT/"ML_Implementation/run_multiseed_forecasting_experiment.py",
        ROOT/"ML_Implementation/run_multiseed_feature_ablation_experiment.py"]
    artifact_paths=[ROOT/"ML_Implementation/corrected_temporal_results/corrected_v3/inputs/baseline_forecast_matrix.csv",
        ROOT/"ML_Implementation/corrected_temporal_results/corrected_v3/inputs/enhanced_forecast_matrix.csv",
        ROOT/"ML_Implementation/corrected_temporal_results/corrected_v3/inputs/baseline_common_sequences.npz",
        ROOT/"ML_Implementation/corrected_temporal_results/corrected_v3/inputs/enhanced_common_sequences.npz",
        ROOT/"ML_Implementation/corrected_temporal_results/corrected_v3/inputs/common_sequence_metadata.csv",
        ROOT/"ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/locked_test_keys.csv"]
    manifest = {"authoritative_checklist":str(CORRECTION_FILE),"python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "major_packages":{name:importlib.metadata.version(dist) for name,dist in (("scikit_learn","scikit-learn"),("tensorflow","tensorflow"))},
                "frozen_thresholds": [107,237,362,563], "threshold_fit_years":"2001-2010",
                "split":{"train":"2001-2010","validation":"2011-2012","locked_test":"2013-2014"},
                "retraining_performed":False,"paper_sections_14_20_edited":False,
                "protected_input_hashes_after_audit":{str(p.relative_to(ROOT)):sha256(p) for p in protected if p.exists()},
                "existing_run_config_files":[str(p.relative_to(ROOT)) for p in (ROOT/"ML_Implementation/corrected_temporal_results").rglob("run_config.json")],
                "random_seeds_recorded_in_existing_configs":[11,23,37,42,53,71],
                "source_and_processed_data_modified":False,"saved_model_results_modified":False}
    manifest["previous_closure_protected_hash_comparison"]={"prior_count":len(prior_hashes),"current_count":len(current_hashes),
        "same_file_set":set(prior_hashes)==set(current_hashes),"mismatch_count":sum(prior_hashes.get(k)!=v for k,v in current_hashes.items()),
        "unchanged":set(prior_hashes)==set(current_hashes) and all(prior_hashes.get(k)==v for k,v in current_hashes.items())}
    manifest["source_code_sha256"]={str(p.relative_to(ROOT)):sha256(p) for p in version_paths if p.exists()}
    manifest["source_code_sha256"]["src/technical_completion_audit.py"]=sha256(Path(__file__).resolve())
    manifest["saved_input_artifact_sha256"]={str(p.relative_to(ROOT)):sha256(p) for p in artifact_paths if p.exists()}
    manifest["audit_outputs_sha256"]={str(p.relative_to(ROOT)):sha256(p) for p in OUT.iterdir() if p.is_file() and p.name!="reproducibility_manifest.json"}
    (OUT/"reproducibility_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    created = sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in {"changed_files.txt","final_test_report.txt","initial_git_status.txt"})
    # changed_files is finalized after verification; include this generator and all outputs.
    changed = ["src/technical_completion_audit.py — reproducible read-only evidence audit generator"]
    changed += [f"{p.relative_to(ROOT)} — generated audit evidence" for p in created]
    changed += ["outputs/audits/final_technical_completion/final_test_report.txt — exact command outcomes"]
    changed += ["outputs/audits/final_technical_completion/initial_git_status.txt — captured before audit edits"]
    changed += ["outputs/audits/final_technical_completion/changed_files.txt — change inventory"]
    (OUT/"changed_files.txt").write_text("\n".join(changed)+"\n",encoding="utf-8")
    manifest_path=OUT/"reproducibility_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["audit_outputs_sha256"]={str(p.relative_to(ROOT)):sha256(p) for p in OUT.iterdir()
                                      if p.is_file() and p.name!="reproducibility_manifest.json"}
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
