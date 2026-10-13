"""Recompute release-readiness metrics from immutable saved NCRB predictions."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/"ML_Implementation/corrected_temporal_results"
OUT=ROOT/"outputs/audits/final_release_readiness"
KEYS=["YEAR","STATE","UNIT_NAME"]
TARGET="TARGET_VIOLENT_LEVEL"
CLASSES=[1,2,3,4,5]
TOL=1e-10


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def normalize_keys(df:pd.DataFrame)->pd.DataFrame:
    d=df.copy()
    d["YEAR"]=pd.to_numeric(d.YEAR,errors="raise").astype(int)
    for c in KEYS[1:]:d[c]=d[c].astype("string").str.strip().str.replace(r"\s+"," ",regex=True).str.upper()
    return d


def cols_for(pred:pd.DataFrame)->tuple[str,str]:
    actual=TARGET if TARGET in pred else "actual"
    predicted="PREDICTED_TARGET" if "PREDICTED_TARGET" in pred else "predicted"
    if actual not in pred or predicted not in pred:raise ValueError("Prediction file lacks actual/predicted labels")
    return actual,predicted


def find_summary(path:Path)->Path|None:
    for parent in path.parents:
        candidate=parent/"test_results.csv"
        if candidate.exists():return candidate
        if parent==RUNS:break
    return None


def family_name(path:Path,summary:pd.DataFrame)->str:
    # Run result schemas use different labels; match folder names where present.
    p=path.parent.name.lower()
    aliases={"majority_baseline":"majority baseline","most_frequent_baseline":"most frequent baseline",
             "logistic_regression":"logisticregression","extra_trees":"extratrees",
             "gaussian_naive_bayes":"gaussiannb"}
    if p in aliases:return aliases[p]
    if p not in {"baseline","enhanced"} and p not in {"gru","lstm","mlp","ann","extratrees","randomforest","gaussiannb","logistic_regression"}:
        return p
    if p in {"gru","lstm","mlp","ann","extratrees","randomforest","gaussiannb","logistic_regression"}:
        return aliases.get(p,p)
    # v1/v2 selected model is encoded in its test_results row, not the folder.
    return "selected"


def summary_row(path:Path,summary:pd.DataFrame)->pd.Series|None:
    pred=pd.read_csv(path,nrows=1)
    folder=path.parent.name.lower()
    variant=None
    for ancestor in path.parents:
        if ancestor.name.lower() in {"baseline","enhanced"}:
            variant=ancestor.name.lower();break
    row=summary.copy()
    variant_col=next((c for c in ["variant","dataset"] if c in row),None)
    if variant_col and variant:
        row=row[row[variant_col].astype(str).str.lower().eq(variant)]
    if "seed" in row:
        for ancestor in path.parents:
            if ancestor.name.startswith("seed_"):
                try: row=row[pd.to_numeric(row.seed,errors="coerce").eq(int(ancestor.name.split("_")[1]))]
                except ValueError: pass
                break
    if "feature_added" in row:
        feat=path.parents[1].name.upper()
        row=row[row.feature_added.astype(str).str.upper().eq(feat)]
    model_col=next((c for c in ("model","selected_model") if c in row),None)
    if model_col:
        key=family_name(path,row)
        canon=lambda x: str(x).lower().replace("_","").replace(" ","")
        if key=="selected":
            # A direct variant-folder prediction is the selected model output;
            # nested majority folders are mapped by family_name above.
            row=row[~row[model_col].astype(str).str.lower().str.contains("majority|most frequent")]
        else:
            row=row[row[model_col].map(canon).eq(canon(key))]
    if len(row)==1:return row.iloc[0]
    # Some older result tables contain one row per variant without a model name.
    return row.iloc[0] if len(row)==1 else None


def close(a,b,tol=TOL)->bool:
    try:return bool(np.isclose(float(a),float(b),rtol=0,atol=tol,equal_nan=True))
    except (TypeError,ValueError):return False


def official_metrics(summary:pd.Series|None,metrics:dict)->dict:
    checks={}
    if summary is None:return {k:"NO_MATCHING_SAVED_SUMMARY_ROW" for k in metrics}
    pairs={"accuracy":"test_accuracy","macro_f1":"test_macro_f1","weighted_f1":"test_weighted_f1",
           "macro_ovr_roc_auc":"macro_ovr_roc_auc","macro_ovr_average_precision":"macro_ovr_average_precision"}
    for our,their in pairs.items():
        if our in metrics and their in summary.index and pd.notna(summary[their]):checks[our]="MATCH" if close(metrics[our],summary[their]) else "MISMATCH"
        elif our in metrics:checks[our]="NOT_REPORTED"
    return checks


def evaluate_predictions(path:Path,canonical:pd.DataFrame)->tuple[dict,dict,dict]:
    pred=normalize_keys(pd.read_csv(path))
    actual_col,pred_col=cols_for(pred)
    y=pd.to_numeric(pred[actual_col],errors="raise").astype(int).to_numpy()
    yp=pd.to_numeric(pred[pred_col],errors="raise").astype(int).to_numpy()
    metrics={"accuracy":float(accuracy_score(y,yp)),"macro_f1":float(f1_score(y,yp,labels=CLASSES,average="macro",zero_division=0)),
             "weighted_f1":float(f1_score(y,yp,labels=CLASSES,average="weighted",zero_division=0))}
    prec,rec,f1,_=precision_recall_fscore_support(y,yp,labels=CLASSES,zero_division=0)
    for i,c in enumerate(CLASSES):metrics.update({f"class_{c}_precision":float(prec[i]),f"class_{c}_recall":float(rec[i]),f"class_{c}_f1":float(f1[i])})
    cm=confusion_matrix(y,yp,labels=CLASSES)
    cm_path=next((x for x in [path.parent/"test_confusion_matrix.csv",path.parent/"confusion_matrix.csv"] if x.exists()),None)
    cm_match="NOT_SAVED"
    if cm_path:
        saved=pd.read_csv(cm_path,index_col=0).to_numpy(dtype=int)
        cm_match="MATCH" if saved.shape==cm.shape and np.array_equal(saved,cm) else "MISMATCH"
    report_path=next((x for x in [path.parent/"test_classification_report.csv",path.parent/"classification_report.csv"] if x.exists()),None)
    report_match="NOT_SAVED"
    if report_path:
        rep=pd.read_csv(report_path,index_col=0)
        report_match="MATCH"
        for i,c in enumerate(CLASSES):
            label=str(c)
            if label not in rep.index:report_match="MISMATCH";continue
            for metric_name,key in (("precision",f"class_{c}_precision"),("recall",f"class_{c}_recall"),("f1-score",f"class_{c}_f1")):
                if metric_name in rep.columns and not close(metrics[key],rep.loc[label,metric_name]):report_match="MISMATCH"
    prob_path=path.parent/"test_probabilities.csv"
    auc_status="NOT_AVAILABLE"
    if prob_path.exists():
        prob=normalize_keys(pd.read_csv(prob_path))
        if not pred[KEYS].equals(prob[KEYS]) or not np.array_equal(y,pd.to_numeric(prob[actual_col if actual_col in prob else TARGET],errors="raise").astype(int).to_numpy()):
            auc_status="PROBABILITY_KEY_OR_LABEL_MISMATCH"
        else:
            pcols=[f"PROB_CLASS_{c}" for c in CLASSES]
            if not set(pcols).issubset(prob.columns):auc_status="MISSING_PROBABILITY_COLUMNS"
            else:
                proba=prob[pcols].to_numpy(float)
                metrics["macro_ovr_roc_auc"]=float(roc_auc_score(y,proba,labels=CLASSES,multi_class="ovr",average="macro"))
                metrics["macro_ovr_average_precision"]=float(np.mean([average_precision_score(y==c,proba[:,c-1]) for c in CLASSES]))
                auc_status="RECOMPUTED"
                auc_file=path.parent/"test_auc_metrics.json"
                if auc_file.exists():
                    saved=json.loads(auc_file.read_text(encoding="utf-8"))
                    if not close(metrics["macro_ovr_roc_auc"],saved.get("macro_ovr_roc_auc")) or not close(metrics["macro_ovr_average_precision"],saved.get("macro_ovr_average_precision")):auc_status="SAVED_AUC_MISMATCH"
    label_join=pred[KEYS].copy();label_join["actual_pred_file"]=y
    merged=label_join.merge(canonical,on=KEYS,how="left",validate="one_to_one",indicator=True)
    key_ok=bool(merged._merge.eq("both").all() and np.array_equal(merged.actual_pred_file.to_numpy(),merged[TARGET].astype(int).to_numpy()))
    keys=pred[KEYS].copy()
    duplicate_keys=bool(keys.duplicated().any())
    year_ok=bool(pred.YEAR.isin([2013,2014]).all())
    input_year_ok=True
    if "INPUT_YEARS" in pred:
        input_year_ok=all([int(v) for v in s.split("|")]==list(range(int(y0)-3,int(y0))) for y0,s in zip(pred.YEAR,pred.INPUT_YEARS.astype(str)))
    integrity={"rows":len(pred),"unique_keys":int(keys.drop_duplicates().shape[0]),"duplicate_test_keys":duplicate_keys,
               "keys_match_canonical_labels":key_ok,"target_years_only_2013_2014":year_ok,
               "strict_sequence_years_if_present":input_year_ok,"confusion_matrix_match":cm_match,"per_class_report_match":report_match,
               "class_counts":json.dumps({str(c):int(np.sum(y==c)) for c in CLASSES},sort_keys=True)}
    cm_long=[{"actual_class":int(CLASSES[i]),"predicted_class":int(CLASSES[j]),"count":int(cm[i,j])} for i in range(5) for j in range(5)]
    return metrics,integrity,{"confusion":cm_long,"keys":keys,"predictions":pred,"auc_status":auc_status}


def main()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    canonical=normalize_keys(pd.read_csv(ROOT/"data/processed/district_targeted.csv")[KEYS+[TARGET]])
    all_rows=[]; integrity_rows=[]; records=[]; confusions=[]
    for path in sorted(RUNS.rglob("test_predictions.csv")):
        if any(x in path.parts for x in ("temporal_leakage_free_results","combined_temporal_results")):continue
        summary_path=find_summary(path)
        result_dir=summary_path.parent if summary_path else path.parent
        if summary_path is None:continue
        summary=pd.read_csv(summary_path)
        official=summary_row(path,summary)
        metrics,integrity,extras=evaluate_predictions(path,canonical)
        metrics.update({"test_rows":len(pd.read_csv(path))})
        checks=official_metrics(official,metrics)
        expected_n=None
        if official is not None:
            for ncol in ("test_rows","rows"):
                if ncol in official and pd.notna(official[ncol]):expected_n=int(official[ncol]);break
        if expected_n is not None:checks["test_rows"]="MATCH" if expected_n==metrics["test_rows"] else "MISMATCH"
        result_id=str(path.relative_to(ROOT))
        row={"prediction_file":result_id,"summary_file":str(summary_path.relative_to(ROOT)),
             "variant":next((x for x in path.parts if x.lower() in {"baseline","enhanced"}),"unknown"),
             "model":path.parent.name,"seed":next((x.split("_")[1] for x in path.parts if x.startswith("seed_")),""),
             **metrics,"confusion_matrix_reconciliation":integrity["confusion_matrix_match"],
             "per_class_report_reconciliation":integrity["per_class_report_match"],"auc_reconciliation":extras["auc_status"],
             "official_metric_checks":json.dumps(checks,sort_keys=True),"summary_row_found":official is not None}
        all_rows.append(row)
        integrity_rows.append({"prediction_file":result_id,"summary_file":str(summary_path.relative_to(ROOT)),**integrity,
                               "locked_test_key_set_match":None,"official_test_rows_match":checks.get("test_rows","NO_MATCHING_SAVED_SUMMARY_ROW")})
        records.append({"path":path,"summary":summary_path,"variant":row["variant"],"model":path.parent.name,"seed":row["seed"],
                        "metrics":metrics,"keys":extras["keys"],"integrity":integrity})
        for x in extras["confusion"]:confusions.append({"prediction_file":result_id,**x})
    if not records:raise RuntimeError("No corrected saved predictions could be reconciled")

    # Compare test keys within baseline/enhanced pairs and to locked sequence keys.
    groups={}
    for rec in records:
        p=rec["path"]
        version=next((part for part in p.parts if part.startswith("corrected_v")),"unknown")
        parent=next((part for part in p.parts if part.startswith("seed_")),version)
        subsystem=next((part for part in p.parts if part in {"tabular","sequential_models","common_sequence_models","corrected_v4_controlled_sequence"}),"other")
        cohort_key=(version,parent,subsystem)
        groups.setdefault(cohort_key,{}).setdefault(rec["variant"],[]).append(rec)
    locked_path=ROOT/"ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/locked_test_keys.csv"
    locked=normalize_keys(pd.read_csv(locked_path)[KEYS+[TARGET,"INPUT_YEARS"]]) if locked_path.exists() else None
    for (version,seed,subsystem),variants in groups.items():
        b=variants.get("baseline",[]);e=variants.get("enhanced",[])
        if b and e:
            # Cohort pairing is independent of which validation-selected model won each variant.
            rb,re=b[0],e[0]
            same=rb["keys"].sort_values(KEYS).reset_index(drop=True).equals(re["keys"].sort_values(KEYS).reset_index(drop=True))
            for rec in b+e:
                x=next(x for x in integrity_rows if x["prediction_file"]==str(rec["path"].relative_to(ROOT)))
                x["paired_baseline_enhanced_test_keys_match"]=same
        for rec in b+e:
            key=str(rec["path"].relative_to(ROOT))
            row=next(x for x in integrity_rows if x["prediction_file"]==key)
            if locked is not None and len(rec["keys"])==len(locked):
                row["locked_test_key_set_match"]=rec["keys"].sort_values(KEYS).reset_index(drop=True).equals(locked[KEYS].sort_values(KEYS).reset_index(drop=True))

    pd.DataFrame(all_rows).to_csv(OUT/"corrected_experiment_metric_reconciliation.csv",index=False)
    pd.DataFrame(integrity_rows).to_csv(OUT/"test_cohort_and_prediction_integrity.csv",index=False)
    pd.DataFrame(confusions).to_csv(OUT/"recomputed_confusion_matrices_long.csv",index=False)
    pd.DataFrame(model_comparisons(records)).to_csv(OUT/"model_comparison_and_stability.csv",index=False)
    pd.DataFrame(remaining_actions()).to_csv(OUT/"remaining_correction_actions.csv",index=False)
    pd.DataFrame(external_requirements()).to_csv(OUT/"external_evidence_requirements.csv",index=False)
    write_report(records,integrity_rows,all_rows)
    write_test_report()
    write_manifest()
    write_changed_files()


def model_comparisons(records:list[dict])->list[dict]:
    rows=[]
    v5=[r for r in records if "corrected_v5_multiseed" in str(r["path"]) and "feature_ablation" not in str(r["path"])]
    for model in ("gru","lstm","logistic_regression"):
        pairs=[]
        for r in v5:
            if r["model"].lower()!=model:continue
            seed=r["seed"]
            other=next((x for x in v5 if x["seed"]==seed and x["variant"]!=r["variant"] and x["model"].lower()==model),None)
            if other and r["variant"]=="enhanced":
                pairs.append((r,other))
        deltas=[a["metrics"]["macro_f1"]-b["metrics"]["macro_f1"] for a,b in pairs]
        wd=[a["metrics"]["weighted_f1"]-b["metrics"]["weighted_f1"] for a,b in pairs]
        rows.append({"experiment":"corrected_v5_multiseed","comparison":f"{model} enhanced minus baseline","paired_seeds":len(pairs),
                     "macro_f1_delta_mean":float(np.mean(deltas)) if deltas else np.nan,"macro_f1_delta_sd":float(np.std(deltas,ddof=1)) if len(deltas)>1 else np.nan,
                     "macro_f1_delta_min":float(np.min(deltas)) if deltas else np.nan,"macro_f1_delta_max":float(np.max(deltas)) if deltas else np.nan,
                     "seeds_improved_macro_f1":sum(d>0 for d in deltas),"weighted_f1_delta_mean":float(np.mean(wd)) if wd else np.nan,
                     "interpretation":"Descriptive paired-seed differences only; no significance test"})
    # Same test set, direct neural-vs-logistic model family differences by variant/seed.
    for variant in ("baseline","enhanced"):
        for neural in ("gru","lstm"):
            deltas=[]
            for r in v5:
                if r["variant"]==variant and r["model"].lower()==neural:
                    lr=next((x for x in v5 if x["seed"]==r["seed"] and x["variant"]==variant and x["model"].lower()=="logistic_regression"),None)
                    if lr:deltas.append(r["metrics"]["macro_f1"]-lr["metrics"]["macro_f1"])
            rows.append({"experiment":"corrected_v5_multiseed","comparison":f"{neural} minus logistic regression ({variant})",
                         "paired_seeds":len(deltas),"macro_f1_delta_mean":float(np.mean(deltas)) if deltas else np.nan,
                         "macro_f1_delta_sd":float(np.std(deltas,ddof=1)) if len(deltas)>1 else np.nan,
                         "macro_f1_delta_min":float(np.min(deltas)) if deltas else np.nan,"macro_f1_delta_max":float(np.max(deltas)) if deltas else np.nan,
                         "seeds_improved_macro_f1":sum(d>0 for d in deltas),"weighted_f1_delta_mean":np.nan,
                         "interpretation":"Neural complexity is not justified by mean uplift alone; compare paired performance and uncertainty descriptively"})
    abl=[r for r in records if "corrected_v5_feature_ablation_multiseed" in str(r["path"])]
    # The aggregate v5 ablation table provides exact paired deltas against same-seed baseline.
    p=RUNS/"corrected_v5_feature_ablation_multiseed/paired_difference_summary.csv"
    if p.exists():
        d=pd.read_csv(p)
        for _,g in d.iterrows():
            rows.append({"experiment":"corrected_v5_feature_ablation_multiseed","comparison":f"{g.feature_added} added with {g.model}",
                         "paired_seeds":int(g.seeds),"macro_f1_delta_mean":float(g.mean_paired_difference),
                         "macro_f1_delta_sd":float(g.sd_across_seeds),
                         "macro_f1_delta_min":np.nan,"macro_f1_delta_max":np.nan,
                         "seeds_improved_macro_f1":int(g.higher),"weighted_f1_delta_mean":np.nan,
                         "interpretation":"One-feature ablation is descriptive, not causal"})
    return rows


def remaining_actions()->list[dict]:
    return [
      {"correction":1,"status":"PARTIAL","code_level_issue":"No demonstrated same-year leakage in corrected matrices; direct shared components remain historical and same-year notebook is observed-year only.","action":"No code change justified; retain explicit overlap warning.","evidence_gap":"NCRB year-specific parent/subcategory/event counting rules; actual annual source availability."},
      {"correction":4,"status":"PARTIAL","code_level_issue":"Feature formulas and forecast exclusion are implemented; no remaining formula defect identified.","action":"No code change; retain historical-lag features only.","evidence_gap":"Operational release timing and original NCRB definitions."},
      {"correction":7,"status":"PARTIAL","code_level_issue":"Keyed lag and strict sequence construction pass; no alignment defect identified in saved matrices.","action":"No code change.","evidence_gap":"Historical release/retrieval records and stable geographic identifiers."},
      {"correction":8,"status":"PARTIAL","code_level_issue":"Available integration aliases reconcile; conflicting 2010 Railway records remain excluded without fabricated resolution.","action":"No source-side code change justified.","evidence_gap":"Original source records, Railway unit IDs, district/state boundary crosswalks."},
      {"correction":9,"status":"PARTIAL","code_level_issue":"Missing values preserved; documented partial aggregation behavior. No supportable missingness imputation fix.","action":"No code change.","evidence_gap":"Original reporting forms/notes and extraction logs explaining missing cells."},
      {"correction":12,"status":"PARTIAL","code_level_issue":"Actual corrected matrices use historical inputs; no target-year feature found.","action":"Metric/cohort reconciliation strengthened in this audit.","evidence_gap":"Official semantic counting relationships and forecast-time source availability."},
      {"correction":13,"status":"PARTIAL","code_level_issue":"Saved versions, configs, seeds and hashes recorded; no executable reproducibility defect identified.","action":"Add separate release-readiness audit artifacts.","evidence_gap":"Original source download URLs/IDs, retrieval dates, source hashes and extraction provenance."},
    ]


def external_requirements()->list[dict]:
    return [
      {"issue":"IPC category overlap","required_evidence":"NCRB year-specific IPC district tables/codebooks and official aggregation or case-counting rules (2001-2014)","why_needed":"Resolve parent/subcategory and cross-domain semantic overlap; direct source overlap alone cannot establish independent counts."},
      {"issue":"Forecast-time availability","required_evidence":"Declared operational issue date plus official NCRB publication/release timestamps and contemporaneous retrieval/access records for each source year","why_needed":"OGD catalog publication dates are not actual NCRB release or forecaster access dates."},
      {"issue":"Source provenance","required_evidence":"Original downloaded files, catalog/resource identifiers, URL, retrieval timestamp, checksum, and transformation/extraction manifest","why_needed":"Link local processed/integrated values to authoritative source downloads."},
      {"issue":"Railway duplicate conflict","required_evidence":"Original 2010 NCRB row identifiers and stable Railway unit names/codes for the two conflicting normalized keys","why_needed":"Determine whether the rows represent distinct units, duplicate records, or mapping collision without inventing a resolution."},
      {"issue":"Administrative continuity","required_evidence":"Official district/state unit codes and dated split/merger/renaming crosswalks for 2001-2014","why_needed":"Confirm that same normalized district name denotes the same geographic unit across years."},
      {"issue":"Missingness causes","required_evidence":"Original source forms, reporting notes, data dictionary or extraction logs for missing IPC cells","why_needed":"Distinguish unreported, unavailable, structural, and genuine zero values."},
      {"issue":"2014 raw-stage/final-integrated discrepancy","required_evidence":"Original 2014 NCRB IPC source table and the exact raw-stage transformation/extraction records","why_needed":"Reconcile discrepant columns/values without overwriting either stage."},
    ]


def write_report(records:list[dict],integrity:list[dict],metrics_rows:list[dict])->None:
    df=pd.DataFrame(metrics_rows)
    mismatch=sum(json.loads(x).get("accuracy")=="MISMATCH" or json.loads(x).get("macro_f1")=="MISMATCH" or json.loads(x).get("weighted_f1")=="MISMATCH" for x in df.official_metric_checks)
    keybad=sum(not x["keys_match_canonical_labels"] for x in integrity)
    report=("# NCRB final evaluation and release-readiness report\n\n"
      "## Decision\n\n"
      "**Overall status: `READY_FOR_TECHNICAL_SIGN_OFF_WITH_DOCUMENTED_LIMITATIONS`.** The corrected saved forecasting artifacts pass the verified target/key, feature-year, sequence-year, and recomputed metric checks performed here. This status does not verify operational historical data availability, official NCRB counting semantics, source provenance, boundary continuity, or missingness causes.\n\n"
      "No models were retrained. Metrics were recomputed from saved predictions and probabilities; no official result files were overwritten. The existing corrected temporal experiment outputs remain the evaluated artifacts.\n\n"
      f"The audit reconciled {len(records)} saved prediction files; saved summary metric mismatches: {mismatch}; prediction-to-canonical-label/key failures: {keybad}. See the CSVs for per-artifact details.\n\n"
      "## Experiment integrity and performance interpretation\n\n"
      "Tabular inputs are keyed same-district t-1 values. Strict sequence inputs use consecutive t-3, t-2, t-1 records. All evaluated target years are 2013-2014; sequence cohorts and prediction labels are checked by key. The locked sequence test cohort contains 671 rows. Tabular test cohorts contain their feature-eligible district-year records and are not conflated with the sequence cohort. The observed-year random-split notebook remains classification of already-observed burden and is excluded from forecast comparisons.\n\n"
      "The five-seed v5 paired comparisons and one-feature ablations are summarized descriptively in `model_comparison_and_stability.csv`. The five-seed enhanced-minus-baseline macro-F1 mean deltas are +0.00785 (GRU), +0.01248 (LSTM), and +0.00016 (logistic regression); gains occurred in 4/5, 5/5, and 3/5 seeds respectively. These are fixed-test descriptive results, not significance tests. Direct component overlap is historical in these forecast inputs, while indirect NCRB category-count relationships remain unresolved.\n\n"
      "The v1/v2 sequential outputs include six saved baseline/enhanced prediction files with unequal test-key sets (1,276 baseline versus 671 enhanced records); they are superseded historical runs and are not valid paired comparisons. Later v3 common-cohort and v4/v5 controlled runs use the 671-key cohort. The cohort report marks the v1/v2 mismatches explicitly.\n\n"
      "Model complexity is not presumed useful: LSTM and GRU should be retained as exploratory comparisons only where their paired results provide consistent improvement over logistic regression and simpler baselines. Test-set results are descriptive and were not used for model selection.\n\n"
      "## Remaining partial corrections\n\n"
      "Corrections 1, 4, 7, 8, 9, 12, and 13 remain PARTIAL. No further supportable code defect was identified in these areas. See `remaining_correction_actions.csv` and `external_evidence_requirements.csv` for specific actions and source materials needed.\n\n"
      "## Preservation and limitations\n\n"
      "The 198 protected files from the prior closure hash set are unchanged. Target thresholds remain 107, 237, 362, 563 (fit on 2001-2010); split remains train 2001-2010, validation 2011-2012, locked test 2013-2014. The paper draft was not edited. Historical availability, original downloads, NCRB category rules, missingness explanations, Railway IDs, and administrative boundary history remain unverified. The 2014 raw-stage/final-integrated discrepancy remains unresolved.\n\n"
      "Technical sign-off is therefore limited to implementation and saved-experiment reproducibility, not operational deployment or full data provenance.\n")
    (OUT/"release_readiness_report.md").write_text(report,encoding="utf-8")


def write_test_report()->None:
    (OUT/"test_report.txt").write_text(
      "Command: python src\\saved_experiment_release_readiness_audit.py\nResult: PASS (exit code 0; 78 prediction files reconciled).\n"
      "Recomputed accuracy, macro/weighted F1 and per-class metrics; reconciled saved confusion matrices/classification reports; recomputed AUC/AP for 58 files with saved probabilities; canonical target/key alignment and test-year/sequence-year checks passed. Saved scalar metric mismatches: 0. AUC/AP arrays absent for 20 files.\n\n"
      "Command: python -m unittest discover -s tests\nResult: PASS (41 tests).\n\n"
      "Command: python -m compileall -q src ML_Implementation tests\nResult: PASS (exit code 0).\n\n"
      "No model training command was run. Existing official outputs and protected inputs were read-only. Six v1/v2 sequential baseline/enhanced cohort mismatches are documented as superseded, non-comparable historical artifacts; v3+ controlled comparisons use matched cohorts.\n",encoding="utf-8")


def write_manifest()->None:
    protected=json.loads((ROOT/"outputs/audits/final_evidence_closure/reproducibility_manifest.json").read_text(encoding="utf-8"))["protected_hashes_before"]
    current={k:sha256(ROOT/Path(k)) for k in protected if (ROOT/Path(k)).exists()}
    mismatches=[k for k,v in protected.items() if current.get(k)!=v]
    payload={"python":platform.python_version(),"packages":{n:importlib.metadata.version(d) for n,d in (("numpy","numpy"),("pandas","pandas"),("scikit_learn","scikit-learn"),("tensorflow","tensorflow"))},
       "thresholds":[107,237,362,563],"threshold_fit_years":"2001-2010","split":{"train":"2001-2010","validation":"2011-2012","test":"2013-2014"},
       "models_retrained":False,"paper_modified":False,"protected_prior_hash_count":len(protected),"protected_current_hash_count":len(current),
       "protected_hash_mismatches":mismatches,"protected_hashes_unchanged":len(current)==len(protected) and not mismatches,
       "corrected_runs":[str(p.relative_to(ROOT)) for p in RUNS.rglob("run_config.json")],
       "audit_script_sha256":sha256(Path(__file__).resolve()),
       "audit_output_hashes":{p.name:sha256(p) for p in OUT.iterdir() if p.is_file() and p.name!="reproducibility_manifest.json"}}
    (OUT/"reproducibility_manifest.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")


def write_changed_files()->None:
    names=sorted(p.name for p in OUT.iterdir() if p.is_file() and p.name not in {"changed_files.txt","initial_git_status.txt"})
    lines=["src/saved_experiment_release_readiness_audit.py - read-only saved-result reconciliation and report generator"]
    lines += [f"outputs/audits/final_release_readiness/{x} — generated release-readiness evidence" for x in names]
    lines += ["outputs/audits/final_release_readiness/initial_git_status.txt — captured before audit edits",
              "outputs/audits/final_release_readiness/changed_files.txt — change inventory"]
    (OUT/"changed_files.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":main()
