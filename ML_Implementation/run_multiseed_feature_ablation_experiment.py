"""Run five-feature GRU/LSTM one-at-a-time ablations for v5 seeds.

Uses the existing v3 common sequence inputs, whose entire locked test-key file
is checked against v4 before any fit. Writes to a fresh v5 ablation directory.
"""
from __future__ import annotations

import contextlib
import importlib.metadata
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3/inputs"
V5 = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v5_multiseed"
V4 = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence"
OUT = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v5_feature_ablation_multiseed"
SEEDS = (11, 23, 37, 53, 71)
KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TARGET = "TARGET_VIOLENT_LEVEL"
TRAIN = list(range(2001, 2011))
VALIDATION = [2011, 2012]
TEST = [2013, 2014]


def scale(train: np.ndarray, *others: np.ndarray):
    nf = train.shape[-1]
    scaler = StandardScaler().fit(train.reshape(-1, nf))
    result = [scaler.transform(train.reshape(-1, nf)).reshape(train.shape).astype(np.float32)]
    result += [scaler.transform(x.reshape(-1, nf)).reshape(x.shape).astype(np.float32) for x in others]
    return result


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import Dense, Dropout, GRU, Input, LSTM

    meta = pd.read_csv(INPUT / "common_sequence_metadata.csv")
    meta.YEAR = meta.YEAR.astype(int)
    meta[TARGET] = meta[TARGET].astype(int)
    test_meta = meta.loc[meta.YEAR.isin(TEST), KEYS + [TARGET, "INPUT_YEARS"]].reset_index(drop=True)
    locked = pd.read_csv(V4 / "locked_test_keys.csv")
    pd.testing.assert_frame_equal(test_meta, locked, check_dtype=False)
    masks = {"train": meta.YEAR.isin(TRAIN).to_numpy(),
             "validation": meta.YEAR.isin(VALIDATION).to_numpy(),
             "test": meta.YEAR.isin(TEST).to_numpy()}

    enhanced = np.load(INPUT / "enhanced_common_sequences.npz")
    x_all = enhanced["X"].astype(np.float32)
    names = enhanced["feature_names"].tolist()
    if not np.array_equal(enhanced["y"].astype(int), meta[TARGET].to_numpy()):
        raise AssertionError("Enhanced sequence labels differ from metadata")
    if len(names) <= 6:
        raise AssertionError(f"Expected engineered predictors after six base groups, got {names}")
    engineered = names[6:]
    if len(engineered) != 5:
        raise AssertionError(f"Expected exactly five engineered features, got {engineered}")

    class SelectEpoch(Callback):
        def __init__(self, xval, yval):
            super().__init__(); self.xval=xval; self.yval=yval; self.best=-1.; self.epoch=0; self.wait=0
        def on_epoch_end(self, epoch, logs=None):
            pred=np.argmax(self.model.predict(self.xval,verbose=0),axis=1)+1
            score=f1_score(self.yval,pred,average="macro",zero_division=0)
            if score>self.best:
                self.best=float(score); self.epoch=epoch+1; self.wait=0
            else:
                self.wait+=1
                if self.wait>=15: self.model.stop_training=True

    rows=[]; selections=[]; status=[]
    OUT.mkdir(parents=True)
    for seed in SEEDS:
        seed_dir=OUT/f"seed_{seed}"; seed_dir.mkdir()
        random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
        baseline=pd.read_csv(V5/f"seed_{seed}"/"test_results.csv")
        for architecture, layer in (("GRU",GRU),("LSTM",LSTM)):
            base=float(baseline.loc[(baseline.variant=="baseline") & (baseline.model==architecture),"test_macro_f1"].iloc[0])
            for added in engineered:
                try:
                    selected=[names.index(n) for n in names[:6]+[added]]
                    xa=x_all[:,:,selected]; y=meta[TARGET].to_numpy()
                    xtr,xv,xte=(xa[masks[p]] for p in ("train","validation","test"))
                    ytr,yv,yte=(y[masks[p]] for p in ("train","validation","test"))
                    tr_s,val_s=scale(xtr,xv)
                    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
                    def create():
                        model=Sequential([Input(shape=(xa.shape[1],xa.shape[2])),layer(64),Dropout(.25),Dense(32,activation="relu"),Dense(5,activation="softmax")])
                        model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
                        return model
                    selector=SelectEpoch(val_s,yv); candidate=create()
                    candidate.fit(tr_s,ytr-1,validation_data=(val_s,yv-1),epochs=100,batch_size=64,callbacks=[selector],verbose=0)
                    fit_x=np.concatenate([xtr,xv]); fit_y=np.concatenate([ytr,yv])
                    fit_s,test_s=scale(fit_x,xte)
                    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
                    model=create(); model.fit(fit_s,fit_y-1,epochs=selector.epoch,batch_size=64,verbose=0)
                    pred=np.argmax(model.predict(test_s,verbose=0),axis=1)+1
                    score=float(f1_score(yte,pred,average="macro",zero_division=0))
                    folder=seed_dir/added.lower()/architecture.lower(); folder.mkdir(parents=True)
                    model.save(folder/f"{architecture.lower()}_model.keras")
                    predictions=test_meta.copy(); predictions["PREDICTED_TARGET"]=pred
                    predictions.to_csv(folder/"test_predictions.csv",index=False)
                    rows.append({"seed":seed,"feature_added":added,"model":architecture,"test_rows":len(yte),"test_macro_f1":score,"baseline_macro_f1_same_seed":base,"paired_macro_f1_difference":score-base,"selected_epoch":selector.epoch,"validation_macro_f1":selector.best,"test_used_for_selection":False,"feature_columns":"|".join(names[:6]+[added])})
                    selections.append({"seed":seed,"feature_added":added,"model":architecture,"selected_epoch":selector.epoch,"validation_macro_f1":selector.best})
                    status.append({"seed":seed,"feature_added":added,"model":architecture,"status":"COMPLETE","detail":"test keys match locked v4 cohort"})
                    print(f"seed={seed} {architecture} + {added}: macro_f1={score:.6f} delta={score-base:+.6f}",flush=True)
                except Exception as exc:
                    status.append({"seed":seed,"feature_added":added,"model":architecture,"status":"FAILED","detail":f"{type(exc).__name__}: {exc}"})
                    traceback.print_exc()
        pd.DataFrame(rows).to_csv(OUT/"per_seed_ablation_metrics.csv",index=False)
        pd.DataFrame(selections).to_csv(OUT/"validation_selection.csv",index=False)
        pd.DataFrame(status).to_csv(OUT/"run_status.csv",index=False)

    frame=pd.DataFrame(rows)
    if len(frame):
        frame.groupby(["feature_added","model"]).paired_macro_f1_difference.agg(seeds="count",mean_paired_difference="mean",sd_across_seeds="std").reset_index().to_csv(OUT/"paired_difference_summary.csv",index=False)
    packages={}
    for package in ("numpy","pandas","scikit-learn","tensorflow"):
        try: packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: packages[package]="not installed"
    config={"predeclared_seeds":list(SEEDS),"train_years":TRAIN,"validation_years":VALIDATION,"locked_test_years":TEST,"thresholds":[107,237,362,563],"locked_test_rows":len(locked),"input":"corrected_v3 common enhanced sequences; entire test key/label/input-year table exactly equals corrected_v4 locked cohort","architectures":["GRU(64)","LSTM(64)"],"ablation":"six grouped features plus exactly one of five engineered features; paired with same-seed same-architecture baseline macro-F1","selection":"validation macro-F1 only; train+validation refit at selected epoch; locked test not used for selection","preprocessing":"StandardScaler fit on training sequences for selection and separately on train+validation for final test","max_epochs":100,"batch_size":64,"patience":15,"tensorflow_seed_control":"Python random, NumPy, tf.keras.utils.set_random_seed","packages":packages,"completed_runs":len(frame),"expected_runs":50,"uncertainty":"seed standard deviation is descriptive optimizer variability, not sampling uncertainty or statistical significance"}
    (OUT/"run_config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    print(f"Completed {len(frame)}/50 feature-ablation fits. Results: {OUT}",flush=True)


if __name__=="__main__": main()
