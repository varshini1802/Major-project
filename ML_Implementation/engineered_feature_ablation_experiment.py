"""One-at-a-time feature ablations on the v3 strict common sequence cohort."""
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v3"
INPUT = BASE / "inputs"
OUT = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_feature_ablation"
PAIRED = ROOT / "ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence"
KEYS = ["YEAR", "STATE", "UNIT_NAME"]
TARGET = "TARGET_VIOLENT_LEVEL"
TRAIN = list(range(2001, 2011)); VAL = [2011, 2012]; TEST = [2013, 2014]
SEED = 42


def scale_sequences(train, *other):
    nf = train.shape[-1]
    scaler = StandardScaler().fit(train.reshape(-1, nf))
    converted = [scaler.transform(train.reshape(-1, nf)).reshape(train.shape).astype(np.float32)]
    for value in other:
        converted.append(scaler.transform(value.reshape(-1, nf)).reshape(value.shape).astype(np.float32))
    return converted


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import Dense, Dropout, GRU, Input, LSTM
    from controlled_sequence_forecast_evaluation import save_evaluation

    random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
    meta = pd.read_csv(INPUT / "common_sequence_metadata.csv")
    meta.YEAR = meta.YEAR.astype(int); meta[TARGET] = meta[TARGET].astype(int)
    masks = {"train": meta.YEAR.isin(TRAIN).to_numpy(),
             "validation": meta.YEAR.isin(VAL).to_numpy(),
             "test": meta.YEAR.isin(TEST).to_numpy()}
    test_meta = meta.loc[masks["test"]].reset_index(drop=True)

    class SelectEpoch(Callback):
        def __init__(self, xval, yval):
            super().__init__(); self.xval=xval; self.yval=yval
            self.best=-1.; self.epoch=0; self.wait=0
        def on_epoch_end(self, epoch, logs=None):
            pred=np.argmax(self.model.predict(self.xval,verbose=0),axis=1)+1
            score=f1_score(self.yval,pred,average="macro",zero_division=0)
            if score>self.best: self.best=float(score); self.epoch=epoch+1; self.wait=0
            else:
                self.wait+=1
                if self.wait>=15: self.model.stop_training=True

    results=[]; selections=[]; OUT.mkdir(parents=True)
    for architecture, layer in (("GRU", GRU), ("LSTM", LSTM)):
        for variant in ("baseline", "enhanced"):
            archive=np.load(INPUT/f"{variant}_common_sequences.npz")
            X=archive["X"].astype(np.float32); names=archive["feature_names"].tolist()
            if not np.array_equal(archive["y"],meta[TARGET].to_numpy()):
                raise AssertionError(f"{variant}: labels do not match common metadata")
            y=archive["y"].astype(int)
            if variant=="baseline":
                continue
            engineered=names[6:]
            for added in engineered:
                selected=[names.index(name) for name in names[:6]+[added]]
                Xa=X[:,:,selected]
                xtr,xv,xte=(Xa[masks[p]] for p in ("train","validation","test"))
                ytr,yv,yte=(y[masks[p]] for p in ("train","validation","test"))
                (tr_s,val_s)=scale_sequences(xtr,xv)
                tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
                def create():
                    model=Sequential([Input(shape=(Xa.shape[1],Xa.shape[2])),layer(64),Dropout(.25),
                                      Dense(32,activation="relu"),Dense(5,activation="softmax")])
                    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
                    return model
                selector=SelectEpoch(val_s,yv); candidate=create()
                candidate.fit(tr_s,ytr-1,validation_data=(val_s,yv-1),epochs=100,batch_size=64,
                              callbacks=[selector],verbose=0)
                fit_x=np.concatenate([xtr,xv]); fit_y=np.concatenate([ytr,yv])
                fit_s,test_s=scale_sequences(fit_x,xte)
                tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
                model=create(); model.fit(fit_s,fit_y-1,epochs=selector.epoch,batch_size=64,verbose=0)
                probs=model.predict(test_s,verbose=0); pred=np.argmax(probs,axis=1)+1
                folder=OUT/added.lower()/architecture.lower(); folder.mkdir(parents=True)
                model.save(folder/f"{architecture.lower()}_model.keras")
                metrics=save_evaluation(folder,test_meta,yte,pred,probs)
                selections.append({"feature_added":added,"model":architecture,"best_epoch":selector.epoch,
                                   "validation_macro_f1":selector.best,"selection_metric":"validation macro-F1"})
                base_row=pd.read_csv(PAIRED/"test_results.csv")
                base_row=base_row[(base_row.variant=="baseline")&(base_row.model==architecture)].iloc[0]
                results.append({"feature_added":added,"model":architecture,"rows":len(yte),
                                "validation_macro_f1":selector.best,"selected_epoch":selector.epoch,
                                **metrics,"baseline_macro_f1":float(base_row.test_macro_f1),
                                "macro_f1_delta_vs_same_architecture_baseline":metrics["test_macro_f1"]-float(base_row.test_macro_f1),
                                "baseline_weighted_f1":float(base_row.test_weighted_f1),
                                "weighted_f1_delta_vs_same_architecture_baseline":metrics["test_weighted_f1"]-float(base_row.test_weighted_f1),
                                "feature_columns":"|".join(names[:6]+[added]),"seed":SEED})
    pd.DataFrame(results).to_csv(OUT/"test_results.csv",index=False)
    pd.DataFrame(selections).to_csv(OUT/"validation_selection.csv",index=False)
    config={"objective":"one-engineered-feature-at-a-time ablation compared with same-architecture baseline",
            "same_test_keys_source":"corrected_v3 common sequence cohort",
            "train_years":TRAIN,"validation_years":VAL,"test_years":TEST,"lookback":3,
            "thresholds":[107,237,362,563],"seed":SEED,"architectures":["GRU(64)","LSTM(64)"],
            "training":"Adam; sparse categorical crossentropy; dropout .25; dense 32; batch 64; max 100 epochs; patience 15 on validation macro-F1",
            "selection":"validation macro-F1 only; final model refit on train+validation at selected epoch",
            "test_used_for_selection":False}
    (OUT/"run_config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__=="__main__": main()
