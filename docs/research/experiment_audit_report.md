# Research Project Experiment Audit Report

**Project Title:** Computational Analysis of Crime and Social System Using Machine Learning  
**Authors / Research Team:** Aravinda Valli (ENG23DS0077), Vaishnavi T.P (ENG23DS0089), Varshini B (ENG23DS0090)  
**Academic Guide:** Prof. Manasvi Bakshi, Department of Computer Science & Engineering (Data Science)  
**Date of Audit:** October 2026  
**Auditor / Review Role:** Senior ML Research Engineer, Experimental-Methodology Reviewer, and Academic Research-Paper Editor  
**Status:** Audit Complete; Implementation and Validation Verified  

---

## 1. Executive Summary

This audit examines the major research project *“Computational Analysis of Crime and Social System Using Machine Learning”*, cross-referencing:
1. The uploaded base paper: *“Crime Detection Approach Using Big Data Analytics and Machine Learning”* by Sapna Singh Kshatri et al. (*NeuroQuantology*, 2022, 20(8): 1480–1495).
2. The literature review papers referenced in the project synopsis and draft (including Albors Zumel et al. 2025, İlgün & Dener 2025, Shafee et al. 2026, Tam & Tanrıöver 2023).
3. The supervisor's critical evaluation report (`correction(2).md` / `source_correction_checklist_correction2_verbatim.md`).
4. The student manuscript draft (`draft paper.pdf`).
5. The complete repository state, commit history, processed datasets, scripts, audit outputs, and automated tests.

### Key Audit Findings
- **Disjoint Research Threads Across Jurisdictions**: The project originally planned to study Chicago, Philadelphia, and India (NCRB). In the repository, Philadelphia raw files were removed in earlier commits (leaving EDA artifacts and literature review references), Chicago was evaluated for binary `Domestic` crime classification on 2023–2025 data, and the primary research contribution was directed at an integrated district-year dataset from the National Crime Records Bureau (NCRB), India (2001–2014).
- **Core Methodological Flaws in the Initial NCRB Submission**:
  1. *Direct Target–Predictor Leakage*: The target `TARGET_VIOLENT_LEVEL` was constructed by summing violent IPC offenses to produce `VIOLENT_CRIME_BURDEN` and discretizing it into 5 bins. The model predictors included six aggregated crime groups from the *same district and same observation year*. Nine IPC source variables directly overlapped between target and predictors, accounting for 76.592% of the total target burden.
  2. *Row-Order Boundary Instability*: Quantile discretization via `rank(method="first")` and `pd.qcut()` arbitrarily assigned identical violent-crime burdens to different classes based solely on dataset row order.
  3. *Temporal Leakage from Random Splits*: Evaluating same-year crime data with an 80:20 stratified random train–test split leaked district-specific and temporal patterns, producing inflated accuracy (e.g. 90.76% for an enhanced MLP) that was incorrectly described as future crime forecasting.
  4. *Scaler Contamination*: Early MLP preprocessing fitted `StandardScaler` on the entire training split before separating validation data.
- **Controlled Baseline and Literature-Guided Enhancements**:
  A leakage-free chronological forecasting protocol was established: frozen training cutoffs (107, 237, 362, 563) on 2001–2010 training data; right-closed interval assignment; chronological split (Train 2001–2010, Validation 2011–2012, Locked Test 2013–2014); and strict district-keyed lag-1 features and 3-year sequences ($t-3, t-2, t-1$). Under this locked test cohort (671 instances), realistic macro-F1 is ~0.70 (GRU: 0.7020, LSTM: 0.7041, Logistic Regression: 0.6761, Majority Baseline: 0.1028).
- **Feature Evaluation Finding**:
  In a 5-seed multi-seed evaluation, adding 5 engineered temporal features yielded modest mean improvements (+0.0078 for GRU, +0.0125 for LSTM) but was not uniform across all metrics and seeds. Feature ablation revealed that features such as `ACTIVE_CRIME_GROUPS`, `CRIME_DIVERSITY_INDEX`, and `PREV_YEAR_TOTAL_CRIME` provided minor gains for LSTM, whereas `TOTAL_GROUPED_CRIME` and `YOY_TOTAL_CRIME_CHANGE` had negative or neutral paired differences. Rigorous feature evaluation, rather than inflated accuracy claims, forms the primary scientifically defensible contribution.

---

## 2. Project Architecture & Inventory of Major Files

| File / Folder Path | Type | Operational Responsibility & Verification Status |
|---|---|---|
| `data/processed/district_integrated_final.csv` | Processed Data | 9,856 rows, 331 columns; master integrated district-year panel (2001–2014) across 14 NCRB tables. |
| `data/processed/district_targeted.csv` | Processed Data | Adds canonical `VIOLENT_CRIME_BURDEN` and `TARGET_VIOLENT_LEVEL` (9,631 valid, 225 missing). |
| `data/processed/district_grouped_reduced.csv` | Processed Data | 10 columns: `YEAR`, `STATE`, `UNIT_NAME`, 6 crime groups, and target. |
| `data/processed/district_enhanced_features.csv` | Processed Data | Adds 5 engineered features (`TOTAL_GROUPED_CRIME`, `ACTIVE_CRIME_GROUPS`, `CRIME_DIVERSITY_INDEX`, `PREV_YEAR_TOTAL_CRIME`, `YOY_TOTAL_CRIME_CHANGE`). |
| `src/data_loading.py` | Python Module | Reusable data loading and schema verification. |
| `src/data_cleaning.py` | Python Module | Text trimming, upper-casing, duplicate checking, and numeric casting. |
| `src/target_creation.py` | Python Module | Frozen boundary assignment (107, 237, 362, 563) with right-closed intervals. |
| `src/target_creation_backup.py` | Protected Backup | Immutable reference copy of target logic. |
| `src/feature_engineering.py` | Python Module | Generates the 5 engineered predictors with district-year keyed lag matching. |
| `src/data_quality_pipeline_audit.py` | Audit Script | Validates missingness, zero counts, and pipeline sample progression. (Fixed syntax error). |
| `src/saved_experiment_release_readiness_audit.py` | Audit Script | Recomputes metrics from saved probabilities/predictions and checks locked test keys. |
| `src/multiseed_forecast_results_audit.py` | Audit Script | Checks 5-seed paired differences and verifies test set lock across seeds 11, 23, 37, 53, 71. |
| `src/multiseed_feature_ablation_audit.py` | Audit Script | Validates all 50 feature-ablation runs against same-seed baselines. |
| `ML_Implementation/forecast_feature_contract.py` | Contract Module | Enforces strict $t-1$ feature availability and forbids target-derived same-year columns. |
| `ML_Implementation/sequence_generation.py` | Module | Generates strict 3-year sequences ($t-3, t-2, t-1$) within intact district boundaries. |
| `ML_Implementation/sequence_cohort.py` | Module | Filters to the 5,842 common cohort eligible across both baseline and enhanced sets. |
| `ML_Implementation/controlled_sequence_forecast_evaluation.py` | Runner | Evaluates GRU, LSTM, and Logistic Regression under validation selection and locked test. |
| `ML_Implementation/plot_saved_paper_results.py` | Visualization | Generates publication figures using Matplotlib Agg backend from verified CSVs. |
| `tests/` | Unit Tests | 6 test modules (41 automated tests) covering target logic, sequence generation, leakage contracts, and reporting. |
| `notebooks/01_source_data_integration.ipynb` | Jupyter Notebook | Step-by-step schema integration across 14 NCRB tables. |
| `notebooks/02_crime_data_exploration.ipynb` | Jupyter Notebook | Exploratory data analysis, distributions, and spatial coverage. |
| `notebooks/observed_year_classification_models.ipynb` | Jupyter Notebook | Evaluates same-year observed burden classification (explicitly separated from forecasting). |
| `outputs/audits/` | Audit Tables & Notes | Machine-readable ledgers verifying data integrity, schema crosswalks, and hash protections. |

---

## 3. Dataset Provenance, Scope & Prediction Targets

### A. Chicago Crime Dataset
- **Source**: City of Chicago Data Portal (`Crimes - 2001 to Present`).
- **Observation Period in Paper**: 2023 to 2025 (2023–June 2024 train, July–Dec 2024 validation, 2025 unseen test).
- **Target**: `Domestic` (Binary: 1 = Domestic, 0 = Non-Domestic).
- **Role in Project**: Addressed in Sections 4.1 & 4.2 of the manuscript; evaluated with Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, MLP with embeddings, and FT-Transformer (all achieving >82% test accuracy).
- **Status**: Complete, intact, and preserved in the manuscript without changes.

### B. Philadelphia Crime Dataset
- **Source**: OpenDataPhilly (`Crime Incidents 2006–2022`).
- **Role in Project**: Explored during initial EDA (e.g. commit `67c0700` and literature review reference İlgün & Dener 2025). The student presentation referenced a planned comparison of 39 encoded features vs a 10-feature subset. However, raw Philadelphia data was removed from the active modelling tree in commit `b8492bd` when the team focused on NCRB. Philadelphia experiments are retained as literature and exploratory context, but do not form part of the primary experimental results in the manuscript.

### C. Indian NCRB Violent-Crime Dataset (Primary Focus)
- **Source**: National Crime Records Bureau (NCRB), Ministry of Home Affairs, Government of India (via Open Government Data Platform / Kaggle `rajanand/crime-in-india`).
- **Observation Period**: 2001–2014 across 35 States/UTs and hundreds of reporting districts.
- **Dimensions**:
  - Raw integrated panel: 9,856 district-year observations $\times$ 331 canonical crime attributes.
  - Valid target observations: 9,631 rows (225 rows lack valid target burden due to all-missing IPC components).
- **Target Definition**:
  - `VIOLENT_CRIME_BURDEN`: Row-wise sum of violent IPC crimes (`IPC_MURDER`, `IPC_ATTEMPT_TO_MURDER`, `IPC_CULPABLE_HOMICIDE_NOT_AMOUNTING_TO_MURDER`, `IPC_RAPE`, `IPC_KIDNAPPING_ABDUCTION`, `IPC_DACOITY`, `IPC_ROBBERY`, `IPC_ARSON`, `IPC_RIOTS`, etc.).
  - `TARGET_VIOLENT_LEVEL`: 5 ordered classes defined by frozen cutoffs (107, 237, 362, 563) computed exclusively on 2001–2010 training data:
    - **Class 1 (Lowest)**: $\text{Burden} \le 107$ (1,938 rows total; 728 train; 113 locked test).
    - **Class 2 (Low-to-Moderate)**: $107 < \text{Burden} \le 237$ (1,824 rows total; 758 train; 76 locked test).
    - **Class 3 (Moderate)**: $237 < \text{Burden} \le 362$ (1,831 rows total; 794 train; 105 locked test).
    - **Class 4 (Moderate-to-High)**: $362 < \text{Burden} \le 563$ (1,968 rows total; 787 train; 145 locked test).
    - **Class 5 (Highest)**: $\text{Burden} > 563$ (2,070 rows total; 780 train; 232 locked test).
- **Predictor Groups**:
  - Baseline (6 features): `GROUP_HOMICIDE`, `GROUP_SEXUAL_VIOLENCE`, `GROUP_KIDNAPPING_ABDUCTION`, `GROUP_ROBBERY_DACOITY`, `GROUP_PUBLIC_VIOLENCE`, `GROUP_PROPERTY_CRIME`.
  - Enhanced (11 features): Baseline + 5 engineered features (`TOTAL_GROUPED_CRIME`, `ACTIVE_CRIME_GROUPS`, `CRIME_DIVERSITY_INDEX`, `PREV_YEAR_TOTAL_CRIME`, `YOY_TOTAL_CRIME_CHANGE`).

---

## 4. Methodological Problems Identified & Corrections Applied

| Problem Area | Original Defect in Reviewed Project | Methodological Risk | Verified Correction Applied |
|---|---|---|---|
| **Target–Predictor Leakage** | Target `TARGET_VIOLENT_LEVEL` was derived from violent IPC counts, while predictors were 6 aggregated crime groups from the *same observation year*. | The model was circular: it reconstructed a label from the exact components given as input. 76.59% of target burden was directly in the inputs. | Strictly enforced temporal separation. For target year $t$, tabular models use only $t-1$ predictors. Recurrent sequence models use consecutive prior years ($t-3, t-2, t-1$). Same-year predictors are strictly barred from forecasting. |
| **Quantile Tie-Breaking** | Used `valid_burden.rank(method="first")` followed by `pd.qcut()`. | Arbitrary tie-breaking based on file row order. Districts with identical burden numbers received different target classes. | Removed row-order dependence. Replaced with frozen numeric thresholds [107, 237, 362, 563] using right-closed intervals (`side="right"`). Identical values receive identical classes deterministically. |
| **Threshold Fitting Scope** | Thresholds were computed over the entire dataset (2001–2014) in some early scripts. | Information leakage from test years into target definitions. | Thresholds were fitted exclusively on training years 2001–2010 and frozen permanently across all subsequent partitions. |
| **Evaluation Strategy** | Models were evaluated using an 80:20 random stratified train–test split. | Leaked future data into the past and mixed rows from the same district across train and test sets. High scores (90.76%) were invalid for forecasting claims. | Implemented strict chronological splitting: Train = 2001–2010, Validation = 2011–2012, Locked Test = 2013–2014. Evaluated on a locked test cohort of 671 district sequences. |
| **Preprocessing Contamination** | `StandardScaler` in early MLP notebooks was fitted before the train/validation split. | Scaler parameters incorporated validation set distribution. | Preprocessing scalers are fitted exclusively on training sequences during validation selection, and refitted on train+val only for final test inference. |
| **Sequence Continuity** | Early LSTM/GRU implementations reshaped tabular rows into 3D tensors without ensuring district continuity. | Sequences crossed geographical boundaries or skipped missing years, corrupting temporal learning. | Created strict keyed sequence builder requiring exactly 3 consecutive calendar years ($t-3, t-2, t-1$) for the *same district*. Incomplete histories are rejected. |
| **Test Set Peeking** | Hyperparameters and epochs were adjusted based on test set scores in early runs. | Overfitting to the test partition, invalidating reported generalization. | Validation partition (2011–2012) is used exclusively for model selection and early stopping. Test set (2013–2014) is locked and evaluated only once. |
| **Implementation Bugs** | Line 1 of `src/data_quality_pipeline_audit.py` had an unclosed docstring `""Final...`, causing a `SyntaxError` that broke tests. | Entire automated test suite failed to execute. | Fixed docstring to `"""Final...`. All 41 unit tests now compile and pass cleanly. |

---

## 5. Comparison with Base Paper (Kshatri et al., 2022)

| Dimension | Base Paper (Kshatri et al. 2022) | Our Corrected Major Project | Methodological Implication |
|---|---|---|---|
| **Dataset Source** | NCRB state/district statistics (2001–2018 / 2001–2015). | NCRB district-year panel (2001–2014) integrated across 14 tables. | Both use NCRB data, but schemas and available years differ. |
| **Sample Size** | **180 instances** in the study dataset. | **9,631 valid district-year records** (5,842 common sequences; 671 locked test rows). | Base paper sample is extremely small (likely state-level or selected districts); ours is a comprehensive nationwide district panel. |
| **Prediction Task** | Static classification of crime into 5 levels using MATLAB. | **Chronological forecasting** of violent burden level for year $t$ using prior years ($t-1$ or $t-3..t-1$). | Static classification cannot be directly compared to future-year forecasting. |
| **Models Evaluated** | Naive Bayes (67.22%), Random Tree (91.67%), Meta Multi-class Classifier (92.22%), Meta Multi Tree Classifier (96.67%). | Logistic Regression, Naive Bayes, Extra Tree (Random Tree counterpart), GRU, LSTM, Majority Baseline. | We implement both traditional ML and recurrent deep sequence models. |
| **Validation Protocol** | 80% train / 20% cross-validation (static random split). | Locked chronological split (Train 2001–2010, Val 2011–2012, Test 2013–2014). | Random cross-validation inflates accuracy on time-series panels; chronological splitting is scientifically defensible. |
| **Reported Performance** | Up to 96.67% accuracy for Meta Multi Tree. | ~70–72% accuracy and ~0.70 Macro F1 on locked test set. | Lower accuracy in our study is expected and methodologically sound, reflecting genuine future-year forecasting difficulty. |

---

## 6. Baseline & Literature-Guided Experimental Results

### A. Locked Test Set Results (v4 Controlled Sequence Evaluation, Seed 42)
*Evaluated on the locked 671-row test cohort (2013–2014).*

| Variant | Model | Accuracy | Macro F1 | Weighted F1 | Macro OvR ROC-AUC | Macro Mean OvR AP |
|---|---|---:|---:|---:|---:|---:|
| **Baseline (6 Groups)** | GRU | 0.7168 | 0.7020 | 0.7228 | 0.9375 | 0.7246 |
| **Enhanced (11 Features)** | GRU | 0.7198 | 0.7054 | 0.7250 | 0.9345 | 0.7133 |
| **Baseline (6 Groups)** | LSTM | 0.7228 | 0.7041 | 0.7275 | 0.9374 | 0.7209 |
| **Enhanced (11 Features)** | LSTM | 0.7198 | 0.6996 | 0.7240 | 0.9354 | 0.7244 |
| **Baseline (6 Groups)** | Logistic Regression | 0.6960 | 0.6761 | 0.7019 | 0.9327 | 0.7095 |
| **Enhanced (11 Features)** | Logistic Regression | 0.6975 | 0.6762 | 0.7026 | 0.9312 | 0.7059 |
| **Naive Baseline** | Majority Classifier | 0.3458 | 0.1028 | 0.1777 | 0.5000 | 0.2000 |

### B. 5-Seed Paired Robustness Comparison (Seeds 11, 23, 37, 53, 71)
*Mean difference = Enhanced minus Baseline on identical seed and test cohort.*

| Model | Metric | Mean Paired $\Delta$ | SD Across Seeds | Direction Counts | Interpretation |
|---|---|---:|---:|---|---|
| **GRU** | Test Macro F1 | +0.00785 | 0.00772 | Higher=4, Lower=1 | Modest average improvement; not uniform across all seeds. |
| **GRU** | Test Accuracy | +0.00566 | 0.00581 | Higher=4, Lower=1 | Slight accuracy gain. |
| **GRU** | Macro OvR ROC-AUC | −0.00099 | 0.00103 | Higher=1, Lower=4 | Probability ranking is slightly degraded. |
| **LSTM** | Test Macro F1 | +0.01248 | 0.00767 | Higher=5, Lower=0 | Positive in all 5 seeds. |
| **LSTM** | Test Accuracy | +0.01133 | 0.00478 | Higher=5, Lower=0 | Consistent ~1.1 percentage point gain. |
| **LSTM** | Macro OvR ROC-AUC | −0.00057 | 0.00185 | Higher=3, Lower=2 | Mixed discrimination effect. |
| **Logistic Regression** | Test Macro F1 | +0.00016 | 0.00000 | Higher=5, Lower=0 | Negligible linear impact. |

### C. Feature Evaluation & One-Feature Ablation Study (5 Seeds $\times$ 5 Features = 50 Fits)
*Mean change in Macro F1 when adding exactly ONE engineered feature to the 6-group baseline.*

| Engineered Feature Added | GRU Paired $\Delta$ Macro F1 (SD) | GRU Direction | LSTM Paired $\Delta$ Macro F1 (SD) | LSTM Direction |
|---|---:|---|---:|---|
| `ACTIVE_CRIME_GROUPS` | +0.00152 (0.00350) | Higher: 3/5 | +0.00446 (0.00571) | Higher: 4/5 |
| `CRIME_DIVERSITY_INDEX` | −0.00178 (0.00848) | Higher: 2/5 | +0.00580 (0.00577) | Higher: 5/5 |
| `PREV_YEAR_TOTAL_CRIME` | −0.00106 (0.00580) | Higher: 3/5 | +0.00487 (0.00339) | Higher: 5/5 |
| `TOTAL_GROUPED_CRIME` | −0.00425 (0.00454) | Higher: 1/5 | +0.00186 (0.00322) | Higher: 3/5 |
| `YOY_TOTAL_CRIME_CHANGE`| −0.00511 (0.01408) | Higher: 2/5 | −0.00156 (0.00978) | Higher: 2/5 |

**Feature Evaluation Interpretation**:
1. Non-linear sequence models (LSTM) benefit moderately from structural diversity metrics (`CRIME_DIVERSITY_INDEX`, +0.0058) and lagged volume (`PREV_YEAR_TOTAL_CRIME`, +0.0049).
2. Rate-of-change metrics (`YOY_TOTAL_CRIME_CHANGE`) introduce volatility and degrade performance across both GRU and LSTM.
3. Feature engineering provides only marginal, architecture-dependent improvements; feature count alone does not guarantee superior forecasting performance.

---

## 7. Status of Tests & Error Verification

- **Compilation Status**: `python -m compileall -q src ML_Implementation tests` $\rightarrow$ **PASS** (Zero syntax errors).
- **Automated Unit Tests**: `python -m unittest discover -s tests -v` $\rightarrow$ **41 Tests, 0 Errors, 0 Failures (100% PASS)**:
  - `test_feature_engineering_availability.py`: 4 tests (PASS).
  - `test_final_correction_reporting.py`: 3 tests (PASS).
  - `test_final_evidence_closure.py`: 11 tests (PASS).
  - `test_forecast_feature_contract.py`: 7 tests (PASS).
  - `test_sequence_generation.py`: 9 tests (PASS).
  - `test_target_creation.py`: 7 tests (PASS).
- **Core Audits**:
  - `src/saved_experiment_release_readiness_audit.py` $\rightarrow$ **PASS**.
  - `src/multiseed_forecast_results_audit.py` $\rightarrow$ **PASS**.
  - `src/multiseed_feature_ablation_audit.py` $\rightarrow$ **PASS** (all 50 fits verified).
  - `src/data_quality_pipeline_audit.py` $\rightarrow$ **PASS**.
  - `src/corrected_results_integrity_audit.py` $\rightarrow$ **PASS** (recomputed metric differences $\le 10^{-7}$).

---

## 8. Classification of Proposed Changes

### A. Essential Changes (Implemented Now)
1. Fix syntax error in `src/data_quality_pipeline_audit.py`.
2. Verify that all 41 unit tests and 5 audit scripts execute with clean exit codes.
3. Preserve the original paper draft (`draft paper.pdf`) as `docs/paper/original_paper_draft.md`.
4. Create the revised conference manuscript `docs/paper/revised_paper_draft.md` (and `outputs/reports/revised_conference_paper_draft.md`), updating *only* Methodology (Section 3) and Results (Section 4.3).
5. Document all legacy claims and unedited section discrepancies in `docs/paper/claims_and_inconsistencies_register.md`.
6. Consolidate the feature evaluation study summary in `outputs/tables/feature_evaluation_study_summary.csv`.
7. Generate the full reproducibility guide `docs/research/experiment_reproducibility.md`.
8. Update repository guides (`README.md`, `PROJECT_FILE_GUIDE.md`).

### B. Optional Research Improvements (Documented, Not Mandatory)
1. Exploration of gradient-boosted trees (e.g. LightGBM / CatBoost) for tabular lagged forecasting.
2. Incorporating spatial distance matrices or adjacency graphs across Indian districts (as suggested by ST-GCN literature).

### C. Future Work (Deferred per Teacher's Explicit Instructions)
1. Real-time crime prediction dashboards or web applications.
2. Dynamic demographic, socio-economic, and literacy rate integration (pending official census crosswalks).
3. Production deployment pipelines or automated continuous retraining architectures.
4. Upstream NCRB primary PDF / raw data extraction for post-2014 data.
