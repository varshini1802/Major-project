# Experiment Reproducibility Guide

**Project:** Computational Analysis of Crime and Social System Using Machine Learning  
**Environment Specification:** Python 3.11.9, Scikit-learn 1.7.2, TensorFlow 2.18.1, Pandas 2.3.3, NumPy 2.0.2, Matplotlib 3.10.7  
**Operating System:** Windows 10/11 x64  
**Date:** October 2026  

---

## 1. Reproducibility Principles & Integrity Guarantees

This repository implements a strictly controlled, reproducible experimental pipeline for district-year crime forecasting and classification:
1. **Frozen Temporal Partitions**:
   - Training: 2001–2010 (10 calendar years)
   - Validation: 2011–2012 (2 calendar years)
   - Locked Test: 2013–2014 (2 calendar years)
2. **Frozen Target Thresholds**:
   - Boundary cutoffs: [107.0, 237.0, 362.0, 563.0] fitted exclusively on 2001–2010 training data.
   - Deterministic right-closed interval classification (`np.searchsorted(boundaries, burden, side="right") + 1`).
   - Tie-breaking is completely invariant to file row ordering.
3. **Leakage Prevention Contracts**:
   - For target year $t$, tabular models are restricted to district-keyed lag-1 features from $t-1$.
   - Recurrent models (GRU and LSTM) require strictly consecutive prior years ($t-3, t-2, t-1$) within intact district boundaries.
   - Same-year features are prohibited from entering forecasting matrices.
   - Preprocessing scalers are fitted on training data only (refitted on train+val only after model/epoch selection for locked test inference).
4. **Locked Test Cohort**:
   - The test set is permanently locked to 671 district-year sequences (`ML_Implementation/corrected_temporal_results/corrected_v4_controlled_sequence/locked_test_keys.csv`).
   - Test data was never used to tune hyperparameters, select epochs, or prune features.

---

## 2. Environment Setup

To ensure identical execution, use Python 3.11.9.

```powershell
# Clone or navigate to the repository root
cd d:\Major-project

# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Verify critical packages
python -c "import sklearn, tensorflow as tf, pandas as pd, numpy as np; print('sklearn:', sklearn.__version__, 'tf:', tf.__version__, 'pandas:', pd.__version__, 'numpy:', np.__version__)"
```

Required versions:
- `scikit-learn >= 1.7.0` (Installed: 1.7.2)
- `tensorflow >= 2.18.0` (Installed: 2.18.1)
- `pandas >= 2.2.0` (Installed: 2.3.3)
- `numpy >= 2.0.0` (Installed: 2.0.2)
- `matplotlib >= 3.10.0` (Installed: 3.10.7)

---

## 3. Fast Verification & Audit Commands (Read-Only)

These commands verify data integrity, test contracts, and recompute metrics without modifying datasets or refitting neural networks:

```powershell
# 1. Verify Python syntax and bytecode compilation across all modules
python -m compileall -q src ML_Implementation tests

# 2. Run the complete automated test suite (47 unit and integration tests)
python -m unittest discover -s tests -v

# 3. Verify saved experiment metrics and test cohort locks
python src/saved_experiment_release_readiness_audit.py
python src/multiseed_forecast_results_audit.py
python src/multiseed_feature_ablation_audit.py

# 4. Check data quality, missingness, duplicate keys, and source provenance
python src/data_quality_pipeline_audit.py
python src/duplicate_normalized_key_audit.py
python src/grouped_feature_missingness_audit.py
python src/grouped_predictor_source_audit.py
python src/forecast_feature_availability_audit.py
python src/corrected_results_integrity_audit.py

# 5. Regenerate publication-quality figures from verified CSVs (Matplotlib Agg)
python ML_Implementation/plot_saved_paper_results.py
```

---

## 4. End-to-End Experiment Execution Pipeline

If full model retraining is required (e.g. on new hardware or clean replication), run the pipeline in this strict sequence:

### Step 1: Target and Feature Construction
```powershell
# Recompute canonical target labels with frozen training thresholds
python src/target_creation.py

# Build engineered features from grouped crime data
python src/feature_engineering.py
```

### Step 2: Forecasting Input & Sequence Generation
```powershell
# Build strict temporal matrices and sequence cohort arrays
python ML_Implementation/prepare_forecasting_inputs.py
python ML_Implementation/generate_forecasting_sequences.py
```

### Step 3: Controlled Baseline & Multi-Seed Robustness Evaluation
```powershell
# Run the controlled 5-seed evaluation (Seeds: 11, 23, 37, 53, 71)
python ML_Implementation/run_multiseed_forecasting_experiment.py
```

### Step 4: Controlled One-Feature Ablation Study
```powershell
# Run 5-seed single-feature ablations (50 total model fits)
python ML_Implementation/run_multiseed_feature_ablation_experiment.py
```

### Step 5: Figure & Table Generation
```powershell
# Generate conference manuscript figures
python ML_Implementation/plot_saved_paper_results.py
```

---

## 5. Directory Layout & Artifact Provenance

| Directory | Content Description | Mutability |
|---|---|---|
| `data/processed/` | Master integrated, targeted, grouped, and enhanced district-year panels. | Read-Only in normal runs |
| `ML_Implementation/corrected_temporal_results/` | Immutable historical and versioned experiment outputs (`corrected_v4_controlled_sequence`, `corrected_v5_multiseed`, `corrected_v5_feature_ablation_multiseed`). | Read-Only Reference |
| `outputs/audits/` | Detailed machine-readable audit reports and evidence ledgers. | Generated / Verified |
| `outputs/figures/final_paper/` | Publication figures (`five_seed_macro_f1.png`, `five_seed_ablation_macro_f1.png`, `selected_v4_enhanced_lstm_confusion_matrix.png`). | Regenerable |
| `outputs/tables/` | Consolidated results (`feature_evaluation_study_summary.csv`, `thesis_ready_final_results.csv`, `v4_recomputed_metrics.csv`). | Audited Results |
| `docs/research/` | Research audit reports and reproducibility protocols. | Documentation |
| `docs/paper/` | Original draft and revised conference paper. | Manuscript Artifacts |

---

## 6. Known Constraints and Limitations

1. **Annual Time Granularity**: Data is recorded at annual resolution. While source data for year $t-1$ is structurally eligible before year $t$, official publication dates of NCRB reports vary historically.
2. **Administrative Boundary Adjustments**: District reorganizations over 2001–2014 mean not all districts have complete consecutive histories. Incomplete sequences are rejected to maintain methodological integrity.
3. **No Direct Operational Claim**: Results demonstrate structural predictability of next-year violent crime burden from historical reports; they do not claim operational real-time deployment readiness.
