# NCRB Crime Classification Research Paper: Required Corrections

## Purpose

The objective of this review is to correct the methodological, implementation, and evaluation issues in the NCRB crime-classification pipeline before the research paper is finalized.

The priority is not to increase model accuracy. It is to establish that the target is meaningful, the predictors are appropriate, and the evaluation supports the claims made in the paper.

The corrections below identify the relevant code sections, explain the issues, and specify the changes required.

---

## 1. Critical: Resolve Target–Predictor Overlap

**Files to highlight:**
- `src/target_creation.py`, lines 63–34 and 136–280. Specifically, inspect the `BASE_PAPER_GROUPS` definition and the `create_target()` function.
- `src/feature_engineering.py`, lines 80–108.
- `basepaper_prediction_models.ipynb`, cells 1–3, where predictors and the target are selected.

**Problem:**

The target `TARGET_VIOLENT_LEVEL` is constructed by calculating `VIOLENT_CRIME_BURDEN` from underlying IPC crime counts and dividing the burden into five quantile-based classes.

However, the modelling notebook uses six aggregated crime-group counts from the same district-year observations as predictors.

This creates a potential target–predictor dependency: the model may be reconstructing a label derived from the crime counts it receives as input.

Excluding the target column from the feature matrix is not sufficient to eliminate this problem.

**Required corrections:**

1. Map every IPC source variable used in target construction to the six aggregated predictor groups.
2. Identify which source variables overlap and quantify how much of the target burden can be reconstructed from the predictors.
3. Determine whether the intended task is:
   - Classification of an already-observed year's violent-crime burden; or
   - Forecasting of violent-crime burden in a future year.
4. If forecasting is the objective, construct the target for year `t+1` and use only information available by year `t`.
5. Rebuild the feature matrix accordingly.
6. Retrain every model after correcting the target–predictor relationship.

**Important:** Do not claim that leakage has been completely eliminated merely because `TARGET_VIOLENT_LEVEL` is absent from the feature matrix.

The target-generation rule and the availability of every predictor at the intended prediction time must be examined.

**Expected outcome:** A clearly defined prediction task in which the reported metrics measure the intended capability.

---

## 2. Correct the Quantile-Based Target Construction

**File to highlight:**
- `src/target_creation.py`, lines 229–280.
- Specifically, highlight `rank(method="first")` and the subsequent `pd.qcut()` operation.

**Problem:**

The current implementation ranks observations using:

```python
ranked_values = valid_burden.rank(method="first")
```

It then divides the ranked observations into five approximately equal-sized classes.

Using `method="first"` breaks ties according to observation order. Consequently, identical violent-crime burdens can receive different labels if they fall on opposite sides of a class boundary.

The dataset audit identified repeated burden values that span multiple target classes. This needs to be corrected because the target is intended to represent ordered levels of violent-crime burden.

**Required corrections:**

1. Calculate the number of unique valid burden values.
2. Count how many burden values occur more than once.
3. Identify repeated values assigned to different classes.
4. Remove the arbitrary dependence on observation order.
5. Evaluate fixed, documented burden thresholds or quantile boundaries that preserve tied values.
6. Document the rationale for the selected thresholds.
7. Recalculate the class distribution and regenerate the target.
8. Rerun all experiments using the corrected labels.

Do not force equal class sizes if doing so requires assigning different labels to observations with identical burdens.

**Expected outcome:** A reproducible target definition that preserves the interpretation of each class.

---

## 3. Clarify What the Five Classes Represent

**Files to highlight:**
- `src/target_creation.py`, lines 8–34 and 229–280.
- `basepaper_prediction_models.ipynb`, cells that describe or display the target classes.
- The paper's methodology section describing target construction.

**Problem:**

The target-generation code defines five levels of violent-crime burden, but parts of the modelling notebook describe the classes as individual crime types.

These interpretations are not equivalent.

A quantile-based burden classification produces ordered severity levels, not five mutually exclusive crime categories.

The source code also acknowledges that the target is a project-defined adaptation rather than a verified reproduction of the original paper's target-generation procedure.

**Required corrections:**

1. Define all five classes consistently:
   - Class 1: Lowest violent-crime burden.
   - Class 2: Low-to-moderate burden.
   - Class 3: Moderate burden.
   - Class 4: Moderate-to-high burden.
   - Class 5: Highest violent-crime burden.
2. Use these definitions consistently in the code, figures, tables, and manuscript.
3. Explicitly state that the target-generation method is a project-defined adaptation.
4. Do not claim to reproduce the original paper's target-generation procedure unless the original procedure can be independently verified.
5. Document the variables, aggregation rules, missing-value handling, and class boundaries.

**Expected outcome:** An unambiguous target definition that readers can independently reproduce.

---

## 4. Reconsider the Five Engineered Features

**File to highlight:**
- `src/feature_engineering.py`, lines 80–108.
- `src/feature_engineering.py`, lines 170–319.
- `src/feature_engineering.py`, lines 320 onward, covering previous-year totals and year-over-year changes.

**Features under review:**

1. `TOTAL_GROUPED_CRIME`
2. `ACTIVE_CRIME_GROUPS`
3. `CRIME_DIVERSITY_INDEX`
4. `PREV_YEAR_TOTAL_CRIME`
5. `YOY_TOTAL_CRIME_CHANGE`

**Problem:**

Three features are directly derived from current-year crime counts. The year-over-year change also incorporates the current year's total.

These features may provide information that closely reconstructs the target because the target is itself derived from crime counts.

Adding more features does not automatically improve the scientific validity of the experiment.

**Required corrections:**

1. Document the formula and source variables for every engineered feature.
2. Classify each feature according to when it becomes available:
   - Before the prediction year.
   - During the prediction year.
   - Only after the outcome has been observed.
3. Exclude features unavailable at the intended prediction time.
4. For a forecasting experiment, use historical features such as previous-year totals, provided they are correctly aligned.
5. Verify that previous-year features never incorporate current-year or future-year observations.
6. Compare baseline and enhanced models using identical data partitions and evaluation procedures.
7. Report whether each engineered feature improves performance on the untouched test set.

**Expected outcome:** A feature set justified by the prediction task rather than by its ability to reconstruct the target.

---

## 5. Replace Random Evaluation with Chronological Evaluation When Forecasting

**File to highlight:**
- `basepaper_prediction_models.ipynb`, cells 3 and 11.
- Highlight the `train_test_split()` calls.

**Problem:**

The notebook uses a stratified random train-test split:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

This preserves the approximate class distribution across the partitions, but it does not establish generalization to future years.

Observations from the same district may appear in both training and testing sets.

If the paper claims to forecast future crime levels, this evaluation is insufficient.

**Required corrections:**

1. Sort observations chronologically.
2. Establish a training period using earlier years.
3. Establish a validation period using subsequent years.
4. Reserve the latest suitable years as the final test period.
5. Ensure that target construction and feature engineering do not use future information.
6. Fit preprocessing transformations using training data only.
7. Keep the final test set untouched until model selection is complete.
8. Report the exact years used in each partition.

For example, a possible chronological design is:

- Training: 2001–2009.
- Validation: 2010–2011.
- Testing: 2012–2014.

These years are illustrative. The final split must be selected after inspecting the available district-year observations and ensuring that the chosen prediction horizon is appropriate.

**Expected outcome:** An evaluation that supports claims about future-year generalization.

---

## 6. Correct the MLP Validation Preprocessing

**File to highlight:**
- `basepaper_prediction_models.ipynb`, cells 38–42.
- Highlight the scaler's `fit_transform()` call and the subsequent training-validation split.

**Problem:**

The notebook fits the scaler on the training partition before dividing that partition into training and validation subsets.

As a result, the scaling parameters incorporate information from the eventual validation subset.

This is validation-set contamination during preprocessing.

**Required corrections:**

1. Divide the training partition into training and validation subsets first.
2. Fit `StandardScaler` exclusively on the training subset.
3. Transform the validation subset using the fitted scaler.
4. Transform the final test set using the same fitted scaler.
5. Apply the same principle to any imputation, feature selection, dimensionality reduction, or other learned preprocessing operations.
6. Ensure that the test set is not used to select the architecture, learning rate, epoch count, or other hyperparameters.

**Expected outcome:** A clean separation between training, validation, and test information.

---

## 7. Verify the Correctness of the Previous-Year Features

**File to highlight:**
- `src/feature_engineering.py`, lines 320 onward.
- `src/leakage_audit.py`, lines 242 onward.

**Problem:**

Historical features are potentially appropriate for forecasting, but only if their temporal alignment is correct.

A previous-year feature must refer to the same district's observations from the preceding year, not merely the preceding row in the dataset.

Districts may have missing years, inconsistent names, or different reporting coverage.

**Required corrections:**

1. Sort observations by district and year before calculating historical features.
2. Group by the appropriate geographical identifiers.
3. Confirm that the previous-year value corresponds to year `t-1`.
4. Verify that no district's history is accidentally assigned to another district.
5. Investigate missing years and duplicated district-year keys.
6. Verify the year-over-year change formula and its handling of zero denominators.
7. Do not silently treat missing historical observations as zero.
8. Create automated tests for temporal alignment.

**Expected outcome:** Historical features that use only the intended district's past observations.

---

## 8. Validate Dataset Integration and District-Year Keys

**Files to highlight:**
- `notebooks/01_Common_Schema_Integration.ipynb`
- `notebooks/02_India_Crime_Final_EDA.ipynb`
- `src/data_cleaning.py`
- `src/data_loading.py`

**Problem:**

The final modelling dataset depends on the correct integration of multiple NCRB datasets.

Incorrect joins, inconsistent district identifiers, duplicate keys, or mismatched reporting years can introduce incorrect observations.

A clean-looking final table does not by itself establish that the original integration was correct.

**Required corrections:**

1. Document the source and meaning of every integrated dataset.
2. Verify that district names and state identifiers are standardized consistently.
3. Confirm that `YEAR`, `STATE`, and `UNIT_NAME` uniquely identify the intended district-year observation.
4. Check duplicate keys before and after every merge.
5. Record the number of rows before and after each join.
6. Inspect unmatched observations and report how they are handled.
7. Validate the treatment of renamed, split, or reorganized administrative units.
8. Verify that overlapping crime categories are not accidentally double-counted.
9. Reconcile the final row count with the source datasets.

**Expected outcome:** A reproducible integration pipeline with documented row counts, key constraints, and missing-data decisions.

---

## 9. Review Missing-Value Handling

**Files to highlight:**
- `src/data_cleaning.py`
- `src/feature_engineering.py`, particularly the numeric preparation and aggregation functions.
- `basepaper_prediction_models.ipynb`, cells 1–3 and the preprocessing cells.

**Problem:**

Crime-reporting datasets can contain missing values for several different reasons, including unavailable data, incomplete reporting, and genuine zero counts.

These situations should not automatically be treated as equivalent.

The feature-engineering code also sums available values in some operations. If some crime groups are missing, the resulting total may represent only a partial count.

**Required corrections:**

1. Distinguish genuine zero counts from missing or unavailable observations.
2. Document how missing values are handled in the original crime variables.
3. Report the number of observations excluded because their targets are missing.
4. Investigate whether missingness differs by year, state, district, or crime group.
5. Avoid globally replacing missing values with zero without justification.
6. If imputation is necessary, fit the imputation procedure on training data only.
7. Verify that partially observed totals are not interpreted as complete totals without qualification.
8. Report the final number of usable observations and explain exclusions.

**Expected outcome:** A transparent missing-data strategy that does not silently alter the meaning of crime counts.

---

## 10. Validate the Meaning of the Classification Metrics

**File to highlight:**
- `basepaper_prediction_models.ipynb`, cell 50 and the associated evaluation cells.

**Problem:**

The notebook reports approximately 90.76% accuracy for the enhanced MLP, alongside ROC-AUC and PR-AUC values.

These metrics do not independently establish that the model solves a meaningful forecasting problem.

The interpretation depends on the target definition, feature availability, class construction, and evaluation protocol.

**Required corrections:**

1. Recalculate every metric after correcting the target and evaluation design.
2. Report accuracy and macro-F1 alongside weighted-F1.
3. Include precision, recall, and F1-score for every class.
4. Include a confusion matrix.
5. Explain whether ROC-AUC and PR-AUC are macro-averaged, micro-averaged, weighted, or calculated using another multiclass convention.
6. State whether the metrics are calculated on validation data or the untouched test set.
7. Compare against a majority-class baseline and an appropriate simple baseline.
8. Report the class distribution in each partition.
9. Explain the practical significance of errors between adjacent burden levels versus errors across distant levels.
10. Avoid comparing scores from experiments that use different targets, splits, or evaluation procedures.

**Expected outcome:** Performance results that are reproducible and interpretable.

---

## 11. Reassess the Need for LSTM and GRU Models

**Files to highlight:**
- `src/deep_learning_models.py`, lines 62–142.
- `basepaper_prediction_models.ipynb`, the cells responsible for constructing sequences and training the LSTM and GRU models.

**Problem:**

LSTM and GRU architectures are intended to learn from ordered sequences. Their use is justified only if the observations are prepared as meaningful temporal sequences.

Simply reshaping tabular observations into three-dimensional tensors does not create valid time-series data.

**Required corrections:**

1. Define the sequence length and prediction horizon.
2. Group observations by district before constructing sequences.
3. Sort every district's observations chronologically.
4. Ensure that sequences do not cross district boundaries.
5. Ensure that future observations do not enter historical input sequences.
6. Handle missing years explicitly.
7. Construct training, validation, and test sequences without temporal leakage.
8. Compare LSTM and GRU against simpler baselines under the same evaluation protocol.
9. Report the amount of training data and the number of valid sequences.
10. Retain these models only if the research design and empirical results justify their inclusion.

**Expected outcome:** A defensible comparison of temporal models rather than a comparison based solely on model architecture.

---

## 12. Improve the Leakage Audit

**File to highlight:**
- `src/leakage_audit.py`, particularly lines 132–239 and 242 onward.

**Problem:**

The existing audit checks for direct target copies, duplicate keys, and some feature-construction inconsistencies.

These checks are useful but insufficient to establish that the experiment is free from target leakage.

A feature does not need to be an exact copy of the target to reveal information that makes the classification task circular.

**Required corrections:**

1. Create a documented list of all source variables used in target construction.
2. Trace those variables through the aggregation and feature-engineering pipeline.
3. Identify direct and indirect overlap between target inputs and model predictors.
4. Check whether any predictor is unavailable at the intended prediction time.
5. Verify the temporal alignment of every historical feature.
6. Fit learned preprocessing operations on training data only.
7. Add automated assertions for the expected feature set.
8. Document the limitations of the leakage audit.

**Expected outcome:** An audit that examines the actual target-generation process, not only whether the target column appears in the feature matrix.

---

## 13. Improve Reproducibility

**Files to highlight:**
- `basepaper_prediction_models.ipynb`
- `src/deep_learning_models.py`
- `requirements.txt`
- The scripts responsible for generating the processed datasets.

**Problem:**

The paper's results should be reproducible from the source data, preprocessing pipeline, target-generation procedure, model configuration, and evaluation code.

Notebook outputs alone are insufficient to establish reproducibility.

**Required corrections:**

1. Document the Python and major library versions.
2. Record all random seeds.
3. Document model architectures and hyperparameters.
4. Record the exact train, validation, and test periods.
5. Save the target definition and class boundaries.
6. Save the feature list used by each experiment.
7. Make dataset-generation scripts reproducible.
8. Ensure that reported metrics can be regenerated from the evaluation code.
9. Record the final sample count at each major pipeline stage.
10. Clearly distinguish results from the original baseline and results from the proposed modifications.

**Expected outcome:** A reproducible experiment whose results can be independently checked.

---

# Required Changes to the Research Paper

## 14. Abstract

Revise the abstract after rerunning the experiments.

It must accurately describe:

- The research problem.
- The target definition.
- The prediction horizon.
- The dataset and observation period.
- The evaluation protocol.
- The corrected experimental results.
- The main contribution and limitations.

Do not retain the existing 90.76% accuracy as evidence of forecasting performance unless the revised experiment supports that interpretation.

## 15. Introduction and Research Objectives

Clearly distinguish between classifying observed crime burden and forecasting future crime burden.

State the research question precisely.

Avoid claims that the system predicts future crime levels if the experiment only classifies contemporaneous crime data.

## 16. Dataset and Preprocessing

Document:

- The source datasets and their reporting periods.
- The geographical and temporal unit of analysis.
- The integration procedure.
- The final observation count.
- Missing-value handling.
- Duplicate-key checks.
- Crime-group definitions.
- Target construction and class boundaries.

Explain the distinction between a missing count and a genuine zero count.

## 17. Methodology

Explain the complete pipeline in a reproducible order:

1. Dataset integration.
2. Data cleaning.
3. Crime-group aggregation.
4. Target construction.
5. Feature engineering.
6. Train-validation-test partitioning.
7. Preprocessing.
8. Model training and selection.
9. Final evaluation.

Explicitly discuss how target–predictor overlap is prevented or limited.

Document which features are available at prediction time.

## 18. Experimental Results

Regenerate all tables and figures after implementing the corrections.

Include:

- Class distribution.
- Baseline results.
- Enhanced-model results.
- Per-class precision, recall, and F1-score.
- Confusion matrix.
- Appropriate multiclass ROC-AUC and PR-AUC.
- Results from the final chronological test period, if forecasting is the objective.

Do not compare old results with corrected results as if they came from the same experiment.

## 19. Discussion

Interpret the results in relation to the research question.

Discuss:

- Whether the models generalize to the intended observations.
- Which classes are difficult to distinguish.
- Whether historical features provide a meaningful improvement.
- The limitations of district-level reported crime counts.
- The limitations of the target definition.
- The implications of missing or inconsistent reporting.
- Whether the evidence supports the claims made in the paper.

Avoid equating high classification accuracy with causal explanation or proof that a model can predict crime before it occurs.

## 20. Limitations and Future Work

Explicitly acknowledge the limitations that remain after the corrections.

Potential topics include:

- Dependence on reported crime counts.
- Missing or inconsistent historical observations.
- Changes in administrative boundaries and reporting practices.
- Limitations of the chosen burden thresholds.
- Temporal generalization.
- The difference between recorded crime and actual crime incidence.

Only include limitations that are relevant to the final dataset and methodology.

---

# Final Verification Checklist

Before submitting the paper, verify every item below.

- [ ] The target is clearly defined and reproducible.
- [ ] Identical burden values are not arbitrarily assigned different classes.
- [ ] The target-generation variables have been mapped against all model predictors.
- [ ] The prediction task and prediction horizon are explicit.
- [ ] Every predictor is available at the intended prediction time.
- [ ] Historical features are aligned to the correct district and year.
- [ ] Dataset joins and district-year keys have been validated.
- [ ] Missing values and genuine zero counts are distinguished.
- [ ] Preprocessing is fitted on training data only.
- [ ] The final evaluation set is not used for model selection.
- [ ] A chronological evaluation is used if forecasting is claimed.
- [ ] LSTM and GRU sequences are constructed correctly if these models are retained.
- [ ] All models are evaluated using the same appropriate protocol.
- [ ] The results have been regenerated after correcting the methodology.
- [ ] All tables and figures match the regenerated results.
- [ ] The abstract, methodology, discussion, and conclusions agree with the final experiments.
- [ ] The code, configuration, and data-processing steps are documented sufficiently for reproduction.

## Final Recommendation

The students should correct the target definition and prediction design before tuning the models further.

The primary objective is to establish what the model is actually learning and whether the evaluation supports the paper's claims.

A lower accuracy from a methodologically sound experiment is more defensible than a high accuracy obtained from a circular target–predictor relationship or an evaluation protocol that does not measure the intended predictive capability.