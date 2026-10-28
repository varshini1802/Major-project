# Computational Analysis of Crime and Social Systems Using Machine Learning (ML) and Deep Learning (DL)

*Revised Conference Paper Draft — Methodology and Results Sections Formally Updated in Accordance with Faculty Review*

---

## Abstract

Crime is an important social issue that affects the safety and well-being of individuals and communities. This project presents a computational analysis of crime and social systems using Machine Learning (ML) and Deep Learning (DL). The study uses crime data from Chicago and India (NCRB) to identify crime patterns, important features, and suitable classification models. The datasets were first inspected, cleaned, standardized, and integrated where required. Missing values, duplicate records, inconsistent formats, high dimensionality, class imbalance, and possible data leakage were carefully addressed. Exploratory Data Analysis (EDA) was then performed to study crime patterns based on time, location, crime type, and other relevant attributes.

For the Chicago dataset, the Domestic attribute was used to classify records into Domestic and Non-Domestic crimes. Different ML and DL models, including Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, MLP with embeddings, and FT-Transformer, were developed and compared. On the unseen 2025 test data, all final models achieved more than 82% accuracy, with FT-Transformer achieving the highest accuracy of 83.32%, while the MLP achieved the highest balanced accuracy, recall, F1-score, and ROC-AUC. For the NCRB dataset, crime variables were grouped into six major categories and a five-level violent-crime target was created. The enhanced MLP achieved the best overall performance with 90.76% accuracy and 90.76% F1-score. The results show that careful preprocessing, feature selection, feature engineering, leakage checking, and model comparison are important for reliable computational crime analysis.

*What does the study contribute? Why do researchers need to care?*

---

## 1. Introduction

Crime is a serious social issue that impacts personal and community safety and wellbeing. It may also have a negative impact on social and economic growth of a location. Crime doesn't occur in the same manner in all locations. It can be different when and where it occurs, the crime committed, and the social environment of the neighborhood. It is thus important to understand crime patterns when investigating the frequency of crime in certain locations and/or time periods.

Police agencies, government agencies and other public sources have been releasing a vast amount of crime data in recent years. These datasets include crime type, date, time, location and other crime incident information. This large amount of historical data provides an opportunity to study crime using computational methods. Past research has demonstrated that examining the spatial and temporal dynamics of crime has a role to play in understanding how crime varies from place to place and over time.

The application of Machine Learning (ML) and Deep Learning (DL) in crime analysis is seen to be growing since they can process a huge amount of data and can find patterns in the data. Various approaches are employed to predict and classify crimes like using Decision Tree, Random Forest, Support Vector Machine, Logistic Regression, XGBoost, LSTM, etc. in different studies. These methods have proven Machine Learning can be useful in uncovering crime patterns and predictions based on past crime data.

But using machine learning is not always simple when dealing with crime data. Poor quality and crime data availability is a key issue. In the real world, data sets can be incomplete, contain duplicate records, have inconsistent formats, or have little data. Some studies also mentioned data storage issues, system performance issues and availability of limited resources. Further, datasets from various sources could have different features and different forms, making it challenging to merge them into a more convenient dataset. The researchers have thus proposed the need for more effective data processing and fusion of data from multiple sources.

Another important question is that the crime record is not the only thing that impacts crime. Social and economic factors can also be related to crime patterns. Demographic and socio-economic data and crime data have been explored in some research, and in other research, the lack of additional data on many crime prediction systems has been noted. For instance, the Chicago COVID-19 study recommended that include other external factors, such as socio-economic factors, in future analysis to better understand the study.

Some limitations in previous studies are also noted in terms of the location and generalisation. A few studies are limited to one city or a short timeframe, making it challenging to determine if the same strategy will be effective in other cities. Researches with very detailed spatial and temporal information may also be plagued by issues of data imbalance, low precision or many false predictions. Another concern is privacy, as exact crime locations and personal information may have to be concealed or changed, potentially impacting the amount of detail available for analysis.

Another difficulty is comprehended results generated by the Machine Learning models. Some sophisticated models can provide acceptable predictions but it might not always be easy to determine why the model made a certain prediction. This is crucial in the field of crime for applications because misleading predictions could impact individuals and communities. Past studies have, therefore, stressed the need for models that can be more easily understood and explained.

The challenges highlight that crime analysis is not a single-directional endeavor, but entails the development of a prediction model. Good data quality is also crucial along with good data pre-processing, understanding crime patterns, selecting useful data features and carefully analysing models. Knowing more about the data prior to model building can assist in choosing appropriate features and models for the problem of interest. In this paper, computational analysis of crime and social systems using machine learning focuses on analyzing the crime data sets from Chicago, Philadelphia and India and learning the pattern of crime in each of the locations.

The datasets are first gathered and readied for analysis. Problems like duplicate records, inconsistent data, missing values are addressed during data preprocessing. An exploratory data analysis (EDA) is then performed on the crime pattern by location, time and crime type. Once the data is used in implementing various machine learning and deep learning models, the important features are identified. The attribute of the Chicago crime dataset that is the target variable is Domestic. The crime records are divided into two types: Domestic and Non-domestic. A variety of models are trained and evaluated on this classification task, to compare their performance.

This work distinguishes itself from other works that primarily focus on the future and the predictions of crime rather than on the entire process of crime analysis, which begins with data preparation and the understanding of data, and continues through to the selection of features and the evaluation of the model. The study will leverage crime data sets across various jurisdictions and examine crime-related and social-system data to give a more comprehensive perspective on computational crime analysis.

The primary objective of this work is to find out the pattern of the crime, which useful features are available and how to assess the Machine Learning models for classification of crimes. The results can be useful for providing data-based information for researchers and for subsequent research into social-system, crime analysis, and crime prediction. Some of the problems faced in previous research were data quality, limited data, there are discrepancies in data, the lack of social data, imbalance of data, privacy issues, and lack of explanation of model results. These challenges have been addressed in this work by preprocessing the data, exploratory analysis, feature selection, and a comparison of the various Machine Learning and Deep Learning models.

### Objectives
The main objectives of this study are:
1. To gather and combine relevant crime and social-system data sets from Chicago, Philadelphia and India for computational analysis of crime.
2. To pre-process and analyze the collected data sets, including managing missing data, duplicate data, inconsistent data format, etc., to address the data quality issues.
3. Conduct Exploratory Data Analysis (EDA) to look for spatial, temporal and crime type patterns within the crime data that is available.
4. To detect and choose suitable attributes that may be helpful for crime-related classification and analysis.
5. To build and to compare the Machine Learning and Deep Learning models for the selected classification problems of the crimes.
6. To assess the performance of the models developed through the use of appropriate evaluation metrics and to make the comparisons.
7. To gain data-driven knowledge from the crime analysis that could be used to better understand crime patterns at each location and in various social settings.

---

## 2. Literature Review

The increasing development of artificial intelligence and machine learning has led to a broader study in the field of crime analysis, crime forecasting and crime prediction. Recent studies have investigated different approaches ranging from traditional machine learning classification models to deep learning and spatio-temporal forecasting. The research goals, input parameter, spatio and temporal resolutions, models used and performance metrics of these studies differ widely and thus with methodological differences present the results are not directly comparable even when similar approaches are used. In recent years, the use of statistical methods, machine learning, and deep learning techniques to develop successful crime prediction models has been a significant research topic. Some studies in this area can be summarized as follows:

Albors Zumel et al. [4] conducted a study on fine-grained spatiotemporal crime forecasting by exploring, whether human mobility information provides an additional predictive value beyond historical crime and sociodemographic variables. The study consisted of crime incident data from Baltimore, Chicago, Los Angeles and Philadelphia from 2019-2023 along with American community survey sociodemographic data and Advan research mobility data. The data was represented using spatial cells of 0.2km² and 12 hour intervals. The approach was based on ConvLSTM which was trained to forecast crime occurrences 12 hours ahead using 2-day and 14-day input sequences and was compared to Logistic Regression, Random Forest and standard LSTM models. The authors found that the combination of crime mobility and sociodemographic features showed the best results with ConvLSTM achieving the highest recall, precision and F-1 score among all the evaluated approaches. The study highlighted the importance of contextual information for fine-grained crime forecasting. However the authors also faced challenges associated with crime sparsity and mobility coverage, which suggested that fine grained prediction may be challenging even when using deep learning approaches.

Kshatri et al. [12] developed an innovative crime detection model that uses big-data analytics and machine learning to classify violent crimes in India. The researches used the crime data published by the National Crime Records Bureau and applied data mining, cleaning, preprocessing, extracting, selecting relevant features, classifying, ensemble, and meta learning processes. The study evaluated different models like naive bayes, random tree, meta multi-class classifier and ensemble bagging classifier. In the results Meta multi-class classifier outranked all the other models with an accuracy of 96.667%, a ROC score of 0.96 and a precision of 0.966. The study illustrates the importance of systematic preprocessing and classification procedures for crime analysis. However the research scope is limited to analysing violent crimes and focusing only on the data provided by the National Crime Records Bureau. The authors also suggested that the future studies must explore additional approaches, including regression, clustering and consider other factors such as socioeconomic features.

İlgün and Dener [18] developed a comprehensive framework which combines Exploratory Data Analysis, crime-type prediction, time-series analysis and trend forecasting. The authors analyzed publicly available crime datasets of San Francisco, Chicago and Philadelphia to which they applied extensive preprocessing, feature engineering, categorical encoding, scaling and selection of relevant features. They applied eight different machine learning algorithms including XGBoost, CatBoost, Random Forest, Decision Tree, Multilayer Perceptron, K-Nearest Neighbor, Gaussian Naive Bayes and Logistic Regression for the crime-type prediction. Additionally, Long-Short Term Memory (LSTM) and Bidirectional LSTM networks were used for time-series analysis, while Holt–Winters Exponential Smoothing, Prophet, and SARIMA were chosen for statistical forecasting. For the Philadelphia crime dataset XGBoost algorithm demonstrated the best performance with 90.5% accuracy and 89.9% F1-score for crime-type prediction. The study is an important contribution to the field, as it provides the comprehensive methodology which can be used for analysis of different larger-scale urban crime datasets. However the authors also note that further research is needed which would involve larger recent datasets, broader exploration of hyperparameters, additional deep-learning algorithms, more up-to-date evaluation metrics and approaches and address the ethical concerns.

### Summary Table of Literature Review

| Ref. | Algorithms Used | Model / Task Applied | Dataset | Key Results | Limitations |
|---|---|---|---|---|---|
| [1] | SVM, NB, DT, FastText | Social-media security and crime | 1.7M+ Bogotá tweets (2019–2020), SIEDCO | SVM F1 = 0.57; Security perception related to robbery. | City- and language-specific study; not direct crime forecasting. |
| [2] | ConvBiLSTM, SVM, Logistic Regression, NAHC, DNN, BERT | Multimodal crime prediction | Chicago crime + 398K tweets | ConvBiLSTM: 97.75% accuracy | Limited to one city and one month; dependence on social-media data and tweet semantics limits generalizability. |
| [3] | DNN, HIN2V, RF, XGBoost, SVM | Street-level crime prediction | Beijing crime + OSM + POI data | HIN2V improved AUC, accuracy and MSE | Beijing only; static network |
| [4] | ConvLSTM, LR, RF, LSTM | spatiotemporal crime forecasting using crime, mobility and sociodemographic information | 4 U.S. cities + mobility + ACS | ConvLSTM performed best. | Sparsity, low precision, mobility coverage |
| [5] | TC under-reporting-aware model, UU, UU(C), IFG | Fairness-aware crime prediction considering under-reporting | Crime + ACS + mobility; 4 U.S. cities | Fairness improved in many cases | Under-reporting; fairness–accuracy trade-off |
| [6] | Logistic Regression, Decision Tree, Random Forest, XGBoost, RNN-LSTM | Crime prediction before, during and after COVID-19 for patrol planning | 2,124,602 Chicago crime records from 2015–2023 with 22 final attributes | RNN-LSTM: 92.74% accuracy | No socioeconomic features |
| [7] | KNN, Random Forest, SVM, Naive Bayes, CNN, LSTM | Crime-hotspot prediction using historical crime and built-environment features | Southeast China; 369 grids + POI/roads | LSTM achieved the strongest hotspot prediction performance. | Temporal window and grid-size limitations |
| [8] | DRL-CNA, Gradient Boosting Machine, Random Forest, SVM | Criminal-network hidden-link prediction using deep reinforcement learning | UCINET cocaine-smuggling datasets | DRL-CNA achieved highest AUC | Small network datasets; GPU dependence |
| [9] | K-Means, DBSCAN, DT, SVM, KNN, RF | Indian crime-pattern analysis | NCRB crime data | Scaled Algorithm; 78% accuracy | Requires larger datasets, broader comparison of ML and deep-learning approaches. |
| [10] | K-Means, MLP | Crime quantity prediction | Buenos Aires crime data | MLP validation MAE = 0.4095. | Older data; reported-crime dependence. |
| [11] | ANN, RF, KNN, DT, LSTM, CNN, SVM | Systematic ML crime review | 68 primary studies using government, law-enforcement, Kaggle, UCI, social-media, CCTV, traffic, weather and census datasets | ML most widely used; RF/ANN/KNN common | Data quality, imbalance, explainability |
| [12] | NB, Random Tree, Meta Classifier, Bagging | Violent-crime classification | NCRB; 5,038 records | Meta Multi-class Classifier achieved 96.667% accuracy, ROC = 0.96 and precision = 0.966. | Limited variables; clustering/regression needed |
| [13] | ML, CNN, RNN, LSTM, BERT, ST-GCN | Systematic ML/DL review | Multiple crime datasets | ML dominant; classification most common | Data, ethics, explainability, scalability |
| [14] | CNN, LSTM, GAN, Random Forest, SVM, STCN, SHAP, LIME | Cognitive Digital Twin hotspot detection | Crime + demographic + weather + sensor data | Hybrid models: 95–97% accuracy | Privacy, interpretability, changing patterns |
| [15] | KPI-AA, Eigenvector Centrality, GAT, GraphSAGE, GCCDC | Criminal-network analysis | Social/fraud network datasets | KPI-AA outperformed baselines | Scalability and hidden relationships |
| [16] | Statistical analysis; no ML prediction model | AI for crime reduction | 170 AI professionals | AI perceived as useful for crime reduction | Questionnaire-based; no prediction model long-term effects require further investigation. |
| [17] | Regression, Decision Tree, Neural Networks | Crime trends & hotspots | Denver crime + socioeconomic data | Trends/hotspots identified | Geographic limitation; bias and data quality |
| [18] | XGBoost, CatBoost, RF, DT, MLP, KNN, LSTM, BLSTM, SARIMA | Crime prediction & forecasting | SF, Chicago & Philadelphia | Philadelphia XGBoost: 90.5% accuracy | Larger/current data; optimization & XAI needed. |

---

## 3. Methodology (Revised Version)

This project was performed in a step-by-step manner, where the first step involved collecting and understanding the crime data, the second involved designing different machine learning and deep learning models, and the last step was to compare the results. The project analysed crime data collected from various sources, at both the local and district level, to identify crime patterns. A number of experiments were conducted during the development to improve the understanding of the data, selection of useful features, development of appropriate target variables and improving the model results. The high dimensionality, class imbalance, potential data leakage and large datasets were also tackled.

### 3.1 Data Collection
The first aspect of the project was to collect suitable crime data. The two main data sources used in the study were the Chicago Crimes dataset and the National Crime Records Bureau (NCRB), India dataset. The Chicago Crimes data is from the Chicago Data Portal. It contains information about reported crime in Chicago including crime type, location, date, time, beat, community area, arrest information, domestic information and other crime related information. From the original data set in Chicago, the raw data is very large, so the raw data is saved and processed data is created to analyze the data. The NCRB data was shared in the form of 76 CSV files containing different crime information shared in NCRB statistical tables. These encompassed IPC offences, crimes against Scheduled Tribes (ST), crimes against children, crimes against women, crime against property, arrests, crime information against Scheduled Castes (SC). The 76 files were first examined to understand the structure of the files, the columns and the data types, and the geographical information in the files and reporting periods.

### 3.2 Check Dataset and Standardize
Once collected, the next phase was to grasp the structure of the data sets, and to ascertain what information was available for analysis. For the Chicago dataset, the number of records, their structure, data type and missing values in the columns were checked. Common identifiers `YEAR`, `STATE` and `UNIT_NAME` were found to be the primary keys for the NCRB datasets. The files were incompatible as they were different in structure and reporting levels. Therefore, the meaning of their columns and their reporting level were extensively checked before merging. Of the 76 NCRB files, 14 District level files were used for district level crime analysis. Other data reporting frameworks that did not fit the chosen district-level framework were not included, such as arrest data, police personnel, court processes, and data on the characteristics of the victims.

### 3.3 Data Integration and Data Cleaning
The 14 datasets selected were categorised into five crime-data families namely IPC, SC, ST, Children, and Women. They were joined on the common keys: `YEAR`–`STATE`–`UNIT_NAME`. This resulted in 9,856 records per district-year and 344 columns of data. Data types and duplicate records, missing values, inconsistent formats and reporting gaps were then audited for the integrated data to ensure the data was clean. No missing value was assumed to be "0" because a missing value may have meant it was not reported rather than that zero crime occurred. A number of variables related to crime were used more than once in various source forms. Therefore, 18 duplicate representations were simplified to common features. The final NCRB data set was 9,856 observations and 331 variables. The data from Chicago was also pre-processed for missing data, duplicate data, unwanted columns and inconsistent formats.

### 3.4 Target Variable Creation and Tie-Breaking Resolution
Two different target variables were designed for the two crime-analysis tasks:
1. **Chicago Dataset**: The `Domestic` attribute was chosen as the target variable (`True` = Domestic, `False` = Non-Domestic), converted to a binary target: 1 (Domestic), 0 (Non-Domestic).
2. **NCRB Dataset**: To establish a 5-level classification of violent crime, all violent IPC offenses were combined into a district-year continuous measure called `VIOLENT_CRIME_BURDEN`. In early experiments, this burden was binned into five classes using a rank-based quantile method (`rank(method="first")`), which arbitrarily broke ties based on dataset row order. Following supervisor review, this arbitrary ordering dependency was resolved:
   - Four frozen numeric cutoffs were fitted exclusively on 2001–2010 training data: **107.0, 237.0, 362.0, and 563.0**.
   - A right-closed classification rule was implemented (`np.searchsorted(boundaries, burden, side="right") + 1`), ensuring that all observations with identical burdens receive the exact same class label deterministically:
     - **Class 1 (Lowest)**: $\text{Burden} \le 107$
     - **Class 2 (Low-to-Moderate)**: $107 < \text{Burden} \le 237$
     - **Class 3 (Moderate)**: $237 < \text{Burden} \le 362$
     - **Class 4 (Moderate-to-High)**: $362 < \text{Burden} \le 563$
     - **Class 5 (Highest)**: $\text{Burden} > 563$
   - Out of 9,856 integrated rows, 9,631 have valid target values and 225 rows with all-missing violent IPC components are preserved as missing without artificial target substitution.

### 3.5 Crime Grouping
The integrated NCRB dataset contained 331 detailed crime columns. To manage the high dimensionality, crime variables were grouped into six broad categories:
1. `GROUP_HOMICIDE`
2. `GROUP_SEXUAL_VIOLENCE`
3. `GROUP_KIDNAPPING_ABDUCTION`
4. `GROUP_ROBBERY_DACOITY`
5. `GROUP_PUBLIC_VIOLENCE`
6. `GROUP_PROPERTY_CRIME`

This reduced the modelling dataset to 10 columns: `YEAR`, `STATE`, `UNIT_NAME`, the six grouped crime features, and `TARGET_VIOLENT_LEVEL` (`data/processed/district_grouped_reduced.csv`).

### 3.6 Exploratory Data Analysis
Exploratory Data Analysis was performed on Chicago and NCRB datasets. For Chicago, EDA examined common crime types, hourly/monthly/daily patterns, arrest rates, domestic proportions (~18.4% Domestic), and community area spatial clusters. For NCRB, distributions of crime groups, temporal trends across states, missingness profiles, and target distributions across the five burden levels were audited.

### 3.7 Initial and Base Models
Initial base models established reference benchmarks:
- For Chicago, baseline linear classifiers were evaluated on the domestic classification task.
- For NCRB, traditional machine-learning models (Gaussian Naive Bayes, Extra Tree Classifier corresponding to Random Tree, and One-vs-Rest Gaussian Naive Bayes) were initially trained on contemporaneous six-group features. However, as noted below, contemporaneous classification creates severe target–predictor circularity, requiring a clear methodological separation between same-year classification and future-year forecasting.

### 3.8 Feature Selection
For Chicago, Mutual Information (MI) feature selection identified `Primary Type`, `Location Description`, and `Beat` as top predictors. For NCRB, dimensionality reduction was achieved through the six crime groups, forming the 6-feature baseline representation.

### 3.9 Feature Engineering for NCRB Data
Five engineered predictors were derived to capture aggregate volume, diversity, and temporal dynamics:
1. `TOTAL_GROUPED_CRIME`: Sum of the six grouped crime counts (`min_count=1`).
2. `ACTIVE_CRIME_GROUPS`: Number of non-zero crime groups in the district.
3. `CRIME_DIVERSITY_INDEX`: Normalized Shannon entropy across the six groups.
4. `PREV_YEAR_TOTAL_CRIME`: Total grouped crime for the *same district* from calendar year $t-1$, verified through keyed geographical lookups (`STATE` + `UNIT_NAME` + `YEAR`).
5. `YOY_TOTAL_CRIME_CHANGE`: Difference between current total and previous-year total:
$$\text{YOY\_TOTAL\_CRIME\_CHANGE} = \text{TOTAL\_GROUPED\_CRIME}_t - \text{PREV\_YEAR\_TOTAL\_CRIME}_{t-1}$$

Two primary feature sets were compared:
- **Baseline (6 features)**: The six grouped crime measures.
- **Enhanced (11 features)**: The six grouped measures + the five engineered features.

### 3.10 Data Leakage Checking & Resolution
A rigorous leakage audit identified that nine violent IPC source variables (`IPC_MURDER`, `IPC_RAPE`, `IPC_ROBBERY`, etc.) were shared between the target burden calculation and the six grouped predictors. Descriptively, these shared counts constituted **76.592%** of the total violent crime burden across valid-target rows. When contemporaneous (same-year) predictors are used, models achieve artificially inflated accuracy (e.g. 90.76%) by reconstructing the target from its own components.

To resolve target–predictor leakage and establish scientifically valid future forecasting:
1. **Temporal Horizon Enforcement**: For forecasting year $t$, tabular models are strictly restricted to predictors observed in year $t-1$.
2. **Sequence Eligibility**: Recurrent models use strictly consecutive prior years ($t-3, t-2, t-1$) for the *same district*. Same-year features are excluded entirely from forecasting matrices.
3. **Preprocessing Scope**: All scalers and imputers are fitted exclusively on training-period data. Validation-set and test-set distributions are never seen during preprocessing.

### 3.11 Chronological Data Splitting & Cohort Definition
Random train–test splits leak temporal trends and district-level constants. Therefore, strict chronological partitioning was enforced for the NCRB forecasting study:
- **Training Period**: 2001–2010 (10 calendar years)
- **Validation Period**: 2011–2012 (2 calendar years; used strictly for model selection and early stopping)
- **Locked Test Period**: 2013–2014 (2 calendar years; locked until final evaluation)

To ensure an exact, fair comparison between baseline (6 features) and enhanced (11 features) sequence models, the paired evaluation was restricted to a common cohort of **5,842 valid three-year sequences** (3,847 training, 1,324 validation, and 671 locked test instances). Missing historical records and nonfinite histories were rejected rather than synthetically imputed.

### 3.12 Machine-Learning Models
For NCRB forecasting, the following models were evaluated:
- **Logistic Regression**: Regularized linear baseline using keyed lag-1 features from $t-1$.
- **Majority-Class Baseline**: Predicts the most frequent class in the training data (Class 5), establishing the naive benchmark.

### 3.13 Deep-Learning Sequence Models
Recurrent sequence models were implemented to capture multi-year dependencies from 3-step historical sequences ($[t-3, t-2, t-1] \rightarrow \text{target}_t$):
- **Gated Recurrent Unit (GRU)**: 64 hidden units, dropout 0.2, Adam optimizer, sparse categorical cross-entropy loss.
- **Long Short-Term Memory (LSTM)**: 64 memory cells, dropout 0.2, Adam optimizer.
Epoch selection was governed by validation macro F1. Preprocessing parameters were refitted on Train + Validation prior to locked test inference.

### 3.14 Model Evaluation & Feature Ablation Protocol
Model performance on the multi-class (5-level) target was evaluated using:
- **Accuracy**: Proportion of correct classifications.
- **Macro F1**: Unweighted mean of class-wise F1 scores, treating all 5 burden classes equally regardless of class support.
- **Weighted F1**: Support-weighted mean F1 score.
- **Macro One-vs-Rest ROC-AUC & Average Precision (AP)**: Multi-class discrimination metrics computed across predicted class probability distributions.
- **Multi-Seed Robustness Protocol**: Evaluated across 5 fixed random seeds (11, 23, 37, 53, 71) to quantify optimization variability.
- **Single-Feature Ablation Protocol**: To assess individual feature utility, each of the 5 engineered features was added individually to the 6-group baseline across all 5 seeds (50 total model fits), measuring the paired delta in Macro F1 relative to the same-seed baseline.

### 3.15 Challenges Faced During the Project
Key challenges addressed include:
- Managing large datasets across heterogeneous urban and national crime repositories.
- Disentangling genuine zero crime counts from missing reporting periods without global zero-imputation.
- Eliminating target–predictor leakage by replacing contemporaneous models with keyed temporal lags.
- Guaranteeing deterministic tie-breaking on quantile target boundaries.

### 3.16 Final Methodology Workflow
The unified, verified experimental workflow is summarized below:
$$\text{Data Ingestion \& Cleaning} \rightarrow \text{District Key Standardization} \rightarrow \text{Frozen Training Quantiles (107, 237, 362, 563)} \rightarrow \text{Crime Grouping (6 Baseline Groups)} \rightarrow \text{Temporal Feature Engineering (5 Features)} \rightarrow \text{Keyed Lag \& Sequence Construction (t-3 to t-1)} \rightarrow \text{Chronological Split (2001-10 / 2011-12 / 2013-14)} \rightarrow \text{Validation Selection} \rightarrow \text{Locked Test Inference (671 Cases)} \rightarrow \text{5-Seed Ablation Study}$$

---

## 4. Results and Performance Analysis (Revised Version)

### 4.1 Exploratory Analysis of the Chicago Crime Dataset
Temporal, categorical, spatial and domestic-crime patterns were determined through exploratory analysis. The most commonly reported types of crime reported were Theft, Battery and Criminal Damage. The number of crimes reported varied from year to year, by month, day and hour, and there were definite peaks in the number of crimes reported on some days and at some times during the month. There was also spatial variation in crime with areas of concentration across community areas. The difference between domestic and non-domestic distribution was quite high, with approximately 81.6% of the records being non-domestic and 18.4% being domestic records. There were also certain geographical variables which were highly correlated such as beat and district.

### 4.2 Initial and Final Classification Results – Chicago Dataset
The first trials gave the following accuracy for the models: Logistic Regression - 35.25%, Random Forest - 34.66%, Linear SVM - 35.21%, Naive Bayes - 31.64% and KNN - 25.59%. These results were positive, and more pre-processing, feature selection, leakage checking, and pipeline refinement were done.

#### Table I. Initial Classification Performance (Chicago)
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 35.25% | 34.46% | 35.25% | 33.83% |
| Random Forest | 34.66% | 33.89% | 34.66% | 34.01% |
| Linear SVM | 35.21% | 34.44% | 35.21% | 33.10% |
| KNN | 25.59% | 25.47% | 25.59% | 24.83% |
| Naive Bayes | 31.64% | 30.59% | 31.64% | 30.13% |

After leakage checking and chronological splitting, the last set of models was tested on an unseen set of 2025 records of 236,927. All the seven models had an accuracy of > 82%. The highest Accuracy was achieved by FT-Transformer while MLP achieved the highest Balanced Accuracy, Recall, F1-score and ROC-AUC.

#### Table II. Final Classification Performance (Chicago 2025 Test Set)
| Rank | Model | Type | Accuracy | Balanced Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| 1 | MLP (Embeddings) | Deep Learning | 83.27% | 83.36% | 53.97% | 83.50% | 65.56% | 91.36% |
| 2 | FT-Transformer (DL) | Deep Learning | 83.32% | 83.20% | 54.08% | 83.01% | 65.49% | 91.27% |
| 3 | XGBoost | Machine Learning | 83.12% | 83.11% | 53.71% | 83.09% | 65.24% | 91.25% |
| 4 | Random Forest | Machine Learning | 83.04% | 83.01% | 53.57% | 82.95% | 65.10% | 91.04% |
| 5 | LightGBM | Machine Learning | 83.02% | 83.01% | 53.53% | 82.99% | 65.08% | 91.14% |
| 6 | HistGradientBoosting | Machine Learning | 82.96% | 82.99% | 53.42% | 83.03% | 65.01% | 91.13% |
| 7 | Logistic Regression | Machine Learning | 82.71% | 82.76% | 52.99% | 82.85% | 64.63% | 90.94% |

---

### 4.3 NCRB Violent-Crime Forecasting and Feature Evaluation Results

Following the supervisor's evaluation, all experimental results for the NCRB dataset were recomputed using the leakage-free chronological protocol. Same-year classification results are superseded by future-year forecasting on the locked 2013–2014 test set.

#### Target Class Distribution
Under the frozen training thresholds (107, 237, 362, 563), the class counts across partitions are detailed in Table III. The test support reflects the locked 671-row common cohort of districts with complete consecutive three-year histories.

#### Table III. Distribution of Violent-Crime Target Burden Classes
| Target Level | Burden Range | Full Dataset (9,631 rows) | Common Sequence Train (3,847 rows) | Common Sequence Validation (1,324 rows) | Locked Test Cohort (671 rows) |
|---|---|---|---|---|---|
| **Class 1 (Lowest)** | $\text{Burden} \le 107$ | 1,938 (20.12%) | 728 (18.92%) | 245 (18.50%) | 113 (16.84%) |
| **Class 2 (Low-to-Moderate)** | $107 < \text{Burden} \le 237$ | 1,824 (18.94%) | 758 (19.70%) | 204 (15.41%) | 76 (11.33%) |
| **Class 3 (Moderate)** | $237 < \text{Burden} \le 362$ | 1,831 (19.01%) | 794 (20.64%) | 225 (16.99%) | 105 (15.65%) |
| **Class 4 (Moderate-to-High)** | $362 < \text{Burden} \le 563$ | 1,968 (20.43%) | 787 (20.46%) | 310 (23.41%) | 145 (21.61%) |
| **Class 5 (Highest)** | $\text{Burden} > 563$ | 2,070 (21.49%) | 780 (20.28%) | 340 (25.68%) | 232 (34.58%) |

#### Controlled Forecasting Performance on Locked Test Cohort
Table IV reports the evaluation metrics for the controlled v4 models on the locked 671 test instances. All recurrent models substantially outperform the majority baseline (Macro F1 = 0.1028) and tabular logistic regression (Macro F1 = 0.6761).

#### Table IV. Controlled Forecasting Performance (Locked 2013–2014 Test Cohort, Seed 42)
| Feature Variant | Model Architecture | Test Accuracy | Macro F1 | Weighted F1 | Macro OvR ROC-AUC | Macro Mean OvR AP |
|---|---|---:|---:|---:|---:|---:|
| **Baseline (6 Groups)** | GRU | 0.7168 | 0.7020 | 0.7228 | 0.9375 | 0.7246 |
| **Enhanced (11 Features)** | GRU | 0.7198 | 0.7054 | 0.7250 | 0.9345 | 0.7133 |
| **Baseline (6 Groups)** | LSTM | **0.7228** | **0.7041** | **0.7275** | **0.9374** | 0.7209 |
| **Enhanced (11 Features)** | LSTM | 0.7198 | 0.6996 | 0.7240 | 0.9354 | **0.7244** |
| **Baseline (6 Groups)** | Logistic Regression | 0.6960 | 0.6761 | 0.7019 | 0.9327 | 0.7095 |
| **Enhanced (11 Features)** | Logistic Regression | 0.6975 | 0.6762 | 0.7026 | 0.9312 | 0.7059 |
| **Naive Benchmark** | Majority Classifier | 0.3458 | 0.1028 | 0.1777 | 0.5000 | 0.2000 |

*All metrics recomputed and verified independently against saved test predictions and probability matrices.*

#### Five-Seed Paired Robustness Analysis
To ensure findings are not artifacts of a single initialization, models were trained across 5 random seeds (11, 23, 37, 53, 71). Table V summarizes the paired enhanced-minus-baseline differences on the identical locked test cohort.

#### Table V. Five-Seed Paired Differences (Enhanced minus Baseline)
| Model | Metric Evaluated | Mean Paired Difference | SD Across Seeds | Seed Direction Counts |
|---|---|---:|---:|---|
| **GRU** | Test Macro F1 | +0.00785 | 0.00772 | Higher: 4, Lower: 1 |
| **GRU** | Test Accuracy | +0.00566 | 0.00581 | Higher: 4, Lower: 1 |
| **GRU** | Macro OvR ROC-AUC | −0.00099 | 0.00103 | Higher: 1, Lower: 4 |
| **LSTM** | Test Macro F1 | **+0.01248** | 0.00767 | Higher: 5, Lower: 0 |
| **LSTM** | Test Accuracy | **+0.01133** | 0.00478 | Higher: 5, Lower: 0 |
| **LSTM** | Macro OvR ROC-AUC | −0.00057 | 0.00185 | Higher: 3, Lower: 2 |
| **Logistic Regression** | Test Macro F1 | +0.00016 | 0.00000 | Higher: 5, Lower: 0 |

While LSTM exhibited a consistent positive average gain (+0.0125 Macro F1 across all 5 seeds), ROC-AUC was slightly lower, demonstrating that engineered features do not universally improve all performance dimensions.

#### Rigorous Feature Evaluation & Single-Feature Ablation Study
Rather than assuming feature expansion is always beneficial, a systematic ablation study was conducted. Each engineered feature was added individually to the 6-group baseline across 5 seeds (50 total model fits). Table VI reports the mean paired change in Macro F1 and across-seed variability.

#### Table VI. Single-Feature Ablation Results (Mean $\Delta$ Macro F1 Relative to 6-Group Baseline)
| One Engineered Feature Added | GRU Paired $\Delta$ Macro F1 (SD) | GRU Positive Seeds | LSTM Paired $\Delta$ Macro F1 (SD) | LSTM Positive Seeds |
|---|---:|---|---:|---|
| `ACTIVE_CRIME_GROUPS` | +0.00152 (0.00350) | 3 / 5 | +0.00446 (0.00571) | 4 / 5 |
| `CRIME_DIVERSITY_INDEX` | −0.00178 (0.00848) | 2 / 5 | **+0.00580** (0.00577) | **5 / 5** |
| `PREV_YEAR_TOTAL_CRIME` | −0.00106 (0.00580) | 3 / 5 | +0.00487 (0.00339) | **5 / 5** |
| `TOTAL_GROUPED_CRIME` | −0.00425 (0.00454) | 1 / 5 | +0.00186 (0.00322) | 3 / 5 |
| `YOY_TOTAL_CRIME_CHANGE`| −0.00511 (0.01408) | 2 / 5 | −0.00156 (0.00978) | 2 / 5 |

**Findings from Feature Evaluation**:
1. `CRIME_DIVERSITY_INDEX` (Shannon entropy across crime types) and `PREV_YEAR_TOTAL_CRIME` provided consistent positive improvements for LSTM (+0.0058 and +0.0049 Macro F1 in all 5 seeds), confirming that historical diversity and aggregate volume provide meaningful temporal context.
2. `TOTAL_GROUPED_CRIME` is largely redundant with the individual crime group inputs.
3. `YOY_TOTAL_CRIME_CHANGE` introduced significant noise and degraded performance across both GRU and LSTM models.
4. Feature selection should prioritize crime diversity and lagged totals while excluding unstable rate-of-change metrics.

#### Validation-Selected Model Performance
Among candidate architectures, Enhanced LSTM achieved the highest validation Macro F1 (0.7669) and was selected for per-class evaluation without inspecting test labels. Table VII details its performance on the locked 671 test cases.

#### Table VII. Per-Class Test Performance (Validation-Selected Enhanced LSTM, Seed 42)
| Violent Burden Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| **Class 1 (Lowest)** | 0.8629 | 0.9469 | 0.9030 | 113 |
| **Class 2 (Low-to-Moderate)** | 0.5354 | 0.6974 | 0.6057 | 76 |
| **Class 3 (Moderate)** | 0.5556 | 0.5714 | 0.5634 | 105 |
| **Class 4 (Moderate-to-High)** | 0.5946 | 0.6069 | 0.6007 | 145 |
| **Class 5 (Highest)** | 0.9115 | 0.7543 | 0.8255 | 232 |

The model achieves strong discrimination at the extremes (F1 = 0.9030 for Class 1 and 0.8255 for Class 5), while intermediate classes exhibit lower precision and recall due to minor boundary transitions across adjacent crime burdens.

---

## References

[1] Chaparro, L. F., Pulido, C., Rudas, J., Victorino, J., Reyes, A. M., Estrada, C., Narváez, L. Á., & Gómez, F. (2021). Quantifying perception of security through social media and its relationship with crime. *IEEE Access*, 9, 139201–139213. https://doi.org/10.1109/ACCESS.2021.3114675  
[2] Tam, S., & Tanrıöver, Ö. Ö. (2023). Multimodal deep learning crime prediction using tweets. *IEEE Access*, 11, 93204–93214. https://doi.org/10.1109/ACCESS.2023.3308967  
[3] Gu, H., Sui, J., & Chen, P. (2024). Graph representation learning for street-level crime prediction. *ISPRS International Journal of Geo-Information*, 13(7), 229. https://doi.org/10.3390/ijgi13070229  
[4] Albors Zumel, A., Tizzoni, M., & Campedelli, G. M. (2025). Deep learning for crime forecasting: The role of mobility at fine-grained spatiotemporal scales. *Journal of Quantitative Criminology*. https://doi.org/10.1007/s10940-025-09629-3  
[5] Wu, J., & Frias-Martinez, V. (2024). Improving the fairness of deep-learning, short-term crime prediction with under-reporting-aware models. *arXiv*. https://doi.org/10.48550/arXiv.2406.04382  
[6] Shafee, M. A., Shelke, N. A., Agrawal, R., & Choubey, C. K. (2026). Crime prediction before, during and after COVID-19 using machine learning and RNN-LSTM models. *Discover Artificial Intelligence*, 6, 875. https://doi.org/10.1007/s44163-026-02009-1  
[7] Zhang, X., Liu, L., Xiao, L., & Ji, J. (2020). Comparison of machine learning algorithms for predicting crime hotspots. *IEEE Access*, 8, 181302–181310. https://doi.org/10.1109/ACCESS.2020.3028420  
[8] Lim, M., Abdullah, A., & Jhanjhi, N. Z. (2021). Performance optimization of criminal network hidden link prediction model with deep reinforcement learning. *Journal of King Saud University—Computer and Information Sciences*, 33(10), 1202–1210. https://doi.org/10.1016/j.jksuci.2019.07.010  
[9] Bandekar, S. R., & Vijayalakshmi, C. (2020). Design and analysis of machine learning algorithms for the reduction of crime rates in India. *Procedia Computer Science*, 172, 122–127. https://doi.org/10.1016/j.procs.2020.05.018  
[10] Forradellas, R. F. R., Náñez Alonso, S. L., Jorge-Vazquez, J., & Rodriguez, M. L. (2021). Applied machine learning in social sciences: Neural networks and crime prediction. *Social Sciences*, 10(1), 4. https://doi.org/10.3390/socsci10010004  
[11] Jenga, K., Catal, C., & Kar, G. (2023). Machine learning in crime prediction. *Journal of Ambient Intelligence and Humanized Computing*, 14, 2887–2913. https://doi.org/10.1007/s12652-023-04530-y  
[12] Kshatri, S. S., Bhonsle, D., Verma, R., Pillai, A. G., & Moyal, V. (2022). Crime detection approach using big data analytics and machine learning. *NeuroQuantology*, 20(8), 1480–1495. https://doi.org/10.14704/nq.2022.20.8.NQ44162  
[13] Mandalapu, V., Elluri, L., Vyas, P., & Roy, N. (2023). Crime prediction using machine learning and deep learning: A systematic review and future directions. *IEEE Access*, 11, 60153–60170. https://doi.org/10.1109/ACCESS.2023.3286344  
[14] Shiraptini, T. C., Dhanushkodi, K., Sakthipriya, S., Akshaya, S., & Alphonse, S. (2026). AI and machine learning-enabled cognitive digital twin for crime hotspot detection and analysis. *IEEE Access*, 14, 16792–16801. https://doi.org/10.1109/ACCESS.2026.3658944  
[15] Cao, X., & Zhang, L. (2026). Social network analysis for crime prediction under social computing and deep learning technology. *Scientific Reports*, 16, 4769. https://doi.org/10.1038/s41598-025-34891-7  
[16] Tarawneh, H. H., Halalsheh, N. Z., Sulaiman, B. A., Alhajjaj, H. A., & Atieh, N. N. (2025). Artificial intelligence as a tool to enhance social interventions in reducing crime. *Frontiers in Artificial Intelligence*, 8, 1661266. https://doi.org/10.3389/frai.2025.1661266  
[17] Alhammadi, R. K. M. A. (2024). Analyzing crime patterns through artificial intelligence [Master's thesis, Rochester Institute of Technology]. *RIT Digital Institutional Repository*. https://repository.rit.edu/theses/11930  
[18] İlgün, E. G., & Dener, M. (2025). Exploratory data analysis, time series analysis, crime type prediction, and trend forecasting in crime data using machine learning, deep learning, and statistical methods. *Neural Computing and Applications*, 37, 11773–11798. https://doi.org/10.1007/s00521-025-11094-9  
