# Computational Analysis of Crime and Social Systems Using Machine Learning (ML) and Deep Learning (DL)

*Original Draft Transcribed Verbatim from `draft paper.pdf` for Archival and Comparative Provenance*

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

## 3. Methodology (Original Version)

This project was performed in a step-by-step manner, where the first step involved collecting and understanding the crime data, the second involved designing different machine learning and deep learning models, and the last step was to compare the results. The project analysed crime data collected from various sources, at both the local and district level, to identify crime patterns. A number of experiments were conducted during the development to improve the understanding of the data, selection of useful features, development of appropriate target variables and improving the model results. The high dimensionality, class imbalance, potential data leakage and large datasets were also tackled.

### 3.1 Data Collection
The first aspect of the project was to collect suitable crime data. The two main data sources used in the study were, Chicago Crimes dataset and National Crime Records Bureau (NCRB), India dataset. The Chicago Crimes data is from the Chicago Data Portal. It contains information about reported crime in Chicago including crime type, location, date, time, beat, community area, arrest information, domestic information and other crime related information. From the original data set in Chicago, the raw data is very large, so the raw data is saved and processed data is created to analyze the data. The NCRB data was shared in the form of 76 CSV files containing different crime information shared in NCRB statistical tables. These encompassed IPC offences, crimes against Scheduled Tribes (ST), crimes against children, crimes against women, crime against property, arrests, crime information against Scheduled Castes (SC). The 76 files were first examined to understand the structure of the files, the columns and the data types, and the geographical information in the files and reporting periods.

### 3.2 Check Dataset and Standardize
Once collected, the next phase was to grasp the structure of the data sets, and to ascertain what information was available for analysis. For the Chicago dataset, the number of records, their structure, data type and missing values in the columns were checked. During the initial experiments, there were several notebooks created to gain a better understanding of the data set and to investigate classification possibilities. Common identifiers YEAR, STATE and UNIT_NAME were found to be the primary keys for the NCRB datasets. The files were incompatible as they were different in structure and reporting levels. Therefore, the meaning of their columns and their reporting level were extensively checked before merging. Of the 76 NCRB files, 14 District level files were used for district level crime analysis. Other data reporting frameworks, that did not fit the chosen district-level framework, were not included, such as arrest data, police personnel, court processes, and data on the characteristics of the victims.

### 3.3 Data Integration and Data Cleaning
The 14 datasets selected were categorised into five crime-data families namely IPC, SC, ST, Children, and Women. They were joined on the common keys: YEAR–STATE–UNIT_NAME. This resulted in 9,856 records per district-year and 344 columns of data. Data types and duplicate records, missing values, inconsistent formats and reporting gaps were then audited for the integrated data to ensure the data was clean. However, no value was assumed to be “0” because a missing value may have meant it was not reported and not that there was no crime. A number of variables related to crime were used more than once in various source forms. Therefore, 18 copies of the source representations were simplified to common features. The final NCRB data set was 9856 observations and 331 variables. The data from Chicago was also pre-processed for missing data, duplicate data, unwanted columns and inconsistent formats. The following columns were not relevant for prediction and were removed.

### 3.4 Target Variable Creation
Two different target variables were designed for the two crime-analysis tasks. The Chicago Crimes dataset included the Domestic attribute chosen for the dataset. Domestic was the existing attribute for the Chicago Crimes dataset that was used as the target attribute. Records that were represented by the value of `True` were classified as Domestic and records that were represented by the value of `False` were classified as Non-Domestic. The next step was to convert the Target to a binary file: 1 (Domestic), 0 (Non-Domestic). In the initial experiments, the Domestic information was also used to understand distribution of the two classes and in the event of the latter class (the Crime information), apply machine learning. There was no direct target to be used for the desired 5-level classification of Violent Crime in the case of NCRB data. Thus a new project specific target TARGET_VIOLENT_LEVEL was added. All variables related to violent crime were combined into a single variable at the District-year level called VIOLENT_CRIME_BURDEN. These values were then quantized into five bins of equal size using a rank-based method. A rank-based method of quantizing these values into five equal-sized quantiles was used. The five groups ranked according to the severity of violent-crime burden (1 = lowest to 5 = highest). This resulted in a close to even 5 class target. This was considered an adaptation for a specific project as it was difficult to determine what the target generation process was in the selected base paper.

### 3.5 Crime Grouping
The integrated NCRB dataset consisted of a large number of crime variables at a detailed level. All such variables would be used directly and would result in a large feature space that's hard to control. Thus, Crime variables were grouped into six general categories of crime. The six groups consisted of: 1. Homicide 2. Sexual Violence 3. Kidnapping and Abduction 4. Robbery and Dacoity 5. Public Violence 6. Property Crime. The detailed IPC (criminal acts related to food and water security), SC (criminal acts involving sexual offences), women-related (criminal acts against women) and children-related (criminal acts against children) crime variables were grouped in the following broad categories. The detailed variables were then removed from the reduced modelling data set. This resulted in fewer features and preserved the main crime features that were used to classify the crime. This NCRB modelling dataset was now a 10 column dataset consisting of the following: YEAR, STATE, UNIT_NAME, six sets of crime features, and TARGET_VIOLENT_LEVEL. It contained 9,856 records. There were 225 records in the validation without any valid target or all six grouped crime values were missing.

### 3.6 Exploratory Data Analysis
After cleaning the data, Exploratory Data Analysis (EDA) was performed to understand the crime patterns.
For the Chicago dataset, different graphs were created to study:
- Most common crime types
- Monthly crime patterns
- Crimes by day of the week
- Crimes by hour
- Arrest rates
- Domestic and Non-Domestic crime distribution
- Areas with high crime counts
- Spatial crime patterns
- Correlations between variables
The EDA helped identify important patterns in the data and helped in deciding which features could be useful for prediction.
For the NCRB dataset, the grouped crime variables were studied at the district and year levels. The target distribution, missing values, crime-group distributions, and relationships between crime groups were also checked before modelling.

### 3.7 Initial and Base Models
Before developing the final models, initial or base models were created to understand how well the available crime features could predict the target.
For the Chicago dataset, different experiments were performed using the available crime features. These early models provided a basic reference for comparing later improvements.
For the NCRB dataset, traditional machine-learning models were developed using the six grouped crime features as the baseline representation.
These initial experiments showed that simply using a large number of features does not always result in better model performance. Therefore, feature selection and feature engineering were performed before developing the final models.

### 3.8 Feature Selection
For the Chicago crime prediction task, different feature-selection methods were tested to identify the most useful features for predicting Domestic crime.
Mutual Information was first used to understand the relationship between different features and the target. After this, features were added one by one and their effect on validation performance was checked.
After comparing the results, Primary Type, Location Description, and Beat were selected as the final important features. Adding these features improved the validation performance, while adding many additional features provided only a small improvement.
For the NCRB data, feature reduction was performed through crime grouping. Additional feature engineering was then carried out to check whether historical and aggregate crime information could improve the results.

### 3.9 Feature Engineering for NCRB Data
Five additional features were created from the six grouped crime variables:
1. **Total Grouped Crime**: TOTAL_GROUPED_CRIME represents the total number of crimes across the six crime groups. It is calculated by adding the six grouped crime values. If all six source values are missing, the value remains missing instead of being treated as zero.
2. **Active Crime Groups**: ACTIVE_CRIME_GROUPS represents the number of crime groups having a crime count greater than zero. If all six values are missing, the value is kept as missing.
3. **Crime Diversity Index**: CRIME_DIVERSITY_INDEX measures how evenly crimes are distributed across the six crime groups using Shannon entropy. A higher value means crime is more evenly distributed across different groups, while a lower value means crime is concentrated in fewer groups.
4. **Previous-Year Total Crime**: PREV_YEAR_TOTAL_CRIME provides historical information by taking the total grouped crime from the actual previous calendar year for the same state and district. The previous year was matched using STATE + UNIT_NAME + YEAR.
5. **Year-over-Year Crime Change**: YOY_TOTAL_CRIME_CHANGE represents the change in total grouped crime between the current year and the previous year:
$$\text{YOY\_TOTAL\_CRIME\_CHANGE} = \text{TOTAL\_GROUPED\_CRIME} - \text{PREV\_YEAR\_TOTAL\_CRIME}$$
This helps identify whether the overall crime burden increased or decreased compared with the previous year.

Two feature configurations were then used:
- **Baseline**: Six grouped crime features
- **Enhanced**: Six grouped crime features + five engineered features

The engineered features were checked for possible leakage because some of them use current-year crime information that is conceptually related to the target. The previous-year feature was considered comparatively safer because it uses information from the actual previous year.

### 3.10 Data Leakage Checking
Data leakage was an important issue during the project. Some features in the crime datasets could indirectly provide the answer to the model.
For the Chicago dataset, the Description column could contain information related to domestic crime. Other columns such as IUCR, FBI Code, Block, Updated On, ID fields, and some repeated location information were also found to be unsuitable for the final prediction. These columns were removed so that the model would not receive information that would not normally be available during prediction.
A separate leakage audit was also performed. The target labels were randomly changed and the model was tested again. Its performance dropped to around the random-prediction level, which provided confidence that the final model was not simply depending on accidental leakage.
For the NCRB feature-engineering process, the possibility of leakage from current-year crime features was also examined. The final modelling pipeline ensured that preprocessing information such as imputation values was learned from the training data and not from the test data.

### 3.11 Time-Based Data Splitting
For the Chicago dataset, a time-based split was used instead of randomly dividing the data. This was done because crime patterns can change over time and the model should be tested on future data. The data was divided as follows:
- 2023 to June 2024: Training
- July to December 2024: Validation
- 2025: Final testing
The 2025 data was not used while selecting features or adjusting the model. This made the final evaluation more realistic because the model learned from older crime records and was tested on later records.
For the NCRB modelling experiments, the traditional machine-learning and MLP models used a stratified 80:20 train-test split so that the five target classes remained similarly distributed in both sets. Median imputation was calculated only from the training data and then applied to the test data. For the LSTM and GRU models, the data was arranged chronologically so that information from previous years could be used to study temporal patterns.
Validation data was used to select a suitable classification threshold, and the final performance was then measured using the unseen 2025 test data. The six Chicago ML and DL models achieved more than 80% accuracy, with their results being fairly close to one another.

### 3.12 Machine-Learning Models
Different machine-learning models were developed and compared for the crime prediction tasks.
For the Chicago Domestic crime classification, the following models were implemented:
- Logistic Regression
- Random Forest
- HistGradientBoosting
- XGBoost
Different encoding methods were used depending on the model. The preprocessing and encoding steps were included in the modelling pipeline so that the same process could be applied consistently.
For the NCRB five-level violent-crime classification, the following traditional machine-learning approaches were implemented:
- Gaussian Naive Bayes
- Extra Tree Classifier, used as the implementation corresponding to the Random Tree approach
- One-vs-Rest Gaussian Naive Bayes
These models were tested using both the baseline and enhanced feature representations.

### 3.13 Deep-Learning Models
Deep-learning models were also developed to check whether neural networks could identify more complex patterns in the crime data.
For the Chicago dataset, an MLP model with embeddings and an FT-Transformer model were implemented using PyTorch. The FT-Transformer was used to learn relationships between the categorical crime features using feature embeddings and Transformer-based attention.
For the NCRB dataset, three deep-learning approaches were used:
- Multilayer Perceptron (MLP)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
The MLP was used for tabular crime classification, while LSTM and GRU were used to study temporal relationships by arranging district-level crime records into chronological sequences. Both baseline and enhanced feature representations were tested.

### 3.14 Model Evaluation
After training the models, their performance was evaluated using several measures rather than depending only on accuracy.
For the Chicago models, the evaluation included: Accuracy, Balanced Accuracy, Precision, Recall, F1-score, ROC-AUC.
The validation data was used to select a suitable classification threshold, and the final performance was then measured using the unseen 2025 test data. The six Chicago ML and DL models achieved more than 80% accuracy, with their results being fairly close to one another.
For the NCRB models, the evaluation included: Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, Classification Report, Confusion Matrix.
The models were compared using both baseline and enhanced feature configurations.

### 3.15 Challenges Faced During the Project
Several challenges were faced during the development of the project:
- **Large datasets**: The Chicago crime dataset was very large, which made data processing and model training more difficult. A processed, model-ready version was therefore created. The NCRB data also contained 76 separate files with different structures, which required detailed inspection before selecting the 14 files suitable for integration.
- **Missing and inconsistent data**: Missing values were present in different crime datasets. They could not always be treated as zero because a missing value could mean that the information was not reported. Therefore, missing-value analysis and suitable imputation methods were used.
- **High dimensionality**: The integrated NCRB dataset initially contained 344 columns and was later reduced to 331 canonical columns and finally to six meaningful crime groups. This made the dataset easier to use for modelling.
- **Duplicate information**: Some NCRB crime information appeared in multiple source formats. Eighteen duplicate representations had to be checked and consolidated.
- **Class imbalance**: The Chicago Domestic and Non-Domestic classes were not perfectly balanced. Therefore, class-balancing experiments were performed in some earlier machine-learning and deep-learning notebooks to understand their effect on the model results.
- **Data leakage**: Some Chicago columns could indirectly reveal the target, while some engineered NCRB features were related to current-year crime values used in target construction. These issues were checked carefully before final modelling.
- **Feature selection**: Initially, it was not clear which features should be used. Different feature-selection experiments were performed before selecting the final Chicago features and the grouped NCRB features.
- **Target construction**: The NCRB dataset did not directly provide the required five-level violent-crime target. Therefore, a project-specific violent-crime burden and five-level classification target had to be created.
- **Model comparison**: Different models produced different results. Therefore, several machine-learning and deep-learning models were tested instead of depending on only one model.
- **Code and experiment corrections**: During development, some early experiments needed correction and improvement. Corrected notebooks were created, and the final Python scripts were organized separately for dataset preparation, model training, deep learning, and evaluation.

### 3.16 Final Methodology Workflow
Finally, all the successful steps were combined into a complete workflow. The overall methodology followed:
$$\text{Data Collection} \rightarrow \text{Dataset Inspection} \rightarrow \text{Data Cleaning} \rightarrow \text{Data Integration} \rightarrow \text{Target Creation} \rightarrow \text{EDA} \rightarrow \text{Initial/Base Models} \rightarrow \text{Feature Selection/Crime Grouping} \rightarrow \text{Feature Engineering} \rightarrow \text{Leakage Checking} \rightarrow \text{Data Splitting} \rightarrow \text{Machine Learning Models} \rightarrow \text{Deep Learning Models} \rightarrow \text{Model Evaluation} \rightarrow \text{Model Comparison}$$

For the Chicago dataset, the final prediction task focused on Domestic vs Non-Domestic crime classification, while the NCRB dataset was used for five-level violent-crime classification.
The project was organized into separate folders for raw data, processed data, notebooks, trained models, outputs, reports, and source-code files. This helped keep the different experiments organized and made the final implementation easier to manage.

---

## 4. Results and Performance Analysis (Original Version)

Two sets of crime data were used for the two experiments. The initial analysis was on the exploratory patterns and classification of crime data from Chicago. The five level classification of violence against the integrated NCRB district level data was then evaluated in the second analysis.

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

### 4.3 NCRB Violent-Crime Classification Results (Original Version Under Faculty Review)
A separate experiment was carried out with the dataset of the Integrated National Crime Reporting Bureau (I-NCRB) district-level crime report. There were 9631 valid observations that were finally available for supervised modelling after data integration, data cleaning, crime grouping and target construction. There was a fairly even distribution across the five target levels with each having about 20% of the observations.

#### Table III. Distribution of Violent-Crime Target Levels
| Target Level | Number of Observations | Percentage |
|---|---|---|
| Level 1 | 1,927 | 20.01% |
| Level 2 | 1,926 | 19.99% |
| Level 3 | 1,926 | 19.99% |
| Level 4 | 1,926 | 19.99% |
| Level 5 | 1,926 | 19.99% |
| **Total** | **9,631** | **100%** |

#### The traditional ML and FE techniques
The baseline traditional models ranged from the accuracy of 77.38%-80.50% with the best accuracy being obtained by the Gaussian Naive Bayes One-vs-Rest model, which also had the best F1-score. Five aggregate and temporal features were then added to be evaluated using feature engineering. Interestingly, both the improved representation resulted in lower accuracy and F1-score for all three traditional models.

#### Table IV. Traditional ML: Baseline vs Enhanced (Original Random Split)
| Model | Baseline Accuracy (%) | Enhanced Accuracy (%) | $\Delta$ Accuracy | Baseline F1 (%) | Enhanced F1 (%) |
|---|---|---|---|---|---|
| Gaussian Naive Bayes | 78.06 | 76.09 | −1.97 | 78.43 | 76.43 |
| Extra Tree | 77.38 | 76.90 | −0.48 | 77.41 | 76.80 |
| One-vs-Rest Gaussian NB | 80.50 | 78.94 | −1.56 | 80.49 | 79.02 |

#### Deep Learning and Overall Comparison
The overall comparison and deep learning are displayed. The deep learning model was then tested using MLP, LSTM and GRU. The overall best performance was achieved by the enhanced MLP with an accuracy of 90.76%, F1-score of 90.76%, ROC-AUC of 99.09%, and PR-AUC of 96.28%.

#### Table V. Deep-Learning Model Performance (Original Random Split)
| Model | Feature Set | Accuracy (%) | F1-Score (%) | ROC-AUC (%) | PR-AUC (%) |
|---|---|---|---|---|---|
| MLP | Baseline | 89.33 | 89.29 | 99.08 | 96.40 |
| MLP | Enhanced | 90.76 | 90.76 | 99.09 | 96.28 |
| LSTM | Baseline | 76.44 | 76.78 | 94.83 | 81.31 |
| LSTM | Enhanced | 75.53 | 75.88 | 94.85 | 81.40 |
| GRU | Baseline | 77.34 | 77.58 | 94.96 | 81.75 |
| GRU | Enhanced | 75.83 | 76.19 | 94.86 | 81.24 |

The NCRB 5-level classification task was performed on the enhanced MLP, which performed best among all the other traditional baseline models, with a score 10.26 percentage points higher.

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
