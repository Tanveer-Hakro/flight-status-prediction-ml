<div align="center">

# Flight Status Prediction with Machine Learning

**A visual machine-learning case study for classifying U.S. domestic flights as delayed or on time**

2022 flight operations · exploratory data analysis · cyclical time features · model benchmarking · XGBoost optimization

</div>

> **Portfolio edition:** this repository presents the project methodology, visual analysis, evaluation, and findings only. Source notebooks and implementation code are intentionally excluded.

![Project workflow](assets/figures/summary/project_workflow.svg)

## Project Overview

The project investigates whether operational, temporal, and flight-context variables can classify whether a flight arrives **more than 15 minutes late**. The binary target is `ArrDel15`, where `1` represents a delay greater than 15 minutes and `0` represents an on-time or shorter-delay arrival.

The source data contain U.S. domestic flights from 2018–2022. The analysis uses the **2022 file**, containing **4,078,318 records and 61 fields**. Exploratory analysis was performed on the full 2022 data, after which the modeling workflow narrowed the data to **ATL-origin flights** for computationally manageable experimentation.

| Project signal | Result |
|---|---:|
| Raw 2022 records | **4,078,318** |
| Raw features | **61** |
| ATL-origin records before final preparation | **183,697** |
| Prepared modeling rows | **179,852** |
| Train / validation / test split | **70% / 15% / 15%** |
| Baseline classifiers compared | **8** |
| Best baseline model | **XGBoost** |
| Baseline XGBoost validation accuracy | **97.21%** |
| Tuned XGBoost validation accuracy | **97.75%** |
| Notebook test accuracy for tuned XGBoost | **97.78%** |
| Delay-class recall | **93.66%** |

![Dataset funnel](assets/figures/summary/dataset_funnel.png)

## Problem Definition

The task is a binary classification problem:

- **Class 0:** arrival delay is 15 minutes or less.
- **Class 1:** arrival delay exceeds 15 minutes.

The modeling data are imbalanced, with roughly four on-time flights for every delayed flight.

![Target distribution](assets/figures/summary/target_distribution.png)

Because a missed delay can be more costly than a false alert in many operational settings, **recall for the delayed class** is an important metric alongside accuracy.

## Exploratory Data Analysis

### Airport activity

The full 2022 dataset shows ATL as the busiest origin airport among the ten highest-volume airports examined.

![Top 10 airports](assets/figures/summary/top_10_airports_2022.png)

### Temporal delay patterns

Average departure delay varies across the calendar and the day. The analysis shows stronger delays during busier summer months and a clear increase through late-afternoon and evening hours.

| Monthly pattern | Time-of-day pattern |
|---|---|
| ![Average departure delay by month](assets/figures/report/02_average_departure_delay_by_month.png) | ![Average departure delay by time of day](assets/figures/report/03_average_departure_delay_by_time_of_day.png) |

The weekday view also shows variation across the week, with higher average delays toward the weekend in the analyzed data.

![Average departure delay by weekday](assets/figures/report/04_average_departure_delay_by_day_of_week.png)

### Correlation structure

Several operational variables are strongly related. In particular, departure-delay indicators and arrival-delay status are closely connected, while flight distance and air time also show expected dependence.

![Correlation matrix before preprocessing](assets/figures/report/05_correlation_matrix_before_preprocessing.png)

## Data Preparation

The modeling workflow reduced the original 61-field dataset to a smaller operational feature set and then transformed it for machine learning.

Key preparation steps included:

- removing redundant identifiers and duplicated geographic descriptors;
- keeping operational variables such as taxi times, distance, air time, airline, and delay indicators;
- transforming five HHMM time variables into **sine/cosine cyclical encodings** so midnight remains close to late-night times;
- mean-imputing and standardizing numeric predictors;
- most-frequent imputation and one-hot encoding for categorical predictors;
- removing duplicated rows;
- removing rows missing the target or departure-delay indicator.

The selected raw modeling table contained **15 columns** before encoding. After cyclical expansion and airline encoding, the prepared table contained **33 columns including the target and `DepDel15`**.

![Feature distributions](assets/figures/report/06_feature_distributions.png)

The post-preprocessing correlation view highlights the engineered cyclical features and encoded airline variables.

![Correlation matrix after preprocessing](assets/figures/report/07_correlation_matrix_after_preprocessing.png)

## Model Benchmarking

Eight baseline classifiers were evaluated:

**Logistic Regression · Decision Tree · Random Forest · XGBoost · AdaBoost · SVM · k-NN · Naive Bayes**

XGBoost delivered the strongest baseline validation accuracy at **97.21%**, followed by SVM and Random Forest.

![Baseline accuracy](assets/figures/summary/baseline_accuracy.png)

The confusion matrices provide a more complete view of how the models handle the minority delayed class.

![Baseline confusion matrices](assets/figures/report/08_baseline_confusion_matrices.png)

## Hyperparameter Optimization

Cross-validated parameter searches were used to evaluate model complexity and select stronger configurations. The reported XGBoost search considered learning rate, tree depth, and number of estimators.

The best GridSearchCV result used:

| Hyperparameter | Selected value |
|---|---:|
| Learning rate | **0.2** |
| Maximum tree depth | **9** |
| Number of estimators | **200** |
| Cross-validation | **5-fold** |

The tuned model reached **99.94% training accuracy** and **97.75% validation accuracy**, indicating strong predictive performance with a small train–validation gap.

![Baseline vs tuned](assets/figures/summary/tuning_comparison.png)

Not every model improved after tuning. XGBoost and SVM improved, while the reported Decision Tree and Random Forest configurations performed worse than their baselines. This is a useful reminder that hyperparameter tuning does not automatically guarantee a better validation score.

### Complexity diagnostics

| Logistic Regression | Decision Tree |
|---|---|
| ![Logistic Regression train and validation loss](assets/figures/report/09_logistic_regression_train_validation_loss.png) | ![Decision Tree train and validation loss](assets/figures/report/10_decision_tree_train_validation_loss.png) |

## Final XGBoost Evaluation

The tuned XGBoost confusion matrix contains:

- **21,380** true negatives
- **268** false positives
- **338** false negatives
- **4,992** true positives

![Tuned XGBoost confusion matrix](assets/figures/report/11_xgboost_confusion_matrix.png)

From these counts, the delayed-flight class achieves approximately **94.90% precision**, **93.66% recall**, and **94.28% F1-score**, with **97.75% overall accuracy**.

![XGBoost metrics](assets/figures/summary/xgboost_metrics.png)

The ROC curve is extremely strong in this experiment.

![XGBoost ROC curve](assets/figures/report/12_xgboost_roc_curve.png)

The learning curve shows a near-perfect training score and a validation score that approaches the high-97% range as the training sample grows.

![XGBoost learning curve](assets/figures/report/13_xgboost_learning_curve.png)

A later experiment introduced shallower trees, row/feature subsampling, regularization, and early stopping to reduce overfitting pressure. Its train and validation log-loss trajectories are shown below.

![XGBoost logloss](assets/figures/report/14_xgboost_logloss.png)

## Efficiency

The reported timing comparison shows that XGBoost achieved a favorable combination of accuracy and computation time relative to several alternatives.

![Training time comparison](assets/figures/summary/training_time.png)

| Model | Baseline accuracy | Tuned accuracy | Reported time |
|---|---:|---:|---:|
| Logistic Regression | 93.28% | 93.28% | 14.2 s |
| Decision Tree | 94.23% | 93.65% | 12 s |
| Random Forest | 96.20% | 94.82% | 3 min 7 s |
| **XGBoost** | **97.21%** | **97.75%** | **34.3 s** |
| AdaBoost | 92.66% | 92.86% | 2 min 20 s |
| SVM | 96.60% | 97.05% | 3 min 18 s |
| k-NN | 92.69% | 92.71% | 5 min 35 s |
| Naive Bayes | 88.74% | — | — |

## Key Findings

1. **Time matters.** Delay behavior changes by month, weekday, and hour, supporting explicit temporal feature engineering.
2. **Cyclical encoding is appropriate for clock time.** Sine/cosine transformations preserve the continuity between times such as 23:59 and 00:01.
3. **Ensemble methods were strongest.** XGBoost led the baseline comparison and remained the best model after tuning.
4. **Accuracy alone is not sufficient.** The delayed class is the minority class, so recall, precision, F1-score, and confusion matrices are essential.
5. **XGBoost balanced performance and compute cost well.** Its reported tuned validation accuracy was 97.75% with substantially shorter training time than SVM and k-NN.

## Modeling Scope and Limitations

The very high scores should be interpreted in the context of the predictors used. The feature set includes variables such as `DepDel15`, actual departure-related times, taxi times, and wheel-off/wheel-on information. Some of these variables become available only after a flight has begun or progressed.

For that reason, this project is best understood as **operational flight-status classification** rather than a strictly pre-departure forecast. A production pre-departure model should restrict inputs to information available at prediction time and should validate performance using chronological splits to reduce temporal leakage.

Other limitations include:

- the modeling implementation focuses on ATL-origin flights rather than the full U.S. network;
- weather, air-traffic-control conditions, and airport congestion forecasts are not explicitly modeled;
- the target distribution is imbalanced;
- random train/validation/test splitting may produce more optimistic estimates than a time-based holdout for future-flight prediction.

## Extended Visual Diagnostics

Additional exploratory and tuning diagnostics from the analysis are available in the project report:

- airline-level delay distributions;
- filtered and log-transformed outlier views;
- a violin distribution of departure delays;
- Random Forest tuning heatmaps;
- AdaBoost tuning heatmaps.

See **[PROJECT_REPORT.md](PROJECT_REPORT.md)** for the complete visual walkthrough and **[MODEL_CARD.md](MODEL_CARD.md)** for the evaluation context and responsible-use notes.

## Dataset

The project uses the **Flight Delay Dataset (2018–2022)** distributed on Kaggle. The analysis shown here is based on the 2022 flight file.

Dataset reference: [Kaggle — Flight Delay Dataset 2018–2022](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022)

More details are provided in **[DATASET.md](DATASET.md)**.

---

### Project focus

**Machine Learning · Classification · Flight Analytics · Feature Engineering · XGBoost · Model Evaluation**

