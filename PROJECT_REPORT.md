# Project Report — Flight Status Prediction

This document provides the extended visual walkthrough for the project. It is intentionally implementation-free: the focus is on the data, modeling logic, results, and interpretation.

## 1. Objective

The task is to classify whether a U.S. domestic flight arrives more than 15 minutes late. The binary target is `ArrDel15`.

The analysis begins with the 2022 portion of a larger 2018–2022 Kaggle flight dataset. The raw 2022 file contains 4,078,318 rows and 61 columns.

![Dataset funnel](assets/figures/summary/dataset_funnel.png)

## 2. Exploratory Analysis

### 2.1 Busiest origin airports

The full 2022 data were first used to rank origin airports by flight volume.

![Top 10 busiest origin airports](assets/figures/summary/top_10_airports_2022.png)

### 2.2 Delay behavior by month

![Average departure delay by month](assets/figures/report/02_average_departure_delay_by_month.png)

Average delay rises during busier summer months in the analyzed period.

### 2.3 Delay behavior by time of day

![Average departure delay by time of day](assets/figures/report/03_average_departure_delay_by_time_of_day.png)

Delay is lowest in the early morning and generally increases through the day, peaking during late-afternoon and evening periods.

### 2.4 Delay behavior by weekday

![Average departure delay by weekday](assets/figures/report/04_average_departure_delay_by_day_of_week.png)

The weekday distribution shows non-uniform delay behavior, supporting the use of temporal variables.

### 2.5 Operational correlation structure

![Correlation matrix before preprocessing](assets/figures/report/05_correlation_matrix_before_preprocessing.png)

Strong relationships appear between several operational variables. These include the expected connection between air time and distance, and between departure and arrival delay indicators.

## 3. Modeling Sample

The exploratory notebook identifies the top ten origin airports across 2022. For the subsequent modeling pipeline, the implementation narrows the dataset to **ATL-origin flights**, leaving 183,697 records before final preparation.

After preprocessing and dropping rows without the target/departure-delay indicator, the prepared dataset contains **179,852 rows**.

![Target distribution](assets/figures/summary/target_distribution.png)

The target is imbalanced: approximately 80.3% are class 0 and 19.7% are class 1.

## 4. Preprocessing and Feature Engineering

The raw modeling table is reduced to the following conceptual groups:

- taxi and airborne operations;
- calendar variables;
- distance and air time;
- airline;
- scheduled and actual clock-time variables;
- departure-delay indicator;
- arrival-delay target.

Five HHMM time fields are converted into circular sine/cosine components. This representation preserves daily periodicity and avoids treating midnight as numerically distant from 23:59.

Numeric predictors are mean-imputed and standardized. Categorical values are imputed using the most frequent category and one-hot encoded.

![Feature distributions](assets/figures/report/06_feature_distributions.png)

![Correlation matrix after preprocessing](assets/figures/report/07_correlation_matrix_after_preprocessing.png)

## 5. Additional EDA Diagnostics

### 5.1 Departure delay by airline

![Departure delay by airline](assets/figures/extended/departure_delay_by_airline.png)

The airline-level boxplot demonstrates heavy right tails and substantial outliers.

### 5.2 Outlier-focused views

![Filtered departure delay outliers](assets/figures/extended/departure_delay_outliers_filtered.png)

![Log-transformed departure delays](assets/figures/extended/departure_delay_log_transformed.png)

![Departure delay violin plot](assets/figures/extended/departure_delay_violin.png)

These views reinforce the highly skewed nature of delay duration.

## 6. Baseline Classification

Eight baseline classifiers were compared using validation accuracy.

![Baseline accuracy](assets/figures/summary/baseline_accuracy.png)

XGBoost achieved the highest baseline validation accuracy at 97.21%.

### Confusion matrices

![Baseline confusion matrices](assets/figures/report/08_baseline_confusion_matrices.png)

The confusion matrices show that models with similar overall accuracy can differ meaningfully in their treatment of delayed flights.

## 7. Hyperparameter Tuning

Five-fold GridSearchCV was used to explore model settings. The report's selected XGBoost configuration was:

- learning rate: 0.2
- maximum depth: 9
- estimators: 200

![Tuning comparison](assets/figures/summary/tuning_comparison.png)

The tuning stage improved XGBoost and SVM, while some configurations for Decision Tree and Random Forest underperformed their baseline settings.

### Logistic Regression complexity curve

![Logistic Regression loss](assets/figures/report/09_logistic_regression_train_validation_loss.png)

### Decision Tree complexity curve

![Decision Tree loss](assets/figures/report/10_decision_tree_train_validation_loss.png)

### Random Forest grid diagnostics

| Validation loss | Training loss |
|---|---|
| ![Random Forest validation loss](assets/figures/extended/random_forest_validation_loss_heatmap.png) | ![Random Forest training loss](assets/figures/extended/random_forest_training_loss_heatmap.png) |

### AdaBoost grid diagnostics

| Validation loss | Training loss |
|---|---|
| ![AdaBoost validation loss](assets/figures/extended/adaboost_validation_loss_heatmap.png) | ![AdaBoost training loss](assets/figures/extended/adaboost_training_loss_heatmap.png) |

## 8. Tuned XGBoost

The tuned model produced the following confusion matrix on a 26,978-row evaluation split:

![Tuned XGBoost confusion matrix](assets/figures/report/11_xgboost_confusion_matrix.png)

| Metric | Value |
|---|---:|
| Accuracy | 97.75% |
| Delay precision | 94.90% |
| Delay recall | 93.66% |
| Delay F1-score | 94.28% |
| True negatives | 21,380 |
| False positives | 268 |
| False negatives | 338 |
| True positives | 4,992 |

![XGBoost metrics](assets/figures/summary/xgboost_metrics.png)

### ROC analysis

![XGBoost ROC](assets/figures/report/12_xgboost_roc_curve.png)

### Learning curve

![XGBoost learning curve](assets/figures/report/13_xgboost_learning_curve.png)

The training curve remains near 1.0 while the validation curve approaches the upper 0.97 range, indicating a persistent but relatively small generalization gap.

## 9. Overfitting-Mitigation Experiment

A later experimental configuration used shallower trees, row and feature subsampling, L1/L2 regularization, and early stopping. The objective was to reduce the strong training fit observed in the selected tuned model.

![XGBoost logloss](assets/figures/report/14_xgboost_logloss.png)

The notebook records a 95.18% test accuracy for this more strongly regularized experiment. Because the project report uses the 97.75–97.78% tuned XGBoost result as the headline model, this later configuration is presented here as an auxiliary robustness experiment rather than replacing the main reported result.

## 10. Computational Comparison

![Reported computation time](assets/figures/summary/training_time.png)

XGBoost offered a strong speed/accuracy balance in the reported comparison.

## 11. Interpretation

The project demonstrates a complete classification workflow: exploratory analysis, feature reduction, cyclical time engineering, preprocessing, baseline comparison, cross-validated tuning, and multi-metric evaluation.

The high predictive scores are partly explained by the use of operational variables that are closely related to arrival delays. For a strict pre-departure prediction system, target-time availability must be enforced: variables such as actual departure status, taxi movements, and wheel-off/wheel-on timings should only be used if they are genuinely available at the intended prediction moment.

## 12. Next Steps

A stronger forecasting-oriented extension would:

1. define a fixed prediction time such as 2 hours before scheduled departure;
2. retain only features known at that time;
3. add weather, airport congestion, and air-traffic-control indicators;
4. use chronological validation rather than random splitting;
5. compare calibration and PR-AUC in addition to ROC-AUC;
6. evaluate performance separately by airport, airline, season, and delay prevalence.

