# Model Card — Tuned XGBoost Flight Delay Classifier

## Model Purpose

The selected model classifies whether a flight's arrival delay exceeds 15 minutes.

**Target:** `ArrDel15`

- `0`: arrival delay ≤ 15 minutes
- `1`: arrival delay > 15 minutes

## Evaluation Data

The prepared ATL-origin dataset contains 179,852 rows and was randomly divided into:

- 70% training
- 15% validation
- 15% testing

The delayed class represents about 19.7% of the prepared data.

## Selected Model

**XGBoost**

Reported GridSearchCV selection:

| Setting | Value |
|---|---:|
| Learning rate | 0.2 |
| Max depth | 9 |
| Estimators | 200 |
| CV folds | 5 |

## Headline Performance

The tuned evaluation confusion matrix is:

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 21,380 | 268 |
| Actual 1 | 338 | 4,992 |

Derived metrics:

| Metric | Score |
|---|---:|
| Accuracy | 97.75% |
| Precision — delayed class | 94.90% |
| Recall — delayed class | 93.66% |
| F1 — delayed class | 94.28% |

The notebook also reports approximately **97.78% test accuracy** for the tuned model.

## Why Recall Matters

For delay-alert applications, false negatives correspond to flights that are delayed but classified as not delayed. If missing a delay carries greater operational cost than issuing an unnecessary warning, recall for class 1 is particularly important.

## Strengths

- high validation and test accuracy in the reported experimental setup;
- strong recall for delayed flights;
- nonlinear modeling of interactions between operational and temporal variables;
- efficient reported training time relative to SVM and k-NN;
- robust handling of mixed transformed features.

## Limitations

### Feature availability

Some features are only known after operations have begun, including departure-delay status and actual movement/taxi timings. These features make the model unsuitable for a strict pre-departure forecast unless removed or the prediction moment is moved later.

### Generalization scope

The modeling implementation is ATL-origin specific. Performance should not be assumed to transfer unchanged to other airports, years, or operational conditions.

### Random split

A random split can mix flights from similar periods across training and evaluation sets. Chronological validation would be more representative for future-flight forecasting.

### Imbalance

The delayed class is the minority class. Accuracy should therefore always be interpreted with precision, recall, F1, confusion matrices, and ideally precision–recall curves.

### Very high ROC performance

The near-perfect ROC curve should be interpreted cautiously because target-proximal operational variables are included. A lower but more realistic score may be expected in a strict advance-prediction setup.

## Experimental Overfitting Mitigation

A later notebook experiment reduced tree depth and added:

- row subsampling;
- column subsampling;
- L1/L2 regularization;
- gamma regularization;
- early stopping.

That experiment recorded approximately **95.18% test accuracy**. It is retained as a robustness experiment rather than the headline model because the project report centers on the 97.75–97.78% tuned XGBoost result.

## Recommended Production Validation

Before deployment, the model should be rebuilt around a clearly defined prediction timestamp and tested with:

- chronological holdout data;
- unseen airports or route groups;
- calibration analysis;
- PR-AUC;
- airline- and airport-specific error analysis;
- seasonal performance analysis;
- robustness under weather and disruption events.
