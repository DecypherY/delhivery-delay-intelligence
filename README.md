## Part A — Data Cleaning and Exploratory Data Analysis

### Data preparation

The raw dataset contained 1,224 shipment rows. Removing 24 exact duplicate rows left 1,200 unique shipments. Missing numeric values were imputed using the historical medians: 314.95 km for distance, 1.83 kg for parcel weight, and INR 1,121.50 for declared value. Service-type text was normalized from 12 distinct spellings to three standardized categories. The modelling table contains 1,200 rows, and the required features and target columns have no missing values.

### Data findings

1. **Service type:** Express shipments had the highest delay rate at 48.97% (119 of 243 shipments), compared with 20.56% for Standard and 8.57% for Economy.

2. **Hub tier:** The delay rate increased from 4.00% at Tier 1 to 21.50% at Tier 2 and 44.00% at Tier 3. Each tier contained 400 shipments in this synthetic dataset.

3. **Distance:** Delay rates rose across the distance bands: 2.19% for 0–200 km, 13.68% for 201–400 km, 43.64% for 401–600 km, and 85.23% for shipments over 600 km.

4. **Correlation with delivery time:** The three strongest correlations with `delivery_hours`, excluding the target itself, were `promised_hours` (+0.8372), `distance_km` (+0.4013), and `n_stops` (+0.3247). These are associations, not proof of causation.

5. **Declared-value transformation:** The median-filled declared-value column had a mean of INR 1,533.05 and a median of INR 1,121.50. After the natural-log transformation, the mean was 7.028 and the median was 7.022. The log transformation reduces the influence of large values and makes this right-skewed variable more balanced.

### Reproducibility note

The dataset used in this project is synthetic and was generated for the capstone exercise. It does not contain real Delhivery shipment records, and this project is not affiliated with Delhivery.


## Part B — Delivery-Hours Regression

### Data splitting and scaling

The modelling dataset was split into:
- Training: 840 rows (70%)
- Validation: 180 rows (15%)
- Test: 180 rows (15%)

StandardScaler was fitted only on the training data to prevent data leakage.
The maximum absolute mean of the scaled training features was approximately
0.0000000000000004, while the test value was 0.2187.

### Model comparison

Five approaches were evaluated on the validation set:

| Model | Validation RMSE (hours) | Validation R² |
|---|---:|---:|
| Mean baseline | 18.8552 | -0.0212 |
| Linear Regression | 4.8912 | 0.9313 |
| Decision Tree (depth 6) | 6.2796 | 0.8867 |
| Random Forest | 5.4784 | 0.9138 |
| Gradient Boosting | 4.9165 | 0.9306 |

The mean baseline's negative R² indicates that predicting the training-set
mean performed slightly worse than predicting the validation-set mean.

Linear Regression was selected as the final model because it achieved the
lowest validation RMSE (4.8912 hours).

### Decision Tree depth analysis

A depth of 6 achieved the lowest validation RMSE of 6.2796 hours.

A depth-1 tree showed underfitting: its training RMSE was 11.6334 and
validation RMSE was 12.1324.

The unlimited-depth tree showed overfitting: its training RMSE was 0.0000,
but its validation RMSE increased to 7.7890. It memorized the training
data without generalizing as well to unseen data.

### Gradient Boosting learning-rate experiment

| Learning rate | Validation RMSE | Validation R² |
|---:|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | 4.9165 | 0.9306 |
| 1.0 | 7.3701 | 0.8440 |

A learning rate of 0.1 performed best. A rate of 0.01 learned too slowly
with 100 estimators, while 1.0 was too aggressive and performed worse.

### Five-fold cross-validation

Cross-validation was performed on the combined training and validation
data using the same shuffled five-fold split configuration.

| Model | Mean R² | Standard deviation |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | 0.9352 | 0.0044 |

Gradient Boosting performed better across the cross-validation folds.
However, Linear Regression remained the final selection because the
assignment specifies choosing the candidate with the lowest validation RMSE.
The difference between the two validation RMSE values was very small.

### Final test evaluation

The selected Linear Regression model was evaluated once on the untouched
test set.

| Metric | Test result |
|---|---:|
| MAE | 3.5671 hours |
| MSE | 20.4624 |
| RMSE | 4.5235 hours |
| R² | 0.9493 |

The final test R² exceeded the assignment's acceptance threshold of 0.88.

Note: The project uses a synthetic shipment dataset. These results
demonstrate performance on the generated data and should not be interpreted
as verified real-world Delhivery performance.

## Model results (classification)

### Baseline and model comparison

The blind baseline predicted every shipment as on time. It achieved
76.67% accuracy but had zero precision, recall, and F1-score because
it failed to identify any delayed shipments.

Among the trained models, Logistic Regression achieved 83.33%
validation recall, while the default Random Forest achieved 91.11%
validation accuracy and 84.21% precision.

### Feature analysis

The seven strongest features by ANOVA F-score were distance_km,
hub_tier, hub_daily_capacity, n_stops, promised_hours, rain_flag,
and sale_event_flag.

All five deliberately added noise features had F-scores below 10
and ranked outside the top seven. Their combined Random Forest
importance was approximately 0.2253.

### Feature selection

Selecting seven features achieved the highest validation F1-score
of 0.8333. This was better than selecting 5, 10, 12, or all 17
features.

### Hyperparameter tuning

Grid Search evaluated 27 combinations using five-fold
cross-validation, while Randomized Search evaluated 9 combinations.
Grid Search achieved higher cross-validation accuracy than
Randomized Search.

However, neither tuned model outperformed the default Random Forest
on validation F1. The default forest achieved 0.8333, compared with
0.7750 for Grid Search and 0.7901 for Randomized Search.

This demonstrates that hyperparameter tuning does not guarantee
better validation performance. The validation set contains only
180 shipments, so the results should also be interpreted with care.

### Alert threshold

The selected alert threshold was 0.5, the highest tested threshold
that maintained validation recall of at least 0.80. At this
threshold, validation precision was 0.7955 and recall was 0.8333.

Increasing the threshold to 0.6 improved precision to 0.8889,
but reduced recall to 0.7619, below the required target.

### Final test results

At the selected threshold of 0.5, the final test results were:

- Accuracy: 86.67%
- Precision: 68.89%
- Recall: 75.61%
- F1-score: 72.09%

The model correctly identified 31 of the 41 delayed shipments and
missed 10. It generated 14 false delay alerts.

Test recall was lower than validation recall (75.61% compared with
83.33%). The test set contained only 41 delayed shipments, so each
missed delayed shipment represents approximately 2.44 percentage
points of recall. The difference is reported rather than hidden.

The final configuration used seven selected features, 100 Random
Forest trees, unlimited tree depth, a minimum leaf size of one,
and an alert threshold of 0.5.
