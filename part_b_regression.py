
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline


from src.shipment_prep import FEATURES


# 1. Load the modelling table
data = pd.read_csv("data/modelling_table.csv")

# 2. Separate input features (X) and target (y)
X = data[FEATURES]
y = data["delivery_hours"]


# 3. First split: 70% training and 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

# 4. Second split: divide the temporary 30% equally
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
)


# 5. Scale the features using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 6. Print the split sizes
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# 7. Check scaling
train_scaled_max_abs_mean = np.abs(
    X_train_scaled.mean(axis=0)
).max()

test_scaled_max_abs_mean = np.abs(
    X_test_scaled.mean(axis=0)
).max()

print("\nTraining scaled max absolute mean:", train_scaled_max_abs_mean)
print("Test scaled max absolute mean:", test_scaled_max_abs_mean)


# 8. Save Part B split and scaling checks

checks = [
    ("train_rows", len(X_train)),
    ("val_rows", len(X_val)),
    ("test_rows", len(X_test)),
    ("train_scaled_max_abs_mean", train_scaled_max_abs_mean),
    ("test_scaled_max_abs_mean", test_scaled_max_abs_mean),
]

checks_df = pd.DataFrame(checks, columns=["metric", "value"])

checks_df.to_csv("outputs/part_b_checks.csv", index=False)

print("\nSaved outputs/part_b_checks.csv")



# 9. Mean baseline model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Calculate average delivery time from training data only.
baseline_prediction = np.full(
    len(y_val),
    y_train.mean(),
)

# Evaluate baseline on validation data.
baseline_mae = mean_absolute_error(y_val, baseline_prediction)
baseline_mse = mean_squared_error(y_val, baseline_prediction)
baseline_rmse = np.sqrt(baseline_mse)
baseline_r2 = r2_score(y_val, baseline_prediction)

print("\nMean Baseline Results")
print("Training mean:", round(y_train.mean(), 4))
print("MAE:", round(baseline_mae, 4))
print("MSE:", round(baseline_mse, 4))
print("RMSE:", round(baseline_rmse, 4))
print("R2:", round(baseline_r2, 4))


depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_results = []

for depth in depths:
    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42,
    )

    tree.fit(X_train, y_train)

    train_predictions = tree.predict(X_train)
    val_predictions = tree.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

    tree_results.append({
        "max_depth": "unlimited" if depth is None else str(depth),
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
    })

tree_results_df = pd.DataFrame(tree_results)

tree_results_df.to_csv(
    "outputs/part_b_tree_depth.csv",
    index=False,
)

print("\nDecision Tree Depth Experiment")
print(tree_results_df.round(4).to_string(index=False))
print("\nSaved outputs/part_b_tree_depth.csv")

# Candidate models

models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),

    "decision_tree": DecisionTreeRegressor(
        max_depth=6,
        random_state=42,
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    ),
}

model_results = [{
    "model": "mean_baseline",
    "split": "val",
    "MAE": baseline_mae,
    "MSE": baseline_mse,
    "RMSE": baseline_rmse,
    "R2": baseline_r2,
}]

for model_name, model in models.items():
    # Train using training data only
    model.fit(X_train, y_train)

    # Predict validation data
    predictions = model.predict(X_val)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predictions)

    model_results.append({
        "model": model_name,
        "split": "val",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    })

model_results_df = pd.DataFrame(model_results)

model_results_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False,
)

print("\nCandidate Model Results")
print(model_results_df.round(4).to_string(index=False))
# Gradient Boosting learning-rate experiment

learning_rates = [0.01, 0.1, 1.0]

gb_results = []

for lr in learning_rates:
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42,
    )

    gb_model.fit(X_train, y_train)

    predictions = gb_model.predict(X_val)

    val_mse = mean_squared_error(y_val, predictions)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, predictions)

    gb_results.append({
        "learning_rate": lr,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
    })

gb_results_df = pd.DataFrame(gb_results)

gb_results_df.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False,
)

print("\nGradient Boosting Learning Rate Experiment")
print(gb_results_df.round(4).to_string(index=False))
print("\nSaved outputs/part_b_gb_learning_rate.csv")


# Five-fold cross-validation for the top two models

X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

top_models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    ),
}

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_results = []

for model_name, model in top_models.items():
    scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=kf,
        scoring="r2",
    )

    cv_results.append({
        "model": model_name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std(),
    })

cv_results_df = pd.DataFrame(cv_results)

cv_results_df.to_csv(
    "outputs/part_b_cv.csv",
    index=False,
)

print("\nFive-Fold Cross-Validation Results")
print(cv_results_df.round(4).to_string(index=False))
print("\nSaved outputs/part_b_cv.csv")


# Final model evaluation on the untouched test set

final_model = models["linear_regression"]

test_predictions = final_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_predictions)

test_result = pd.DataFrame([{
    "model": "linear_regression",
    "split": "test",
    "MAE": test_mae,
    "MSE": test_mse,
    "RMSE": test_rmse,
    "R2": test_r2,
}])

final_metrics_df = pd.concat(
    [model_results_df, test_result],
    ignore_index=True,
)

final_metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False,
)

print("\nFinal Test Set Results")
print(test_result.round(4).to_string(index=False))
print("\nUpdated outputs/part_b_metrics.csv")

