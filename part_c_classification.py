import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.shipment_prep import FEATURES, NOISE_COLUMNS


# Load the modelling dataset
data = pd.read_csv("data/modelling_table.csv")

# Separate input features and classification target
X = data[FEATURES]
y = data["is_delayed"]


# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# Second split: 50% validation, 50% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)


# Display dataset sizes
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# Display delayed shipment counts
print("\nDelayed shipments in each dataset:")
print("Training:", int(y_train.sum()))
print("Validation:", int(y_val.sum()))
print("Test:", int(y_test.sum()))


# Display class distribution
print("\nClass distribution:")
print("Training:\n", y_train.value_counts().sort_index())
print("Validation:\n", y_val.value_counts().sort_index())
print("Test:\n", y_test.value_counts().sort_index())


# Verify the required feature configuration
print("\nTotal features:", len(FEATURES))
print("Noise columns:", NOISE_COLUMNS)


# Classification evaluation helper

def evaluate_model(model_name, y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "model": model_name,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
    }


# Blind baseline: predict every shipment is on time

baseline_predictions = np.zeros(len(y_val), dtype=int)

classification_results = []

classification_results.append(
    evaluate_model(
        "blind_all_on_time",
        y_val,
        baseline_predictions,
    )
)


# Logistic Regression

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_val)

classification_results.append(
    evaluate_model(
        "logistic_regression",
        y_val,
        logistic_predictions,
    )
)


# Find the best Decision Tree depth using validation F1

tree_depths = [2, 3, 4, 5, 6, 8, 10]

tree_results = []

for depth in tree_depths:
    tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42,
    )

    tree_model.fit(X_train, y_train)

    predictions = tree_model.predict(X_val)

    score = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    tree_results.append({
        "depth": depth,
        "val_f1": score,
    })

tree_results_df = pd.DataFrame(tree_results)

best_tree_depth = int(
    tree_results_df.sort_values(
        ["val_f1", "depth"],
        ascending=[False, True],
    ).iloc[0]["depth"]
)

print("\nDecision Tree Depth Results")
print(tree_results_df.round(4).to_string(index=False))
print("\nSelected tree depth:", best_tree_depth)


# Train the selected Decision Tree

selected_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=best_tree_depth,
    random_state=42,
)

selected_tree.fit(X_train, y_train)

tree_predictions = selected_tree.predict(X_val)

classification_results.append(
    evaluate_model(
        f"decision_tree_depth_{best_tree_depth}",
        y_val,
        tree_predictions,
    )
)


# Default Random Forest

default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

default_forest.fit(X_train, y_train)

forest_predictions = default_forest.predict(X_val)

classification_results.append(
    evaluate_model(
        "random_forest_default",
        y_val,
        forest_predictions,
    )
)


# Save classification comparison

classification_results_df = pd.DataFrame(
    classification_results
)

classification_results_df.to_csv(
    "outputs/part_c_models.csv",
    index=False,
)

print("\nClassification Model Results")
print(
    classification_results_df.round(4).to_string(index=False)
)

print("\nSaved outputs/part_c_models.csv")



# --------------------------------------------------
# FEATURE IMPORTANCE AND FEATURE SELECTION
# --------------------------------------------------

# 1. Train a Random Forest using all 17 features.
feature_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

feature_forest.fit(X_train, y_train)

# 2. Calculate ANOVA F-scores using training data only.
f_selector = SelectKBest(
    score_func=f_classif,
    k="all",
)

f_selector.fit(X_train, y_train)

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_selector.scores_,
    "rf_importance": feature_forest.feature_importances_,
})

# Rank features by F-score. Rank 1 means strongest signal.
feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(ascending=False, method="min")
    .astype(int)
)

# Identify the five deliberately added noise columns.
feature_scores["is_noise"] = (
    feature_scores["feature"].isin(NOISE_COLUMNS)
)

feature_scores = feature_scores[
    [
        "feature",
        "f_score",
        "f_rank",
        "rf_importance",
        "is_noise",
    ]
].sort_values("f_rank")

feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False,
)

print("\nFeature Importance Results")
print(feature_scores.round(4).to_string(index=False))

print("\nSaved outputs/part_c_feature_scores.csv")


# 3. Test different numbers of selected features.
selection_results = []

for k in [5, 7, 10, 12, 17]:

    selection_pipeline = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ])

    selection_pipeline.fit(X_train, y_train)

    predictions = selection_pipeline.predict(X_val)

    score = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    selection_results.append({
        "k": k,
        "val_f1": score,
    })


selection_results_df = pd.DataFrame(selection_results)

selection_results_df.to_csv(
    "outputs/part_c_selectk.csv",
    index=False,
)

print("\nFeature Selection Results")
print(selection_results_df.round(4).to_string(index=False))

print("\nSaved outputs/part_c_selectk.csv")



# --------------------------------------------------
# HYPERPARAMETER TUNING
# --------------------------------------------------

from sklearn.model_selection import ParameterGrid

# Select the top 7 features using training data only.
top7_selector = SelectKBest(
    score_func=f_classif,
    k=7,
)

X_train_top7 = top7_selector.fit_transform(X_train, y_train)
X_val_top7 = top7_selector.transform(X_val)

# Define the hyperparameter combinations.
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

# 1. Grid Search: evaluate every combination using 5-fold CV.
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
)

grid_search.fit(X_train_top7, y_train)

grid_predictions = grid_search.best_estimator_.predict(X_val_top7)

grid_val_f1 = f1_score(
    y_val,
    grid_predictions,
    zero_division=0,
)

print("\nGrid Search")
print("Candidates:", len(ParameterGrid(param_grid)))
print("Best parameters:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)
print("Validation F1:", grid_val_f1)


# 2. Randomized Search: evaluate 9 sampled combinations.
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42,
)

random_search.fit(X_train_top7, y_train)

random_predictions = random_search.best_estimator_.predict(X_val_top7)

random_val_f1 = f1_score(
    y_val,
    random_predictions,
    zero_division=0,
)

print("\nRandomized Search")
print("Candidates:", 9)
print("Best parameters:", random_search.best_params_)
print("Best CV accuracy:", random_search.best_score_)
print("Validation F1:", random_val_f1)


# 3. Compare against an untuned default Random Forest.
default_top7_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

default_top7_forest.fit(X_train_top7, y_train)

default_predictions = default_top7_forest.predict(X_val_top7)

default_val_f1 = f1_score(
    y_val,
    default_predictions,
    zero_division=0,
)

print("\nDefault Forest using 7 features")
print("Validation F1:", default_val_f1)


# 4. Save the tuning comparison.
tuning_results = pd.DataFrame([
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1,
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_val_f1,
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": str({
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1,
        }),
        "best_cv_accuracy": np.nan,
        "val_f1": default_val_f1,
    },
])

tuning_results.to_csv(
    "outputs/part_c_tuning.csv",
    index=False,
)

print("\nTuning Comparison")
print(tuning_results.round(4).to_string(index=False))

print("\nSaved outputs/part_c_tuning.csv")



# --------------------------------------------------
# FINAL MODEL AND ALERT THRESHOLD
# --------------------------------------------------

# Compare configurations using validation F1.
# Tie priority: default forest, grid search, randomized search.
candidates = [
    {
        "name": "default_forest_same_7_features",
        "val_f1": default_val_f1,
        "priority": 0,
        "params": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
    },
    {
        "name": "grid_search",
        "val_f1": grid_val_f1,
        "priority": 1,
        "params": grid_search.best_params_,
    },
    {
        "name": "randomized_search",
        "val_f1": random_val_f1,
        "priority": 2,
        "params": random_search.best_params_,
    },
]

winner = sorted(
    candidates,
    key=lambda item: (-item["val_f1"], item["priority"]),
)[0]

chosen_params = winner["params"]
chosen_from = winner["name"]

print("\nSelected Final Configuration")
print("Chosen from:", chosen_from)
print("Validation F1:", winner["val_f1"])
print("Parameters:", chosen_params)


# Fit the final pipeline using training data only.
final_model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    (
        "model",
        RandomForestClassifier(
            random_state=42,
            **chosen_params,
        ),
    ),
])

final_model.fit(X_train, y_train)


# Get validation probabilities.
# Class 1 represents a delayed shipment.
val_probabilities = final_model.predict_proba(X_val)[:, 1]


# Test thresholds from 0.1 to 0.9.
threshold_values = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9,
]

threshold_results = []

for threshold in threshold_values:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ).ravel()

    threshold_results.append({
        "threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        ),
    })


threshold_results_df = pd.DataFrame(threshold_results)

threshold_results_df.to_csv(
    "outputs/part_c_thresholds.csv",
    index=False,
)

print("\nAlert Threshold Results")
print(threshold_results_df.round(4).to_string(index=False))

print("\nSaved outputs/part_c_thresholds.csv")


# Select the highest threshold with recall >= 0.80.
eligible_thresholds = threshold_results_df[
    threshold_results_df["recall"] >= 0.80
]

if eligible_thresholds.empty:
    chosen_threshold = 0.1
else:
    chosen_threshold = float(
        eligible_thresholds["threshold"].max()
    )

print("\nChosen Alert Threshold:", chosen_threshold)


# --------------------------------------------------
# FINAL TEST EVALUATION — PERFORM ONLY ONCE
# --------------------------------------------------

# Do not change the model or threshold after this evaluation.
test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
).ravel()

final_test_results = pd.DataFrame([{
    "threshold": chosen_threshold,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "accuracy": accuracy_score(y_test, test_predictions),
    "precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_test, test_predictions, zero_division=0
    ),
}])

final_test_results.to_csv(
    "outputs/part_c_final_test.csv",
    index=False,
)

print("\nFinal Test Results")
print(final_test_results.round(4).to_string(index=False))

print("\nSaved outputs/part_c_final_test.csv")


# Save the final model configuration.
final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": (
        chosen_params["max_depth"]
        if chosen_params["max_depth"] is not None
        else "unlimited"
    ),
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from,
}])

final_config.to_csv(
    "outputs/part_c_final_config.csv",
    index=False,
)

print("\nFinal Model Configuration")
print(final_config.to_string(index=False))

print("\nSaved outputs/part_c_final_config.csv")

