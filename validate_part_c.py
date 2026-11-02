import pandas as pd
import numpy as np

models = pd.read_csv("outputs/part_c_models.csv")
features = pd.read_csv("outputs/part_c_feature_scores.csv")
selectk = pd.read_csv("outputs/part_c_selectk.csv")
tuning = pd.read_csv("outputs/part_c_tuning.csv")
thresholds = pd.read_csv("outputs/part_c_thresholds.csv")
test = pd.read_csv("outputs/part_c_final_test.csv")
config = pd.read_csv("outputs/part_c_final_config.csv")

checks = []

def check(name, condition):
    checks.append((name, bool(condition)))

# Model comparison
check("Four model comparison rows", len(models) == 4)

baseline = models[models["model"] == "blind_all_on_time"].iloc[0]
check(
    "Blind baseline confusion matrix",
    (baseline["TN"], baseline["FP"], baseline["FN"], baseline["TP"])
    == (138, 0, 42, 0),
)

for _, row in models.iterrows():
    tn, fp, fn, tp = row[["TN", "FP", "FN", "TP"]]
    total = tn + fp + fn + tp

    accuracy = (tn + tp) / total
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    check(
        f"{row['model']} metrics match confusion matrix",
        np.allclose(
            [row["accuracy"], row["precision"], row["recall"], row["f1"]],
            [accuracy, precision, recall, f1],
            atol=1e-4,
        ),
    )

# Feature analysis
expected_top7 = [
    "distance_km",
    "hub_tier",
    "hub_daily_capacity",
    "n_stops",
    "promised_hours",
    "rain_flag",
    "sale_event_flag",
]

noise = features[features["is_noise"] == True]

check("17 feature score rows", len(features) == 17)
check(
    "Expected top seven features",
    features.sort_values("f_rank")["feature"].head(7).tolist()
    == expected_top7,
)
check("Five noise features identified", len(noise) == 5)
check("All noise ranks are 8 or worse", (noise["f_rank"] >= 8).all())
check("All noise F-scores below 10", (noise["f_score"] < 10).all())
check(
    "Smallest top-seven F-score above 30",
    features.nsmallest(7, "f_rank")["f_score"].min() > 30,
)

# Feature selection
check("Five feature-selection experiments", len(selectk) == 5)
check(
    "Seven features have highest validation F1",
    int(selectk.loc[selectk["val_f1"].idxmax(), "k"]) == 7,
)

# Tuning
grid = tuning[tuning["search"] == "grid_search"].iloc[0]
random = tuning[tuning["search"] == "randomized_search"].iloc[0]
default = tuning[
    tuning["search"] == "default_forest_same_7_features"
].iloc[0]

check("Grid search has 27 candidates and 135 fits",
      grid["candidates"] == 27 and grid["total_fits"] == 135)
check("Randomized search has 9 candidates and 45 fits",
      random["candidates"] == 9 and random["total_fits"] == 45)
check("Grid CV accuracy >= randomized CV accuracy",
      grid["best_cv_accuracy"] >= random["best_cv_accuracy"])
check("Default forest has one fit",
      default["candidates"] == 1 and default["total_fits"] == 1)
check("Default forest wins on validation F1",
      default["val_f1"] == tuning["val_f1"].max())

# Threshold selection
check("Nine thresholds tested", len(thresholds) == 9)
check(
    "Recall does not increase as threshold increases",
    thresholds["recall"].is_monotonic_decreasing,
)

eligible = thresholds[thresholds["recall"] >= 0.80]
expected_threshold = (
    eligible["threshold"].max() if len(eligible) else 0.1
)

chosen_threshold = float(config.iloc[0]["threshold"])

check("Chosen threshold follows recall rule",
      chosen_threshold == expected_threshold)

# Final configuration and test results
check("Final test has exactly one row", len(test) == 1)
check("Test threshold matches final configuration",
      test.iloc[0]["threshold"] == chosen_threshold)
check("Final selector uses seven features",
      config.iloc[0]["selector_k"] == 7)
check("Final configuration uses 100 trees",
      config.iloc[0]["n_estimators"] == 100)
check("Final configuration records default forest",
      config.iloc[0]["chosen_from"] == "default_forest_same_7_features")

print("\nPART C ACCEPTANCE CHECKS")
passed = 0

for name, result in checks:
    print(f"{'PASS' if result else 'FAIL'} - {name}")
    passed += result

print(f"\nResult: {passed}/{len(checks)} checks passed.")

if passed != len(checks):
    raise SystemExit("Some Part C checks failed.")
