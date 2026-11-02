import numpy as np
import pandas as pd

checks = pd.read_csv("outputs/part_b_checks.csv").set_index("metric")["value"]
tree = pd.read_csv("outputs/part_b_tree_depth.csv")
metrics = pd.read_csv("outputs/part_b_metrics.csv")
gb = pd.read_csv("outputs/part_b_gb_learning_rate.csv")
cv = pd.read_csv("outputs/part_b_cv.csv")

results = []

def check(name, condition):
    results.append((name, bool(condition)))

# Data split and scaling
check("Training rows = 840", checks["train_rows"] == 840)
check("Validation rows = 180", checks["val_rows"] == 180)
check("Test rows = 180", checks["test_rows"] == 180)
check("Training scaling has no leakage", checks["train_scaled_max_abs_mean"] < 1e-9)
check("Test scaling uses training scaler", checks["test_scaled_max_abs_mean"] > 0.01)

# Decision tree depth experiment
check("Eight tree depths evaluated", len(tree) == 8)
check("Training RMSE decreases with depth", tree["train_rmse"].is_monotonic_decreasing)
depth1 = tree[tree["max_depth"] == "1"].iloc[0]
unlimited = tree[tree["max_depth"] == "unlimited"].iloc[0]
best_tree = tree.loc[tree["val_rmse"].idxmin()]

check("Depth 1 shows high bias", depth1["val_rmse"] - depth1["train_rmse"] <= 2.5)
check("Unlimited tree shows overfitting", unlimited["val_rmse"] - unlimited["train_rmse"] >= 5.0)
check("Depth 1 is worse than best depth by at least 4", depth1["val_rmse"] - best_tree["val_rmse"] >= 4)
check("Best tree depth is at least 4", best_tree["max_depth"] == "unlimited" or int(best_tree["max_depth"]) >= 4)

tree_model = metrics[
    (metrics["model"] == "decision_tree") &
    (metrics["split"] == "val")
].iloc[0]
check("Decision tree uses selected depth", np.isclose(tree_model["RMSE"], best_tree["val_rmse"]))

# Gradient Boosting learning rate
check("Three learning rates evaluated", len(gb) == 3)
r = gb.set_index("learning_rate")["val_r2"]
check("Learning rate R2 ordering", r[0.1] > r[1.0] > r[0.01])
check("Learning rate 0.1 R2 >= 0.90", r[0.1] >= 0.90)
check("Learning rate 0.01 R2 <= 0.80", r[0.01] <= 0.80)

# Cross-validation
check("Two models cross-validated", len(cv) == 2)
check("CV mean R2 >= 0.90", (cv["mean_r2"] >= 0.90).all())
check("CV standard deviation <= 0.03", (cv["std_r2"] <= 0.03).all())

# Final model metrics
check("Metrics CSV has six rows", len(metrics) == 6)
check("RMSE equals square root of MSE", np.allclose(metrics["RMSE"], np.sqrt(metrics["MSE"]), atol=0.001))
check("Exactly one final test evaluation", len(metrics[metrics["split"] == "test"]) == 1)

winner = metrics[metrics["split"] == "val"].sort_values("RMSE").iloc[0]
test_result = metrics[metrics["split"] == "test"].iloc[0]

check("Test model matches validation winner", test_result["model"] == winner["model"])
check("Final test R2 >= 0.88", test_result["R2"] >= 0.88)

# README
with open("README.md", encoding="utf-8") as f:
    readme = f.read()

check("README contains Part B results", "Part B — Delivery-Hours Regression" in readme)

print("\nPART B ACCEPTANCE CHECKS")
print("=" * 55)

for name, passed in results:
    print(f"{'PASS' if passed else 'FAIL'} | {name}")

passed_count = sum(passed for _, passed in results)
print("=" * 55)
print(f"RESULT: {passed_count}/{len(results)} checks passed")

if passed_count != len(results):
    raise SystemExit("Some acceptance checks failed.")
