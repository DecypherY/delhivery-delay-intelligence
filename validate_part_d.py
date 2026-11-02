import os
import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES


def check(condition, description):
    if not condition:
        raise AssertionError(f"FAILED: {description}")
    print(f"PASS: {description}")


# 1. Hub profile
profile = pd.read_csv("outputs/hub_profile.csv")

check(len(profile) == 24, "Exactly 24 hubs in hub profile")
check((profile["shipments"] == 50).all(), "50 shipments per hub")
check(
    np.isclose(
        profile.loc[profile.hub_id == "HUB01", "mean_delivery_hours"].iloc[0],
        45.685,
        atol=0.01,
    ),
    "HUB01 delivery-hours spot check",
)
check(
    np.isclose(
        profile.loc[profile.hub_id == "HUB13", "delay_rate"].iloc[0],
        0.46,
        atol=0.01,
    ),
    "HUB13 delay-rate spot check",
)

# 2. K-Means separation
separation = pd.read_csv("outputs/part_d_separation.csv")

expected_sizes = ["15-9", "9-8-7", "8-7-5-4", "8-7-4-3-2"]
expected_ratios = [2.894, 5.393, 6.305, 6.887]

check(
    separation["cluster_sizes"].tolist() == expected_sizes,
    "K-Means cluster sizes for K=2,3,4,5",
)
check(
    np.allclose(
        separation["across_over_within"],
        expected_ratios,
        atol=0.01,
    ),
    "K-Means separation ratios",
)

# 3. Named segments
segments = pd.read_csv("outputs/hub_segments.csv")
segment_profile = pd.read_csv("outputs/segment_profile.csv")

check(len(segments) == 24, "All 24 hubs assigned a segment")
check(
    segments["segment"].value_counts().to_dict()
    == {"Escalate": 9, "Watch": 8, "Standard": 7},
    "Segment counts: Standard 7, Watch 8, Escalate 9",
)

expected_rates = {
    "Standard": 0.0257,
    "Watch": 0.1550,
    "Escalate": 0.4600,
}

for segment, expected_rate in expected_rates.items():
    actual_rate = segment_profile.loc[
        segment_profile["segment"] == segment,
        "delay_rate",
    ].iloc[0]

    check(
        np.isclose(actual_rate, expected_rate, atol=0.01),
        f"{segment} segment delay rate",
    )

# 4. Tier mismatches
mismatches = pd.read_csv("outputs/tier_vs_segment_mismatch.csv")

expected_mismatches = {
    ("HUB07", "Kolkata"),
    ("HUB13", "Nagpur"),
    ("HUB15", "Kochi"),
    ("HUB22", "Dehradun"),
}

actual_mismatches = set(
    zip(mismatches["hub_id"], mismatches["hub_city"])
)

check(
    actual_mismatches == expected_mismatches,
    "Exactly four expected tier-versus-segment mismatches",
)

# 5. Model artifacts
model_files = [
    "models/delay_model.pkl",
    "models/delay_threshold.csv",
    "models/feature_order.csv",
    "models/impute_values.csv",
]

for file in model_files:
    check(os.path.exists(file), f"Model artifact exists: {file}")

feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()

check(feature_order == FEATURES, "Saved feature order matches training features")

# 6. Model reload verification
reload_check = pd.read_csv("outputs/part_d_reload_check.csv")
reload_values = dict(zip(reload_check["metric"], reload_check["value"]))

check(
    reload_values["test_rows_scored"] == 180,
    "180 test rows scored during reload verification",
)
check(
    reload_values["reload_max_abs_probability_diff"] <= 1e-9,
    "Reloaded model probability difference <= 1e-9",
)

# 7. New shipment scoring
actions = pd.read_csv("outputs/new_shipment_actions.csv")

required_columns = [
    "shipment_id",
    "hub_id",
    "segment",
    "delay_probability",
    "flagged",
    "action",
]

check(len(actions) == 48, "48 new shipments scored")
check(actions.columns.tolist() == required_columns, "Scoring output columns and order")
check(actions["shipment_id"].nunique() == 48, "All shipment IDs unique")
check(actions.isna().sum().sum() == 0, "No missing values in scoring output")
check(
    actions["delay_probability"].between(0, 1).all(),
    "All probabilities between 0 and 1",
)

threshold = float(
    pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0]
)

expected_flags = (actions["delay_probability"] >= threshold).astype(int)

check(
    np.array_equal(actions["flagged"], expected_flags),
    "Flags agree with saved classification threshold",
)

action_counts = actions["action"].value_counts().to_dict()

check(
    action_counts == {
        "No action": 40,
        "Escalate to hub manager": 6,
        "Proactive customer call": 2,
    },
    "Expected operational action counts",
)

print("\nAll Part D validation checks passed!")
