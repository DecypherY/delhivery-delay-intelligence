import joblib
import pandas as pd

from src.shipment_prep import FEATURES, clean_shipments, build_features


def predict_new_shipments():
    # 1. Load saved model artifacts
    model = joblib.load("models/delay_model.pkl")

    threshold = float(
        pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0]
    )

    feature_order = pd.read_csv(
        "models/feature_order.csv"
    )["feature"].tolist()

    impute_df = pd.read_csv("models/impute_values.csv")

    medians = dict(
        zip(impute_df["column"], impute_df["median"])
    )

    # 2. Load new shipments and hub information
    raw_shipments = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # 3. Clean shipments using historical imputation medians
    clean, _ = clean_shipments(
        raw_shipments,
        medians=medians,
    )

    # 4. Build features using the shared preprocessing function
    featured = build_features(clean, hubs)

    # 5. Arrange features in the exact training order
    X_new = featured[feature_order]

    # 6. Predict probability of delay
    probabilities = model.predict_proba(X_new)[:, 1]

    # 7. Apply the saved classification threshold
    flagged = (probabilities >= threshold).astype(int)

    # 8. Attach hub segment information
    results = featured[["shipment_id", "hub_id"]].merge(
        pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]],
        on="hub_id",
        how="left",
        validate="many_to_one",
    )

    results["delay_probability"] = probabilities.round(4)
    results["flagged"] = flagged

    # 9. Recommend operational action
    results["action"] = "No action"

    results.loc[
        (results["flagged"] == 1) &
        (results["segment"] == "Escalate"),
        "action",
    ] = "Escalate to hub manager"

    results.loc[
        (results["flagged"] == 1) &
        (results["segment"] != "Escalate"),
        "action",
    ] = "Proactive customer call"

    # 10. Save required output columns
    output = results[
        [
            "shipment_id",
            "hub_id",
            "segment",
            "delay_probability",
            "flagged",
            "action",
        ]
    ]

    output.to_csv(
        "outputs/new_shipment_actions.csv",
        index=False,
    )

    # 11. Display scoring summary
    print("\nShipment scoring completed.")

    print("Total shipments:", len(output))
    print("Flagged shipments:", int(output["flagged"].sum()))

    print("\nOperational action counts:")
    print(output["action"].value_counts().to_string())

    print("\nSegment-wise results:")
    print(
        output.groupby("segment", observed=True)
        .agg(
            shipments=("shipment_id", "count"),
            flagged=("flagged", "sum"),
        )
        .to_string()
    )

    print("\nFirst 10 scored shipments:")
    print(output.head(10).to_string(index=False))

    print("\nSaved: outputs/new_shipment_actions.csv")


if __name__ == "__main__":
    predict_new_shipments()
