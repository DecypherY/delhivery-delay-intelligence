import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import FEATURES, clean_shipments


def package_model():
    # --------------------------------------------------
    # 1. Read the selected configuration from Part C
    # --------------------------------------------------
    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]

    selector_k = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])
    max_depth = None if config["max_depth"] == "unlimited" else int(config["max_depth"])
    min_samples_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    print("Selected model configuration:")
    print(config.to_string())

    # --------------------------------------------------
    # 2. Load the modelling dataset
    # --------------------------------------------------
    data = pd.read_csv("data/modelling_table.csv")

    X = data[FEATURES]
    y = data["is_delayed"]

    # Recreate the exact Part C split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    print("\nRecreated dataset split:")
    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Test rows:", len(X_test))

    # --------------------------------------------------
    # 3. Build and train the final classification pipeline
    # --------------------------------------------------
    final_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=selector_k)),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )),
    ])

    final_model.fit(X_train, y_train)

    # --------------------------------------------------
    # 4. Create the models directory
    # --------------------------------------------------
    import os
    os.makedirs("models", exist_ok=True)

    # Save the trained model
    joblib.dump(final_model, "models/delay_model.pkl")

    # Save the classification threshold
    pd.DataFrame({
        "threshold": [threshold]
    }).to_csv("models/delay_threshold.csv", index=False)

    # Save the exact feature order
    pd.DataFrame({
        "feature": FEATURES
    }).to_csv("models/feature_order.csv", index=False)

    # --------------------------------------------------
    # 5. Calculate and save imputation medians
    # --------------------------------------------------
    raw_data = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw_data)

    impute_values = pd.DataFrame([
        {"column": column, "median": median}
        for column, median in medians.items()
    ])

    impute_values.to_csv("models/impute_values.csv", index=False)

    # --------------------------------------------------
    # 6. Reload the model and verify probabilities
    # --------------------------------------------------
    reloaded_model = joblib.load("models/delay_model.pkl")

    original_probabilities = final_model.predict_proba(X_test)[:, 1]
    reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

    max_difference = np.max(
        np.abs(original_probabilities - reloaded_probabilities)
    )

    # --------------------------------------------------
    # 7. Save reload verification results
    # --------------------------------------------------
    reload_check = pd.DataFrame({
        "metric": [
            "test_rows_scored",
            "reload_max_abs_probability_diff",
            "threshold",
        ],
        "value": [
            len(X_test),
            max_difference,
            threshold,
        ],
    })

    reload_check.to_csv(
        "outputs/part_d_reload_check.csv",
        index=False,
    )

    print("\nModel packaging complete.")
    print("\nReload verification:")
    print(reload_check.to_string(index=False))

    print("\nSaved model artifacts:")
    print("models/delay_model.pkl")
    print("models/delay_threshold.csv")
    print("models/feature_order.csv")
    print("models/impute_values.csv")


if __name__ == "__main__":
    package_model()
