import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def create_hub_profile():
    # Load the modelling table created in Part A
    df = pd.read_csv("data/modelling_table.csv")

    # Calculate operational metrics for every hub
    hub_profile = (
        df.groupby("hub_id")
        .agg(
            shipments=("shipment_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .reset_index()
        .sort_values("hub_id")
    )

    hub_profile.to_csv("outputs/hub_profile.csv", index=False)

    print("\nHub profile:")
    print(hub_profile.to_string(index=False))
    print(f"\nTotal hubs: {len(hub_profile)}")

    return hub_profile


def create_clusters_and_separation(hub_profile):
    metrics = [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops",
    ]

    # Scale hub-level metrics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(hub_profile[metrics])

    # Required three-cluster model
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )

    hub_profile["cluster"] = kmeans.fit_predict(X_scaled)

    # Evaluate separation for K = 2, 3, 4, 5
    separation_results = []

    for k in [2, 3, 4, 5]:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X_scaled)

        within_distances = []
        across_distances = []

        for i, j in combinations(range(len(X_scaled)), 2):
            distance = np.linalg.norm(X_scaled[i] - X_scaled[j])

            if labels[i] == labels[j]:
                within_distances.append(distance)
            else:
                across_distances.append(distance)

        cluster_sizes = sorted(
            pd.Series(labels).value_counts().tolist(),
            reverse=True,
        )

        ratio = np.mean(across_distances) / np.mean(within_distances)

        separation_results.append({
            "K": k,
            "across_over_within": ratio,
            "cluster_sizes": "-".join(map(str, cluster_sizes)),
        })

    separation_df = pd.DataFrame(separation_results)
    separation_df.to_csv("outputs/part_d_separation.csv", index=False)

    print("\nK-Means cluster assignments:")
    print(hub_profile[["hub_id", "cluster"]].to_string(index=False))

    print("\nCluster separation:")
    print(separation_df.to_string(index=False))




def create_named_segments(hub_profile):
    # Calculate average delay rate for each cluster
    cluster_delay = (
        hub_profile.groupby("cluster")["delay_rate"]
        .mean()
        .sort_values()
    )

    # Assign business names based on delay severity
    segment_names = {
        cluster_delay.index[0]: "Standard",
        cluster_delay.index[1]: "Watch",
        cluster_delay.index[2]: "Escalate",
    }

    hub_profile["segment"] = hub_profile["cluster"].map(segment_names)

    # Load hub details
    hubs = pd.read_csv("data/hubs.csv")

    # Create hub-to-segment mapping
    hub_segments = hub_profile[["hub_id", "segment"]].merge(
        hubs[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left",
    )

    hub_segments.to_csv("outputs/hub_segments.csv", index=False)

    # Calculate segment-level operational profiles
    segment_profile = (
        hub_profile.groupby("segment")
        .agg(
            hubs=("hub_id", "count"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean"),
        )
        .reset_index()
    )

    segment_order = ["Standard", "Watch", "Escalate"]
    segment_profile["segment"] = pd.Categorical(
        segment_profile["segment"],
        categories=segment_order,
        ordered=True,
    )
    segment_profile = segment_profile.sort_values("segment")
    segment_profile.to_csv("outputs/segment_profile.csv", index=False)

    # Compare actual segments with expected tier assignments
    tier_mapping = {
        1: "Standard",
        2: "Watch",
        3: "Escalate",
    }

    hub_segments["expected_segment"] = hub_segments["hub_tier"].map(tier_mapping)

    mismatches = hub_segments[
        hub_segments["segment"] != hub_segments["expected_segment"]
    ][["hub_id", "hub_city", "hub_tier", "segment", "expected_segment"]]

    mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

    print("\nNamed hub segments:")
    print(hub_segments.to_string(index=False))

    print("\nSegment profiles:")
    print(segment_profile.to_string(index=False))

    print("\nTier vs segment mismatches:")
    print(mismatches.to_string(index=False))

if __name__ == "__main__":
    hub_profile = create_hub_profile()
    create_clusters_and_separation(hub_profile)
    create_named_segments(hub_profile)
