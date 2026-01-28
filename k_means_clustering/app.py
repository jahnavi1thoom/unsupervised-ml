import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# --------------------------------------------------
# Title & Description
# --------------------------------------------------
st.title("🟢 Customer Segmentation Dashboard")
st.write(
    "This system uses **K-Means Clustering** to group customers based on their "
    "purchasing behavior and similarities."
)
st.caption("👉 Discover hidden customer groups without predefined labels.")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "Wholesale customers data.csv")
df = pd.read_csv(DATA_PATH)


# Keep only numeric columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# --------------------------------------------------
# Sidebar – Inputs
# --------------------------------------------------
st.sidebar.header("⚙️ Clustering Controls")

feature_1 = st.sidebar.selectbox("Select Feature 1", numeric_cols)
feature_2 = st.sidebar.selectbox(
    "Select Feature 2",
    [col for col in numeric_cols if col != feature_1]
)

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=3
)

random_state = st.sidebar.number_input(
    "Random State (Optional)",
    value=42,
    step=1
)

run_clustering = st.sidebar.button("🟦 Run Clustering")

# --------------------------------------------------
# Main Panel
# --------------------------------------------------
if run_clustering:

    # ----------------------------
    # Data Preparation
    # ----------------------------
    X = df[[feature_1, feature_2]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----------------------------
    # K-Means Clustering
    # ----------------------------
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=random_state,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_scaled)
    df["Cluster"] = cluster_labels

    # ----------------------------
    # Visualization (SMALL & CLEAN)
    # ----------------------------
    st.subheader("📊 Cluster Visualization")

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=cluster_labels,
        cmap="viridis",
        s=35,
        alpha=0.7
    )

    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        c="red",
        s=80,
        marker="X",
        label="Cluster Centers"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    
    ax.set_title("Customer Clusters", fontsize=10)
    ax.legend(fontsize=8)

    ax.tick_params(axis='both', labelsize=8)

    st.pyplot(fig,use_container_width=False)

    # ----------------------------
    # Cluster Summary (SAFE VERSION)
    # ----------------------------
    st.subheader("📋 Cluster Summary")

    summary = df.groupby("Cluster").agg(
        Number_of_Customers=("Cluster", "count"),
        Avg_Feature_1=(feature_1, "mean"),
        Avg_Feature_2=(feature_2, "mean")
    ).round(2)

    # Rename for display
    summary = summary.rename(columns={
        "Avg_Feature_1": f"Average {feature_1}",
        "Avg_Feature_2": f"Average {feature_2}"
    })

    st.dataframe(summary, use_container_width=True)

    # ----------------------------
    # Business Interpretation
    # ----------------------------
    st.subheader("💼 Business Interpretation")

    overall_avg_1 = summary[f"Average {feature_1}"].mean()
    overall_avg_2 = summary[f"Average {feature_2}"].mean()

    for cluster_id in summary.index:
        avg_1 = summary.loc[cluster_id, f"Average {feature_1}"]
        avg_2 = summary.loc[cluster_id, f"Average {feature_2}"]

        if avg_1 > overall_avg_1 and avg_2 > overall_avg_2:
            insight = "High-spending customers across selected categories"
            color = "🟢"
        elif avg_1 < overall_avg_1 and avg_2 < overall_avg_2:
            insight = "Budget-conscious customers with low spending"
            color = "🟡"
        else:
            insight = "Moderate spenders with selective purchasing behavior"
            color = "🔵"

        st.write(f"{color} **Cluster {cluster_id}:** {insight}")

    # ----------------------------
    # User Guidance Box
    # ----------------------------
    st.info(
        "Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
    )

else:
    st.warning("⬅️ Select features and click **Run Clustering** to view results.")
