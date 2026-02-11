import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="News Topic Discovery Dashboard",
    layout="wide"
)

st.title("🟣 News Topic Discovery Dashboard")
st.markdown("""
This system uses **Hierarchical Clustering** to automatically group similar news articles based on textual similarity.

👉 Discover hidden themes without defining categories upfront.
""")

# =========================================================
# SIDEBAR - DATA UPLOAD
# =========================================================

st.sidebar.header("📂 Dataset Upload")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file to begin.")
    st.stop()

# Read file safely (handles messy comma text)
raw_df = pd.read_csv(
    uploaded_file,
    encoding="latin1",
    header=None,
    engine="python"
)

# Combine text columns properly
if raw_df.shape[1] > 2:
    sentiment_col = raw_df.iloc[:, 0]
    text_col = raw_df.iloc[:, 1:].astype(str).agg(" ".join, axis=1)

    df = pd.DataFrame({
        "sentiment": sentiment_col,
        "text": text_col
    })
else:
    df = raw_df.copy()
    df.columns = ["sentiment", "text"]

# Clean text
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)

st.sidebar.success("Dataset Loaded Successfully")

# =========================================================
# SIDEBAR - TEXT VECTORIZATION
# =========================================================

st.sidebar.header("📝 Text Vectorization Settings")

max_features = st.sidebar.slider(
    "Maximum TF-IDF Features",
    100, 2000, 1000
)

use_stopwords = st.sidebar.checkbox(
    "Remove English Stopwords",
    value=True
)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1, 1)
elif ngram_option == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)

# =========================================================
# SIDEBAR - CLUSTER SETTINGS
# =========================================================

st.sidebar.header("🌳 Hierarchical Clustering Settings")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

subset_size = st.sidebar.slider(
    "Articles for Dendrogram",
    20, min(200, len(df)), 100
)

num_clusters = st.sidebar.slider(
    "Number of Clusters",
    2, 10, 3
)

# =========================================================
# TF-IDF VECTORIZE
# =========================================================

tfidf = TfidfVectorizer(
    stop_words="english" if use_stopwords else None,
    max_features=max_features,
    ngram_range=ngram_range
)

X = tfidf.fit_transform(df["text"])

st.write("🔢 TF-IDF Feature Count:", X.shape[1])

# =========================================================
# SAFE DIMENSIONALITY REDUCTION
# =========================================================

n_features = X.shape[1]
n_components = min(100, n_features - 1)

if n_components > 1:
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
else:
    X_reduced = X.toarray()

# =========================================================
# DENDROGRAM SECTION
# =========================================================

if st.button("🟦 Generate Dendrogram"):

    st.subheader("📊 Dendrogram")

    X_subset = X_reduced[:subset_size]

    linked = linkage(X_subset, method=linkage_method)

    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(linked, ax=ax)
    ax.set_xlabel("Articles")
    ax.set_ylabel("Distance")

    st.pyplot(fig)

    st.info("Inspect large vertical gaps to determine natural cluster separation.")

# =========================================================
# APPLY CLUSTERING
# =========================================================

if st.button("🟩 Apply Clustering"):

    model = AgglomerativeClustering(
        n_clusters=num_clusters,
        linkage=linkage_method
    )

    clusters = model.fit_predict(X_reduced)

    df["Cluster"] = clusters

    # -----------------------------------------------------
    # SILHOUETTE SCORE
    # -----------------------------------------------------

    score = silhouette_score(X_reduced, clusters)

    st.subheader("📊 Silhouette Score")
    st.metric("Score", round(score, 4))

    if score > 0.5:
        st.success("Well-separated clusters")
    elif score > 0.2:
        st.warning("Moderate cluster separation")
    else:
        st.error("Clusters overlap significantly")

    # -----------------------------------------------------
    # PCA VISUALIZATION
    # -----------------------------------------------------

    st.subheader("📈 Cluster Visualization (2D Projection)")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_reduced)

    plot_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": clusters,
        "Snippet": df["text"].str[:120]
    })

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=["Snippet"],
        title="PCA Projection of Clusters"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # CLUSTER SUMMARY
    # -----------------------------------------------------

    st.subheader("📋 Cluster Summary")

    terms = tfidf.get_feature_names_out()
    summary_data = []

    for i in range(num_clusters):
        cluster_indices = np.where(clusters == i)[0]
        cluster_tfidf = X[cluster_indices]

        mean_tfidf = cluster_tfidf.mean(axis=0)
        top_words = np.argsort(mean_tfidf).tolist()[0][-10:]
        keywords = [terms[word] for word in top_words]

        snippet = df[df["Cluster"] == i]["text"].iloc[0][:200]

        summary_data.append({
            "Cluster ID": i,
            "Articles": len(cluster_indices),
            "Top Keywords": ", ".join(keywords),
            "Sample Snippet": snippet
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

    # -----------------------------------------------------
    # BUSINESS INTERPRETATION
    # -----------------------------------------------------

    st.subheader("🧠 Editorial Interpretation")

    for row in summary_data:
        st.markdown(f"""
        🔵 **Cluster {row['Cluster ID']}**  
        Articles in this group mainly discuss:  
        _{row['Top Keywords']}_
        """)

    # -----------------------------------------------------
    # USER GUIDANCE
    # -----------------------------------------------------

    st.info("""
Articles grouped in the same cluster share similar vocabulary and themes.  
These clusters can be used for automatic tagging, recommendations, and content organization.
""")
