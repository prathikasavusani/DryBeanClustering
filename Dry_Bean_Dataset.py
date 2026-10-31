
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

warnings.filterwarnings("ignore")

# -------------------- SETTINGS --------------------

DATA_FILE = "Dry_Bean_Dataset.xlsx"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
K = 7

# -------------------- LOAD DATA --------------------

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Cannot find {DATA_FILE}. Put it in the same folder as this script."
    )

df = pd.read_excel(DATA_FILE)
df.columns = df.columns.str.strip()

print("=" * 60)
print("DRY BEAN UNSUPERVISED CLUSTERING")
print("=" * 60)
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:", df.isnull().sum().sum())

# Keep class labels separate. Never use them to train clustering.
label_column = next(
    (c for c in df.columns if c.lower() == "class"), None
)

if label_column:
    true_labels = df[label_column].copy()
    X_df = df.drop(columns=[label_column])
else:
    true_labels = None
    X_df = df.copy()

# Only numerical measurements are used for clustering.
X_df = X_df.select_dtypes(include=np.number)
X_df = X_df.replace([np.inf, -np.inf], np.nan)
X_df = X_df.fillna(X_df.median())

# Standardization prevents large-scale features dominating distances.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

# Use a reproducible sample for the algorithms and visualizations.
rng = np.random.RandomState(RANDOM_STATE)
sample_size = min(3000, len(X_scaled))
indices = rng.choice(len(X_scaled), sample_size, replace=False)

X = X_scaled[indices]
sample_labels = (
    true_labels.iloc[indices].reset_index(drop=True)
    if true_labels is not None else None
)

print("\nFeatures used:", X_df.shape[1])
print("Rows used for clustering:", len(X))
print("Class labels are excluded from clustering.")

# -------------------- EVALUATION --------------------

results = []

def evaluate(name, labels, data):
    labels = np.asarray(labels)
    unique = set(labels)
    n_clusters = len(unique - {-1})
    noise_count = int(np.sum(labels == -1))

    silhouette = np.nan
    db_index = np.nan

    # Silhouette and Davies-Bouldin require at least
    # two clusters, and cannot treat DBSCAN noise as a cluster.
    valid = labels != -1
    valid_labels = labels[valid]
    valid_data = data[valid]

    if (
        len(set(valid_labels)) >= 2
        and len(set(valid_labels)) < len(valid_labels)
    ):
        try:
            silhouette = silhouette_score(
                valid_data, valid_labels
            )
            db_index = davies_bouldin_score(
                valid_data, valid_labels
            )
        except ValueError:
            pass

    results.append({
        "Algorithm": name,
        "Clusters": n_clusters,
        "Noise points": noise_count,
        "Silhouette score": silhouette,
        "Davies-Bouldin index": db_index
    })

    print(f"\n{name}")
    print("Clusters:", n_clusters)
    print("Noise points:", noise_count)
    print("Silhouette score:", silhouette)
    print("Davies-Bouldin index:", db_index)

# -------------------- 1. ELBOW METHOD --------------------

inertias = []
k_values = range(2, 11)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=RANDOM_STATE
    )
    model.fit(X)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method for Dry Bean Clustering")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_elbow.png", dpi=200)
plt.close()

# -------------------- 2. K-MEANS++ --------------------

kmeans = KMeans(
    n_clusters=K,
    init="k-means++",
    n_init=10,
    random_state=RANDOM_STATE
)
km_labels = kmeans.fit_predict(X)
evaluate("K-Means++", km_labels, X)

# -------------------- PCA VISUALIZATION --------------------

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)

print(
    "\nPCA variance retained:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

def plot_clusters(points, labels, title, filename):
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        points[:, 0], points[:, 1],
        c=labels, cmap="tab10", s=10, alpha=0.7
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=200)
    plt.close()

plot_clusters(
    X_pca, km_labels,
    "K-Means++ Clusters Visualized Using PCA",
    "02_kmeans_pca.png"
)

# -------------------- 3. HIERARCHICAL CLUSTERING --------------------

# Limit this part to 1000 samples to keep it manageable.
hier_size = min(1000, len(X))
X_hier = X[:hier_size]

hierarchical = AgglomerativeClustering(
    n_clusters=K,
    linkage="ward"
)
hier_labels = hierarchical.fit_predict(X_hier)
evaluate("Hierarchical (Ward)", hier_labels, X_hier)

hier_pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_hier_pca = hier_pca.fit_transform(X_hier)

plot_clusters(
    X_hier_pca, hier_labels,
    "Hierarchical Clustering (PCA)",
    "03_hierarchical_pca.png"
)

# Dendrogram on a smaller subset for readable output.
dendro_size = min(150, len(X_hier))
Z = linkage(X_hier[:dendro_size], method="ward")

plt.figure(figsize=(12, 6))
dendrogram(Z, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Bean samples")
plt.ylabel("Linkage distance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_dendrogram.png", dpi=200)
plt.close()

# -------------------- 4. DBSCAN --------------------

# Use the first 10 PCA components to reduce dimensionality.
pca10 = PCA(n_components=10, random_state=RANDOM_STATE)
X_pca10 = pca10.fit_transform(X)

dbscan = DBSCAN(eps=2.5, min_samples=10)
db_labels = dbscan.fit_predict(X_pca10)

evaluate("DBSCAN", db_labels, X_pca10)

plot_clusters(
    X_pca, db_labels,
    "DBSCAN Clusters and Noise (PCA)",
    "05_dbscan_pca.png"
)

# -------------------- 5. K-MEDOIDS --------------------

# scikit-learn-extra supplies the K-Medoids implementation.
# Its installation can be incompatible with some Python versions.
try:
    from sklearn_extra.cluster import KMedoids

    medoid_size = min(1000, len(X))
    X_medoid = X[:medoid_size]

    medoids = KMedoids(
        n_clusters=K,
        metric="euclidean",
        init="k-medoids++",
        random_state=RANDOM_STATE
    )
    medoid_labels = medoids.fit_predict(X_medoid)
    evaluate("K-Medoids", medoid_labels, X_medoid)

    medoid_pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_medoid_pca = medoid_pca.fit_transform(X_medoid)

    plot_clusters(
        X_medoid_pca, medoid_labels,
        "K-Medoids Clusters (PCA)",
        "06_kmedoids_pca.png"
    )

except ImportError:
    print(
        "\nK-Medoids skipped: scikit-learn-extra could not be imported."
    )
    print(
        "Try installing a compatible version of scikit-learn-extra "
        "for your Python environment."
    )

# -------------------- 6. t-SNE --------------------

tsne_size = min(2000, len(X))
X_tsne_input = X[:tsne_size]
tsne_labels = km_labels[:tsne_size]

tsne = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=RANDOM_STATE
)
X_tsne = tsne.fit_transform(X_tsne_input)

plot_clusters(
    X_tsne, tsne_labels,
    "K-Means++ Clusters Visualized Using t-SNE",
    "07_tsne.png"
)

# -------------------- 7. OPTIONAL LABEL COMPARISON --------------------

# Labels are used only after clustering, for interpretation.
if sample_labels is not None:
    comparison = pd.crosstab(
        pd.Series(km_labels, name="Cluster"),
        sample_labels.rename("Known bean variety")
    )
    comparison.to_csv(
        f"{OUTPUT_DIR}/cluster_vs_known_variety.csv"
    )
    print("\nCluster versus known bean variety:")
    print(comparison)

# -------------------- SAVE RESULTS --------------------

report = pd.DataFrame(results)
report.to_csv(f"{OUTPUT_DIR}/clustering_comparison.csv", index=False)

print("\n" + "=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)
print(report.to_string(index=False))
print(f"\nAll generated files are saved in: {OUTPUT_DIR}")
