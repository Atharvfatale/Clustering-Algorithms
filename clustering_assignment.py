import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from scipy.cluster.hierarchy import dendrogram, linkage

RS = 42
out = {}

# 1. LOAD DATASET
data = load_wine(as_frame=True)
df = data.frame.copy()
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head().to_string())
print("\nMissing values:", int(df.isnull().sum().sum()))

# 2. SEPARATE FEATURES (labels kept ONLY for external validation)
y_true = df["target"].values
X = df.drop(columns=["target"])
print("\nFeatures:", X.columns.tolist())
print("Class sizes (not used for clustering):", np.bincount(y_true).tolist())

# 3. SCALING
X_scaled = StandardScaler().fit_transform(X)

# 4. CHOOSING K: elbow + silhouette + Davies-Bouldin for k = 2..10
ks = list(range(2, 11))
inertia, sil, dbi, chi = [], [], [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, km.labels_))
    dbi.append(davies_bouldin_score(X_scaled, km.labels_))
    chi.append(calinski_harabasz_score(X_scaled, km.labels_))
sel = pd.DataFrame({"k": ks, "Inertia": inertia, "Silhouette": sil,
                    "Davies_Bouldin": dbi, "Calinski_Harabasz": chi})
print("\nK SELECTION")
print(sel.round(4).to_string(index=False))
best_k = int(ks[int(np.argmax(sil))])
print("\nBest k by silhouette:", best_k)
out["best_k"] = best_k
out["k_selection"] = sel.round(4).to_dict(orient="records")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow Method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia (WCSS)")
ax[1].plot(ks, sil, "o-", color="green"); ax[1].set_title("Silhouette Score vs k"); ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette")
ax[2].plot(ks, dbi, "o-", color="red"); ax[2].set_title("Davies-Bouldin Index vs k"); ax[2].set_xlabel("k"); ax[2].set_ylabel("DBI (lower is better)")
for a in ax: a.grid(True)
plt.tight_layout(); plt.savefig("fig1_k_selection.png", dpi=150); plt.close()

# 5. MODELS
results = []
labels_store = {}

def evaluate(name, labels, extra=""):
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    n_noise = int((~mask).sum())
    if n_clusters >= 2:
        s = silhouette_score(X_scaled[mask], labels[mask])
        d = davies_bouldin_score(X_scaled[mask], labels[mask])
        c = calinski_harabasz_score(X_scaled[mask], labels[mask])
    else:
        s = d = c = float("nan")
    ari = adjusted_rand_score(y_true, labels)
    results.append({"Method": name, "Clusters": n_clusters, "Noise": n_noise,
                    "Silhouette": s, "Davies_Bouldin": d,
                    "Calinski_Harabasz": c, "ARI": ari, "Note": extra})
    labels_store[name] = labels

# 5a K-Means
km = KMeans(n_clusters=best_k, n_init=10, random_state=RS).fit(X_scaled)
evaluate("K-Means", km.labels_)

# 5b Hierarchical (Ward) + linkage comparison
for link in ["ward", "complete", "average", "single"]:
    ag = AgglomerativeClustering(n_clusters=best_k, linkage=link).fit(X_scaled)
    evaluate(f"Hierarchical ({link})", ag.labels_)

# 5c GMM
gmm = GaussianMixture(n_components=best_k, covariance_type="full", n_init=5, random_state=RS).fit(X_scaled)
evaluate("Gaussian Mixture", gmm.predict(X_scaled))

# 5d DBSCAN: eps from k-distance graph (min_samples = 2*dim rule is too strict for 178 pts -> use 5)
min_samples = 5
nn = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
kdist = np.sort(nn.kneighbors(X_scaled)[0][:, -1])
plt.figure(figsize=(7, 4)); plt.plot(kdist); plt.xlabel("Points sorted by distance"); plt.ylabel(f"{min_samples}-NN distance")
plt.title("k-distance Graph for DBSCAN eps"); plt.grid(True); plt.tight_layout()
plt.savefig("fig2_kdistance.png", dpi=150); plt.close()

grid = []
for eps in np.arange(1.5, 3.55, 0.25):
    lab = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(X_scaled)
    m = lab != -1
    nc = len(set(lab[m]))
    s = silhouette_score(X_scaled[m], lab[m]) if nc >= 2 else float("nan")
    grid.append({"eps": round(float(eps), 2), "Clusters": nc, "Noise": int((~m).sum()), "Silhouette": s})
grid_df = pd.DataFrame(grid)
print("\nDBSCAN GRID (min_samples=5)")
print(grid_df.round(4).to_string(index=False))
out["dbscan_grid"] = grid_df.round(4).fillna("NA").to_dict(orient="records")
valid = grid_df[(grid_df.Clusters >= 2) & (grid_df.Noise <= 0.3 * len(X_scaled))]
if len(valid):
    best_eps = float(valid.sort_values("Silhouette", ascending=False).iloc[0]["eps"])
else:
    best_eps = float(grid_df.iloc[len(grid_df) // 2]["eps"])
print("Selected eps:", best_eps)
out["dbscan_eps"] = best_eps
db_lab = DBSCAN(eps=best_eps, min_samples=min_samples).fit_predict(X_scaled)
evaluate("DBSCAN", db_lab, f"eps={best_eps}, min_samples={min_samples}")

res = pd.DataFrame(results)
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(res.drop(columns=["Note"]).round(4).to_string(index=False))
res.round(4).to_csv("clustering_results.csv", index=False)
out["results"] = res.round(4).fillna("NA").to_dict(orient="records")

# 6. PCA VISUALISATION
pca = PCA(n_components=2, random_state=RS)
Z = pca.fit_transform(X_scaled)
ev = pca.explained_variance_ratio_
print("\nPCA explained variance (PC1, PC2):", ev.round(4).tolist(), "sum=", round(float(ev.sum()), 4))
out["pca_var"] = [round(float(v), 4) for v in ev]

show = ["K-Means", "Hierarchical (ward)", "Gaussian Mixture", "DBSCAN"]
fig, ax = plt.subplots(2, 2, figsize=(11, 9))
for a, name in zip(ax.ravel(), show):
    lab = labels_store[name]
    sc = a.scatter(Z[:, 0], Z[:, 1], c=lab, cmap="viridis", s=25, alpha=0.85)
    a.set_title(name); a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.grid(True)
plt.suptitle("Clusters in PCA Space"); plt.tight_layout()
plt.savefig("fig3_pca_clusters.png", dpi=150); plt.close()

# 7. SILHOUETTE PLOT FOR K-MEANS
from sklearn.metrics import silhouette_samples
sv = silhouette_samples(X_scaled, km.labels_)
plt.figure(figsize=(7, 5))
y0 = 0
for c in range(best_k):
    v = np.sort(sv[km.labels_ == c])
    plt.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, alpha=0.8, label=f"Cluster {c}")
    y0 += len(v) + 5
plt.axvline(silhouette_score(X_scaled, km.labels_), color="red", linestyle="--", label="Mean")
plt.xlabel("Silhouette coefficient"); plt.ylabel("Samples"); plt.title("K-Means Silhouette Plot")
plt.legend(); plt.tight_layout(); plt.savefig("fig4_silhouette_plot.png", dpi=150); plt.close()
out["kmeans_neg_sil"] = int((sv < 0).sum())
print("K-Means samples with negative silhouette:", out["kmeans_neg_sil"])

# 8. DENDROGRAM
Zl = linkage(X_scaled, method="ward")
plt.figure(figsize=(9, 5))
dendrogram(Zl, truncate_mode="lastp", p=20, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram (Ward, truncated)"); plt.ylabel("Distance")
plt.tight_layout(); plt.savefig("fig5_dendrogram.png", dpi=150); plt.close()

# 9. METRIC BAR CHART
main = res[res.Method.isin(show)]
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, col, ttl in zip(ax, ["Silhouette", "Davies_Bouldin", "Calinski_Harabasz"],
                       ["Silhouette (higher better)", "Davies-Bouldin (lower better)", "Calinski-Harabasz (higher better)"]):
    a.bar(main.Method, main[col]); a.set_title(ttl); a.tick_params(axis="x", rotation=25); a.grid(axis="y")
plt.tight_layout(); plt.savefig("fig6_metric_comparison.png", dpi=150); plt.close()

# 10. CLUSTER PROFILE (K-Means), original units
prof = X.copy(); prof["Cluster"] = km.labels_
profile = prof.groupby("Cluster").mean().round(2)
print("\nK-Means cluster profile (original units):")
print(profile.T.to_string())
out["cluster_sizes"] = np.bincount(km.labels_).tolist()
print("Cluster sizes:", out["cluster_sizes"])
profile.to_csv("kmeans_profile.csv")

# 11. STABILITY: K-Means with 20 different seeds
aris = []
base = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X_scaled).labels_
for sd in range(1, 21):
    l = KMeans(n_clusters=best_k, n_init=10, random_state=sd).fit(X_scaled).labels_
    aris.append(adjusted_rand_score(base, l))
out["stability_min_ari"] = round(float(min(aris)), 4)
print("\nK-Means seed stability (min ARI vs seed 0 over 20 seeds):", out["stability_min_ari"])

with open("results.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\nDone.")
