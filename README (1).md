# Practical 6: Clustering Algorithms

TY B.Tech. AIML Practical Assignment

| | |
|---|---|
| Student | Atharv Jeevan Fatale |
| PRN | 202401070076 |
| Branch / Division / Batch | ENTC / A / A2 |
| Faculty | Mrs. Anita Shingade |

## Aim

Implement clustering algorithms on a real-world dataset to identify meaningful groups, and evaluate the results using Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index and visual inspection.

## Dataset

UCI Wine dataset, loaded with `sklearn.datasets.load_wine`.

- 178 samples, 13 numerical chemical features, no missing values
- The class label (3 cultivars: 59, 71, 48 wines) is **not** used for clustering. It is used only for external validation with the Adjusted Rand Index (ARI).

## Methods

1. Standardize all features (`StandardScaler`)
2. Select k (2 to 10) using the Elbow method, Silhouette Score and Davies-Bouldin Index
3. K-Means, Hierarchical (Ward, complete, average, single), Gaussian Mixture and DBSCAN
4. Evaluate with Silhouette, Davies-Bouldin, Calinski-Harabasz and ARI
5. Visualize with PCA, a silhouette plot and a dendrogram
6. Profile the K-Means clusters and test stability over 20 random seeds

## How to run

```bash
pip install numpy pandas matplotlib scikit-learn scipy
python clustering_assignment.py
```

The script needs no input files. It runs in a few seconds and writes the outputs listed below to the current folder.

## Outputs

| File | Content |
|---|---|
| `fig1_k_selection.png` | Elbow, Silhouette and Davies-Bouldin vs k |
| `fig2_kdistance.png` | k-distance graph for DBSCAN eps |
| `fig3_pca_clusters.png` | Clusters of the four methods in PCA space |
| `fig4_silhouette_plot.png` | K-Means silhouette plot |
| `fig5_dendrogram.png` | Ward dendrogram |
| `fig6_metric_comparison.png` | Metric comparison bar charts |
| `clustering_results.csv` | Final comparison table |
| `kmeans_profile.csv` | K-Means cluster means (original units) |
| `results.json` | All key numbers |

## Results (random_state = 42)

| Method | Clusters | Noise | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI |
|---|---|---|---|---|---|---|
| **K-Means** | 3 | 0 | 0.2849 | 1.3892 | 70.9400 | 0.8975 |
| Hierarchical (Ward) | 3 | 0 | 0.2774 | 1.4186 | 67.6475 | 0.7899 |
| Hierarchical (Complete) | 3 | 0 | 0.2038 | 1.8961 | 48.9898 | 0.5771 |
| Hierarchical (Average) | 3 | 0 | 0.1575 | 1.0299 | 4.0314 | -0.0054 |
| Hierarchical (Single) | 3 | 0 | 0.1827 | 0.9105 | 4.0617 | -0.0068 |
| Gaussian Mixture | 3 | 0 | 0.2844 | 1.3938 | 70.6878 | 0.8804 |
| DBSCAN (eps 2.25, min_samples 5) | 2 | 46 | 0.3371* | 1.1712* | 71.0774* | 0.3962 |

\* Computed on the 132 non-noise points only, so not directly comparable with the other rows.

## Key findings

- k = 3 is supported by the maximum Silhouette, minimum Davies-Bouldin and maximum Calinski-Harabasz among k = 2 to 10.
- A Silhouette of about 0.28 means the clusters overlap (moderate-to-weak separation), yet ARI = 0.8975 shows they match the true wine classes well.
- K-Means is stable: the minimum ARI over 20 random seeds is 0.982.
- DBSCAN's better-looking metrics are misleading because 25.8% of the points are discarded as noise.
- Average and single linkage have the lowest Davies-Bouldin but put 174 of 178 points in one cluster, so one metric alone is not enough.

## Tested with

scikit-learn 1.9.1, SciPy 1.18.1, pandas 3.0.5, NumPy 2.5.3, Matplotlib 3.11.2. Other versions may change results in the last decimals.

## Files

- `clustering_assignment.py`: full implementation
- `Practical_6_Clustering_Algorithms.docx`: practical report
- `README.md`: this file
