"""
Tab 7: clustering, shared by both shopper-segment and brand-segment views
(same methodology, different feature sets -- see data_utils.py's
encode_shopper_clustering_features / encode_brand_clustering_features for
which columns feed each, and why demographics/categoricals are excluded).

K-Means requires scaled, Euclidean-meaningful continuous features -- mixing
in one-hot categoricals would distort distances, so both feature sets here
are continuous-only by construction.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def elbow_and_silhouette(X: pd.DataFrame, k_min: int = 2, k_max: int = 15) -> pd.DataFrame:
    """Elbow (inertia) and silhouette across a range of k -- used together
    since elbow alone is often ambiguous; the dashboard shows where they
    agree/disagree on the 'best' k rather than picking one from the elbow's
    visual kink alone."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 else np.nan
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def run_clustering(X: pd.DataFrame, k: int) -> dict:
    """K-Means + PCA(2D) + hierarchical-clustering validation on the same
    feature set and same k, all in one call so the UI can show them together."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    km_labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_

    # Hierarchical clustering as a second, independent validation method --
    # K-Means assumes spherical clusters; if Ward-linkage agglomerative
    # clustering broadly agrees at the same k, that's evidence the segments
    # are real rather than an artifact of K-Means' own assumptions.
    Z = linkage(X_scaled, method="ward")
    hier_labels = fcluster(Z, t=k, criterion="maxclust")
    agreement = adjusted_rand_score(km_labels, hier_labels)

    return {
        "kmeans_labels": km_labels,
        "pca_coords": coords,
        "explained_variance": explained_variance,
        "linkage_matrix": Z,
        "hier_labels": hier_labels,
        "agreement_ari": float(agreement),
        "silhouette": float(silhouette_score(X_scaled, km_labels)),
        "X_scaled": X_scaled,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_utils import clean_ecommerce, encode_shopper_clustering_features, load_brand_survey, encode_brand_clustering_features

    df, _ = clean_ecommerce()
    Xs = encode_shopper_clustering_features(df)
    es = elbow_and_silhouette(Xs)
    print("Shopper elbow/silhouette:")
    print(es.to_string(index=False))
    best_k = int(es.loc[es["silhouette"].idxmax(), "k"])
    print(f"Best k by silhouette: {best_k}")
    r = run_clustering(Xs, best_k)
    print(f"Silhouette at chosen k: {r['silhouette']:.3f}, PCA explained variance: {r['explained_variance']}")
    print(f"K-Means vs hierarchical agreement (Adjusted Rand Index): {r['agreement_ari']:.3f}")

    print("\nBrand segments:")
    df_b = load_brand_survey()
    Xb = encode_brand_clustering_features(df_b)
    eb = elbow_and_silhouette(Xb)
    best_kb = int(eb.loc[eb["silhouette"].idxmax(), "k"])
    print(f"Best k by silhouette: {best_kb}")
    rb = run_clustering(Xb, best_kb)
    print(f"Silhouette: {rb['silhouette']:.3f}, agreement ARI: {rb['agreement_ari']:.3f}")
    print("\nAll clustering checks ran without error.")
