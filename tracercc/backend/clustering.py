"""Clustering layer: PCA + sklearn HDBSCAN by default, UMAP+HDBSCAN if installed.

Defaults to the sklearn-only path to keep cold start fast (no numba/llvmlite).
Empirically ~80% of the full-path savings on the dev corpus.
"""

from __future__ import annotations

import numpy as np


def has_full_clusterer() -> bool:
    try:
        import umap  # noqa: F401
        import hdbscan  # noqa: F401
        return True
    except Exception:
        return False


def cluster_turns(
    emb: np.ndarray,
    min_cluster_size: int = 20,
    min_samples: int = 5,
) -> tuple[np.ndarray, str]:
    """Return (labels, backend_name). -1 means noise.

    Strategy:
      1. If umap+hdbscan installed, use the full path.
      2. Else PCA + sklearn HDBSCAN (great for >=200 turns).
      3. If HDBSCAN finds 0 clusters AND we have a small corpus,
         fall back to KMeans with silhouette-picked k. This is the
         realistic regime for a single-developer wrapped report.
    """
    if emb.shape[0] == 0:
        return np.array([], dtype=int), "noop"
    if has_full_clusterer():
        from .cluster_full import cluster_hdbscan, project
        X10 = project(emb, n_components=10, random_state=42, n_neighbors=15)
        return cluster_hdbscan(X10, min_cluster_size=min_cluster_size,
                               min_samples=min_samples), "full"

    from sklearn.cluster import HDBSCAN as SkHDBSCAN
    from sklearn.decomposition import PCA
    n_comp = max(2, min(30, emb.shape[0] - 1, emb.shape[1]))
    Xp = PCA(n_components=n_comp, random_state=42, whiten=True).fit_transform(emb)
    labels = SkHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        n_jobs=-1,
    ).fit_predict(Xp)
    n_found = len(set(labels.tolist()) - {-1})
    if n_found > 0:
        return labels, "lite"

    n = emb.shape[0]
    if n < 6:
        return np.zeros(n, dtype=int), "single"
    return _kmeans_silhouette(emb), "kmeans"


def _kmeans_silhouette(emb: np.ndarray) -> np.ndarray:
    """Pick k in [2, min(8, n//4)] by silhouette on cosine-normalised embeddings."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    X = normalize(emb, norm="l2", axis=1)
    n = X.shape[0]
    k_max = max(2, min(8, n // 4))
    best_k, best_score, best_labels = 2, -1.0, None
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        try:
            s = silhouette_score(X, km.labels_, metric="cosine", sample_size=min(n, 500))
        except Exception:
            continue
        if s > best_score:
            best_score, best_k, best_labels = s, k, km.labels_
    if best_labels is None:
        km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
        best_labels = km.labels_
    return best_labels


def medoid_and_examples(
    emb: np.ndarray,
    idx: np.ndarray,
    texts: list[str],
    k: int = 3,
) -> tuple[str, list[str]]:
    from sklearn.metrics.pairwise import cosine_distances
    if len(idx) == 0:
        return "", []
    sub = emb[idx]
    d = cosine_distances(sub)
    medoid = int(d.sum(axis=1).argmin())
    nn = d[medoid].argsort()[:k]
    return texts[int(idx[medoid])], [texts[int(idx[i])] for i in nn]
