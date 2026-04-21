"""Optional UMAP + HDBSCAN clustering path.

Imported lazily by ``clustering.py`` only when both ``umap`` and ``hdbscan`` are
installed. Kept in its own module so the lite path runs without numba/llvmlite
cold-start cost.
"""

import numpy as np


def project(emb: np.ndarray, n_components: int = 10, random_state: int = 42,
            n_neighbors: int = 15) -> np.ndarray:
    import umap
    n_neighbors = max(2, min(n_neighbors, max(2, emb.shape[0] - 1)))
    n_components = max(2, min(n_components, emb.shape[0] - 1, emb.shape[1]))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=random_state,
        low_memory=True,
    )
    return reducer.fit_transform(emb)


def cluster_hdbscan(X: np.ndarray, min_cluster_size: int = 20,
                    min_samples: int = 5) -> np.ndarray:
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, min_cluster_size),
        min_samples=max(1, min_samples),
        metric="euclidean",
        core_dist_n_jobs=-1,
        prediction_data=False,
    )
    return clusterer.fit_predict(X)
