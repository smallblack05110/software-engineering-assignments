"""
ANN Index Module
Supports HNSW and KD-Tree methods
"""
import numpy as np

class ANNIndex:
    def __init__(self, method="hnsw", dim=128):
        self.method = method
        self.dim = dim
        self.index = None

    def build(self, data: np.ndarray):
        """Build the index from cell expression vectors."""
        if self.method == "hnsw":
            self._build_hnsw(data)
        elif self.method == "kdtree":
            self._build_kdtree(data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _build_hnsw(self, data):
        # Placeholder for HNSW construction
        self.index = data

    def _build_kdtree(self, data):
        from scipy.spatial import KDTree
        self.index = KDTree(data)

    def search(self, query: np.ndarray, k: int = 10):
        """Return indices of k approximate nearest neighbors."""
        if self.method == "kdtree":
            _, indices = self.index.query(query, k=k)
            return indices
        # HNSW: brute-force placeholder
        dists = np.linalg.norm(self.index - query, axis=1)
        return np.argsort(dists)[:k]