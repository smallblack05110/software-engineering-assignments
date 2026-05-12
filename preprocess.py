"""
Preprocessing Module
Dimensionality reduction and feature selection
"""
import numpy as np

def pca_reduce(data: np.ndarray, n_components: int = 70) -> np.ndarray:
    """
    Reduce dimensionality via PCA.
    Used before building ANN index to speed up retrieval.
    """
    mean = data.mean(axis=0)
    centered = data - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    return centered @ components.T

def filter_low_variance(data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Remove genes with variance below threshold."""
    variances = data.var(axis=0)
    mask = variances >= threshold
    return data[:, mask]

def log_normalize(data: np.ndarray) -> np.ndarray:
    """Log1p normalization commonly used in scRNA-seq pipelines."""
    return np.log1p(data)