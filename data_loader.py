"""
Data Loader Module
Loads single-cell gene expression data
"""
import numpy as np
import csv

def load_data(filepath: str) -> np.ndarray:
    """
    Load cell expression matrix from CSV.
    Each row is a cell, each column is a gene.
    """
    data = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)

def normalize(data: np.ndarray) -> np.ndarray:
    """L2-normalize each cell vector."""
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    return data / (norms + 1e-8)

def generate_mock_data(n_cells=500, n_genes=128, seed=42) -> np.ndarray:
    """Generate mock single-cell expression data for testing."""
    rng = np.random.default_rng(seed)
    return rng.random((n_cells, n_genes)).astype(np.float32)