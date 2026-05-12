"""
Evaluation Module
Metrics for ANN retrieval quality
"""
import numpy as np

def recall_at_k(true_neighbors, approx_neighbors, k):
    """
    Compute Recall@K: fraction of true top-k neighbors
    found in approximate top-k results.
    """
    true_set = set(true_neighbors[:k])
    approx_set = set(approx_neighbors[:k])
    return len(true_set & approx_set) / k

def mean_recall(true_list, approx_list, k):
    """Average Recall@K over all queries."""
    scores = [
        recall_at_k(t, a, k)
        for t, a in zip(true_list, approx_list)
    ]
    return float(np.mean(scores))

def query_time_stats(times: list):
    """Return mean and std of query latency (ms)."""
    arr = np.array(times)
    return {"mean_ms": float(arr.mean()), "std_ms": float(arr.std())}