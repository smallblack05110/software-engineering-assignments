# Single-Cell ANN Retrieval System

A high-dimensional Approximate Nearest Neighbor (ANN) retrieval system
designed for single-cell RNA sequencing data.

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point |
| `ann_index.py` | ANN index construction and search |
| `data_loader.py` | Data loading and mock data generation |
| `preprocess.py` | Normalization and dimensionality reduction |
| `evaluate.py` | Recall@K and latency metrics |
| `config.ini` | System configuration |
| `requirements.txt` | Python dependencies |

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Methods Supported

- **HNSW** (Hierarchical Navigable Small World)
- **KD-Tree**