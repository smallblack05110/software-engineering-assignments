"""
Single-cell ANN Retrieval System
Entry point
"""
from ann_index import ANNIndex
from data_loader import load_data

# def main():
#     data = load_data("data/cells.csv")
#     index = ANNIndex(method="hnsw", dim=data.shape[1])
#     index.build(data)

#     query = data[0]
#     neighbors = index.search(query, k=20)
#     print("Top-10 nearest neighbors:", neighbors)

if __name__ == "__main__":
    main()