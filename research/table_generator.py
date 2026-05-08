import time
import numpy as np
from app.core.faiss_index import FaissIndex


def measure_index(index_type, vectors, query):
    index = FaissIndex(index_type=index_type)

    for v in vectors:
        index.add(v, {})

    start = time.time()
    D, I = index.search(query, 5)
    end = time.time()

    return (end - start) * 1000


def run():
    sizes = [100, 500, 1000]

    for size in sizes:
        print(f"\nDataset Size: {size}")

        vectors = np.random.rand(size, 512).astype("float32")
        query = vectors[0]

        for idx_type in ["flat", "ivf", "hnsw"]:
            t = measure_index(idx_type, vectors, query)
            print(f"{idx_type}: {t:.2f} ms")


if __name__ == "__main__":
    run()