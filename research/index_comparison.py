#old file

# import time
# import numpy as np
# from app.core.faiss_index import FaissIndex


# def run():
#     vectors = np.random.rand(1000, 512).astype("float32")
#     query = vectors[0]

#     for idx_type in ["flat", "ivf", "hnsw"]:
#         index = FaissIndex(index_type=idx_type)

#         for v in vectors:
#             index.add(v, {})

#         start = time.time()
#         index.search(query, 10)
#         end = time.time()

#         print(f"{idx_type} → {(end-start)*1000:.2f} ms")


# if __name__ == "__main__":
#     run()
