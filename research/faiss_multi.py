import faiss
import numpy as np
import time


class MultiFaiss:
    def __init__(self, dim=512):
        self.dim = dim

    def build_indices(self, embeddings):
        xb = np.array(embeddings).astype("float32")

        # -------------------------
        # FLAT (Exact Search)
        # -------------------------
        self.flat = faiss.IndexFlatL2(self.dim)
        self.flat.add(xb)

        # -------------------------
        # IVF (Optimized)
        # -------------------------
        nlist = int(np.sqrt(len(xb)))  # dynamic cluster size

        quantizer = faiss.IndexFlatL2(self.dim)
        self.ivf = faiss.IndexIVFFlat(quantizer, self.dim, nlist)

        self.ivf.train(xb)
        self.ivf.add(xb)

        # 🔥 TUNING (important for speed vs accuracy)
        self.ivf.nprobe = max(5, nlist // 10)

        # -------------------------
        # HNSW (Optimized)
        # -------------------------
        self.hnsw = faiss.IndexHNSWFlat(self.dim, 32)

        # 🔥 BUILD QUALITY (must be before add)
        # self.hnsw.hnsw.efConstruction = 40

        # self.hnsw.add(xb)
        # self.hnsw = faiss.IndexHNSWFlat(self.dim, 64)

        self.hnsw.hnsw.efConstruction = 80
        self.hnsw.add(xb)

        self.hnsw.hnsw.efSearch = 80

        # 🔥 SEARCH QUALITY (set after add or before search)
        self.hnsw.hnsw.efSearch = 64

        self.db = xb

    def search(self, query, top_k=5):
        q = np.array([query]).astype("float32")

        results = {}

        for name, index in {
            "flat": self.flat,
            "ivf": self.ivf,
            "hnsw": self.hnsw
        }.items():

            start = time.perf_counter()  # 🔥 more accurate timing
            D, I = index.search(q, top_k)
            end = time.perf_counter()

            results[name] = {
                "distances": D[0].tolist(),
                "indices": I[0].tolist(),
                "time_ms": round((end - start) * 1000, 4)
            }

        return results



# import faiss
# import numpy as np
# import time

# class MultiFaiss:
#     def __init__(self, dim=512):
#         self.dim = dim

#     def build_indices(self, embeddings):
#         xb = np.array(embeddings).astype("float32")

#         # Flat
#         self.flat = faiss.IndexFlatL2(self.dim)
#         self.flat.add(xb)

#         # IVF
#         nlist = 100
#         quantizer = faiss.IndexFlatL2(self.dim)
#         self.ivf = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
#         self.ivf.train(xb)
#         self.ivf.add(xb)

#         # HNSW
#         self.hnsw = faiss.IndexHNSWFlat(self.dim, 32)
#         self.hnsw.add(xb)

#         self.db = xb

#     def search(self, query, top_k=5):
#         q = np.array([query]).astype("float32")

#         results = {}

#         for name, index in {
#             "flat": self.flat,
#             "ivf": self.ivf,
#             "hnsw": self.hnsw
#         }.items():

#             start = time.time()
#             D, I = index.search(q, top_k)
#             end = time.time()

#             results[name] = {
#                 "distances": D[0],
#                 "indices": I[0],
#                 "time_ms": (end - start) * 1000
#             }

#         return results