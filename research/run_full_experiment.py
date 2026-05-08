
import os
import numpy as np
from collections import defaultdict
import random

from app.research.dataset_loader import DatasetLoader
from app.research.faiss_multi import MultiFaiss
from app.research.adaptive import AdaptiveThreshold

DATASET = r"C:\Users\ACER\Desktop\R Paper\All Crickter\images"

CACHE_DIR = "research_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

EMB_FILE = os.path.join(CACHE_DIR, "emb.npy")
LBL_FILE = os.path.join(CACHE_DIR, "lbl.npy")
PTH_FILE = os.path.join(CACHE_DIR, "pth.npy")

# -----------------------------
# LOAD / EXTRACT EMBEDDINGS
# -----------------------------
if os.path.exists(EMB_FILE):
    print("⚡ Loading cached embeddings...")
    embeddings = np.load(EMB_FILE, allow_pickle=True)
    labels = np.load(LBL_FILE, allow_pickle=True)
    paths = np.load(PTH_FILE, allow_pickle=True)
else:
    print("🔄 Extracting embeddings...")
    loader = DatasetLoader(DATASET)
    embeddings, labels, paths = loader.load()

    np.save(EMB_FILE, embeddings)
    np.save(LBL_FILE, labels)
    np.save(PTH_FILE, paths)

    print("✅ Saved embeddings")

# -----------------------------
# SPLIT DATASET (PERSON-WISE)
# -----------------------------
def split_dataset(embeddings, labels, paths, test_ratio=0.3):
    data = defaultdict(list)

    for emb, label, path in zip(embeddings, labels, paths):
        data[label].append((emb, path))

    train_emb, train_lbl = [], []
    test_emb, test_lbl = [], []

    for label, items in data.items():
        if len(items) < 2:
            continue

        random.shuffle(items)
        split_idx = int(len(items) * (1 - test_ratio))

        for i, (emb, _) in enumerate(items):
            if i < split_idx:
                train_emb.append(emb)
                train_lbl.append(label)
            else:
                test_emb.append(emb)
                test_lbl.append(label)

    return train_emb, train_lbl, test_emb, test_lbl


train_emb, train_lbl, test_emb, test_lbl = split_dataset(
    embeddings, labels, paths
)

print(f"Train size: {len(train_emb)} | Test size: {len(test_emb)}")

# -----------------------------
# BUILD FAISS
# -----------------------------
faiss_multi = MultiFaiss()
faiss_multi.build_indices(train_emb)

def compute_global_threshold(train_emb):
    import numpy as np

    # sample random pairs
    sample_size = min(500, len(train_emb))
    idx = np.random.choice(len(train_emb), sample_size, replace=False)

    distances = []
    for i in range(sample_size - 1):
        d = np.linalg.norm(train_emb[idx[i]] - train_emb[idx[i+1]])
        distances.append(d)

    return np.percentile(distances, 40)  # stable global boundary


GLOBAL_TH = compute_global_threshold(train_emb)
print("Global Threshold:", GLOBAL_TH)

# -----------------------------
# EVALUATION
# -----------------------------
top1 = 0
top5 = 0
total = len(test_emb)

all_times = {"flat": [], "ivf": [], "hnsw": []}

# FAR / FRR counters
tp_fixed = fp_fixed = fn_fixed = tn_fixed = 0
tp_adapt = fp_adapt = fn_adapt = tn_adapt = 0

FIXED_THRESHOLD = 1.2

for i in range(total):
    query = test_emb[i]
    true_label = test_lbl[i]

    results = faiss_multi.search(query, top_k=50)

    for method in results:
        all_times[method].append(results[method]["time_ms"])

    distances = results["flat"]["distances"]
    indices = results["flat"]["indices"]

    pred_labels = [train_lbl[idx] for idx in indices]

    # -------------------------
    # Accuracy
    # -------------------------
    if true_label == pred_labels[0]:
        top1 += 1

    if true_label in pred_labels:
        top5 += 1

    # -------------------------
    # Adaptive Threshold
    # -------------------------
    # adaptive_th, _ = AdaptiveThreshold.compute(distances)
    adaptive_th, _ = AdaptiveThreshold.compute(distances, GLOBAL_TH)

    # -------------------------
    # FAR / FRR
    # -------------------------
    for d, pred in zip(distances, pred_labels):

        # FIXED
        if d <= FIXED_THRESHOLD:
            if pred == true_label:
                tp_fixed += 1
            else:
                fp_fixed += 1
        else:
            if pred == true_label:
                fn_fixed += 1
            else:
                tn_fixed += 1

        # ADAPTIVE
        if d <= adaptive_th:
            if pred == true_label:
                tp_adapt += 1
            else:
                fp_adapt += 1
        else:
            if pred == true_label:
                fn_adapt += 1
            else:
                tn_adapt += 1


# -----------------------------
# METRICS
# -----------------------------
def safe_div(a, b):
    return a / b if b != 0 else 0

top1_acc = top1 / total if total > 0 else 0
top5_acc = top5 / total if total > 0 else 0

avg_times = {k: float(np.mean(v)) for k, v in all_times.items()}

# FAR / FRR
far_fixed = safe_div(fp_fixed, (fp_fixed + tn_fixed))
frr_fixed = safe_div(fn_fixed, (fn_fixed + tp_fixed))

far_adapt = safe_div(fp_adapt, (fp_adapt + tn_adapt))
frr_adapt = safe_div(fn_adapt, (fn_adapt + tp_adapt))

# -----------------------------
# FINAL OUTPUT
# -----------------------------
print("\n📊 FINAL RESULTS (FULL)")

print({
    "train_size": len(train_emb),
    "test_size": len(test_emb),

    "top1_accuracy": top1_acc,
    "top5_accuracy": top5_acc,

    "avg_time_ms": avg_times,

    "threshold_comparison": {
        "fixed": {
            "FAR": far_fixed,
            "FRR": frr_fixed
        },
        "adaptive": {
            "FAR": far_adapt,
            "FRR": frr_adapt
        }
    }
})


print("\n================ FINAL EXPERIMENT RESULTS ================\n")

print(f"Total_images of Different 70 Pesons : {len(train_emb) + len(test_emb)}")

print(f"Train Size              : {len(train_emb)}")
print(f"Test Size               : {len(test_emb)}\n")

print("---- Retrieval Accuracy ----")
print(f"Top-1 Accuracy          : {top1_acc * 100:.2f}%")
print(f"Top-5 Accuracy          : {top5_acc * 100:.2f}%\n")

print("---- Average Retrieval Time ----")
print(f"Flat Index              : {avg_times['flat']:.3f} ms")
print(f"IVF Index               : {avg_times['ivf']:.3f} ms")
print(f"HNSW Index              : {avg_times['hnsw']:.3f} ms\n")

print("---- Threshold Analysis ----")

print("Fixed Threshold:")
print(f"  False Accept Rate     : {far_fixed:.4f}")
print(f"  False Reject Rate     : {frr_fixed:.4f}")

print("\nAdaptive Threshold:")
print(f"  False Accept Rate     : {far_adapt:.4f}")
print(f"  False Reject Rate     : {frr_adapt:.4f}")

print("\n==========================================================\n")
# loader = DatasetLoader(DATASET)
# embeddings, labels, paths = loader.load()

# print("Total images:", len(embeddings))

# faiss = MultiFaiss()
# faiss.build(embeddings, labels)

# adaptive = AdaptiveThreshold()

# # --------------------
# # METRICS
# # --------------------
# top1, top5 = 0, 0
# correct_fixed, correct_adaptive = 0, 0
# far_fixed, frr_fixed = 0, 0
# far_adaptive, frr_adaptive = 0, 0

# time_total = {"flat":0, "ivf":0, "hnsw":0}

# # --------------------
# # LOOP
# # --------------------
# for i in range(len(embeddings)):

#     query_emb = embeddings[i]
#     true_label = labels[i]
#     img = cv2.imread(paths[i])

#     results = faiss.search(query_emb, k=5)

#     distances = [m["distance"] for m in results["flat"]["matches"]]
#     adaptive_threshold, quality = adaptive.compute(distances, img)

#     # Top-K
#     matches_flat = results["flat"]["matches"]

#     if matches_flat and matches_flat[0]["label"] == true_label:
#         top1 += 1

#     if any(m["label"] == true_label for m in matches_flat):
#         top5 += 1

#     for name, data in results.items():
#         time_total[name] += data["time_ms"]

#         matches = data["matches"]

#         # FIXED
#         accepted_fixed = [m for m in matches if m["distance"] <= FIXED_THRESHOLD]

#         if any(m["label"] == true_label for m in accepted_fixed):
#             correct_fixed += 1
#         else:
#             frr_fixed += 1

#         if any(m["label"] != true_label for m in accepted_fixed):
#             far_fixed += 1

#         # ADAPTIVE
#         accepted_adaptive = [m for m in matches if m["distance"] <= adaptive_threshold]

#         if any(m["label"] == true_label for m in accepted_adaptive):
#             correct_adaptive += 1
#         else:
#             frr_adaptive += 1

#         if any(m["label"] != true_label for m in accepted_adaptive):
#             far_adaptive += 1

# # --------------------
# # FINAL RESULTS
# # --------------------
# N = len(embeddings)

# output = {
#     "dataset_size": N,
#     "top1_accuracy": top1 / N,
#     "top5_accuracy": top5 / N,
#     "fixed_threshold": {
#         "accuracy": correct_fixed / N,
#         "FAR": far_fixed / N,
#         "FRR": frr_fixed / N
#     },
#     "adaptive_threshold": {
#         "accuracy": correct_adaptive / N,
#         "FAR": far_adaptive / N,
#         "FRR": frr_adaptive / N
#     },
#     "avg_time_ms": {k: v/N for k,v in time_total.items()}
# }

# print(json.dumps(output, indent=2))