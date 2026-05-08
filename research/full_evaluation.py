import cv2
import numpy as np
import time
from app.core.model import InsightFaceModel
from app.core.faiss_index import FaissIndex


# -----------------------------
# Image Quality
# -----------------------------
def analyze_image_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness > 120 and blur > 100:
        quality = "high"
    elif brightness > 80:
        quality = "medium"
    else:
        quality = "low"

    return brightness, blur, quality


def get_threshold(quality):
    if quality == "high":
        return 1.0
    elif quality == "medium":
        return 1.2
    else:
        return 1.4


# -----------------------------
# Load model
# -----------------------------
model = InsightFaceModel()


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def evaluate(query_image_path):

    print("\n===== FULL EVALUATION REPORT =====\n")

    img = cv2.imread(query_image_path)
    if img is None:
        print("Invalid image path")
        return

    # -----------------------------
    # Face extraction
    # -----------------------------
    faces = model.extract_faces(img)
    if not faces:
        print("No face detected")
        return

    query_emb = faces[0]["embedding"]

    # -----------------------------
    # Quality analysis
    # -----------------------------
    brightness, blur, quality = analyze_image_quality(img)
    threshold = get_threshold(quality)

    print(f"Image Quality Analysis:")
    print(f"  Brightness: {brightness:.2f}")
    print(f"  Blur: {blur:.2f}")
    print(f"  Quality: {quality}")
    print(f"  Adaptive Threshold: {threshold}")

    # -----------------------------
    # Fake dataset (replace with real embeddings if needed)
    # -----------------------------
    dataset_size = 1000
    vectors = np.random.rand(dataset_size, 512).astype("float32")

    # -----------------------------
    # FAISS comparison
    # -----------------------------
    for index_type in ["flat", "ivf", "hnsw"]:

        print(f"\n--- Testing {index_type.upper()} Index ---")

        index = FaissIndex(index_type=index_type)

        for v in vectors:
            index.add(v, {"id": "sample"})

        start = time.time()
        distances, indices = index.search(query_emb, 5)
        end = time.time()

        search_time = (end - start) * 1000

        print(f"Search Time: {search_time:.2f} ms")

        print("Top Matches:")
        for d, i in zip(distances, indices):
            status = "MATCH" if d <= threshold else "REJECTED"
            print(f"  Index: {i}, Distance: {d:.4f}, {status}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    evaluate("test.jpg")  # 🔥 replace with your image