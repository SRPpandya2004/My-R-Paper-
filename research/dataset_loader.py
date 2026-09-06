import os
import cv2
from app.core.model import InsightFaceModel

class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = InsightFaceModel()

    def load(self):
        embeddings = []
        labels = []
        paths = []

        for person in os.listdir(self.dataset_path):
            person_dir = os.path.join(self.dataset_path, person)

            if not os.path.isdir(person_dir):
                continue

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)

                img = cv2.imread(img_path)
                if img is None:
                    continue

                faces = self.model.extract_faces(img)
                if not faces:
                    continue

                emb = faces[0]["embedding"]

                embeddings.append(emb)
                labels.append(person)
                paths.append(img_path)

        return embeddings, labels, paths
