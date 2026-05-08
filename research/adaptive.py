import numpy as np
import cv2

class AdaptiveThreshold:

    @staticmethod
    def image_quality(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()

        quality = "high" if blur > 1000 else "medium" if blur > 500 else "low"

        return brightness, blur, quality

    @staticmethod
    def compute(distances, global_th, img=None):
        """
        FINAL HYBRID ADAPTIVE THRESHOLD
        ✔ Global stability
        ✔ Local adaptability
        ✔ Optional quality adjustment
        """

        if len(distances) == 0:
            return global_th, {}

        # -----------------------------
        # LOCAL THRESHOLD (robust)
        # -----------------------------
        local_th = np.percentile(distances, 30)

        # -----------------------------
        # HYBRID COMBINATION
        # -----------------------------
        alpha = 0.7   # weight for global (important)
        threshold = alpha * global_th + (1 - alpha) * local_th

        quality_info = {}

        # -----------------------------
        # OPTIONAL IMAGE QUALITY
        # -----------------------------
        if img is not None:
            brightness, blur, quality = AdaptiveThreshold.image_quality(img)

            if quality == "low":
                threshold += 0.03
            elif quality == "high":
                threshold -= 0.02

            quality_info = {
                "brightness": float(brightness),
                "blur": float(blur),
                "quality": quality
            }

        return float(threshold), quality_info