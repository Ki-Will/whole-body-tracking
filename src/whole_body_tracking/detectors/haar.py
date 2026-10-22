"""Optional Haar Cascade face-detector baseline bundled with OpenCV."""
import cv2
import numpy as np


class HaarFaceDetector:
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load OpenCV Haar cascade: {path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors
        )
        return [
            np.array([x, y, x + w, y + h], dtype=np.float32)
            for x, y, w, h in faces
        ]
