"""OpenCV YuNet face detector wrapper."""
from pathlib import Path
import cv2
import numpy as np


class YuNetFaceDetector:
    def __init__(self, model_path: str | Path, confidence: float = 0.80):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YuNet model not found: {self.model_path}. Download the ONNX file "
                "from the OpenCV Zoo YuNet model directory or pass --no-face."
            )
        self.confidence = float(confidence)
        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path), "", (320, 320), self.confidence, 0.3, 5000
        )

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        results = []
        for face in faces:
            x, y, w, h = map(float, face[:4])
            results.append(np.array([x, y, x + w, y + h], dtype=np.float32))
        return results
