"""YuNet landmark extraction helper used by enrollment and recognition."""
import cv2
import numpy as np


def detect_faces_with_landmarks(detector, frame):
    height, width = frame.shape[:2]
    detector.detector.setInputSize((width, height))
    _, rows = detector.detector.detect(frame)
    if rows is None:
        return []
    return [{
        "box": np.asarray([r[0], r[1], r[0] + r[2], r[1] + r[3]], dtype=np.float32),
        "landmarks": np.asarray(r[4:14], dtype=np.float32).reshape(5, 2),
        "score": float(r[14]),
    } for r in rows]
