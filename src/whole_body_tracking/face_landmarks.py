"""YuNet landmark extraction helper used by enrollment and recognition."""
import numpy as np


def detect_faces_with_landmarks(detector, frame):
    height, width = frame.shape[:2]
    detector.detector.setInputSize((width, height))
    _, rows = detector.detector.detect(frame)
    if rows is None:
        return []
    results = []
    for row in rows:
        # YuNet order: right eye, left eye, nose, right mouth, left mouth.
        # ArcFace's canonical template expects left eye, right eye, nose,
        # left mouth, right mouth. Reorder before alignment.
        points = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
        points = points[[1, 0, 2, 4, 3]]
        results.append({
            "box": np.asarray([row[0], row[1], row[0] + row[2], row[1] + row[3]], dtype=np.float32),
            "landmarks": points,
            "score": float(row[14]),
        })
    return results
