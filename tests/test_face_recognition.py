import numpy as np

from whole_body_tracking.face_recognition import FaceRecognizer


def test_embedding_normalization_is_unit_length():
    vector = FaceRecognizer._normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_invalid_embedding_is_rejected():
    try:
        FaceRecognizer._normalize(np.zeros(4, dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("zero embedding should be rejected")
