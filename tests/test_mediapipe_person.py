import numpy as np

from whole_body_tracking.detectors.mediapipe_person import MediaPipePersonDetector


def test_mediapipe_anchor_count_and_bounds():
    anchors = MediaPipePersonDetector._generate_anchors()
    assert anchors.shape == (2254, 2)
    assert np.all(anchors > 0.0)
    assert np.all(anchors < 1.0)


def test_mediapipe_sigmoid_converts_logits_to_probabilities():
    probabilities = MediaPipePersonDetector._sigmoid(
        np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
    )
    assert probabilities[0] >= 0.0
    assert probabilities[0] < 1e-6
    assert probabilities[1] == 0.5
    assert probabilities[2] > 0.999
