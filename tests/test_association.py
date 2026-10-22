import numpy as np
from whole_body_tracking.association import associate_faces_to_bodies


def test_face_inside_upper_body_is_associated():
    face = np.array([40, 10, 60, 35], dtype=np.float32)
    body = np.array([20, 5, 80, 150], dtype=np.float32)
    assert associate_faces_to_bodies([face], [body]) == {0: 0}


def test_face_outside_body_is_not_associated():
    face = np.array([110, 10, 130, 35], dtype=np.float32)
    body = np.array([20, 5, 80, 150], dtype=np.float32)
    assert associate_faces_to_bodies([face], [body]) == {}
