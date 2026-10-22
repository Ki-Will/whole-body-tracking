"""Small constant-velocity centre/size Kalman filter for box prediction."""
import cv2
import numpy as np


class BoxKalman:
    def __init__(self, box):
        self.filter = cv2.KalmanFilter(8, 4)
        self.filter.transitionMatrix = np.array([
            [1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1],
        ], dtype=np.float32)
        self.filter.measurementMatrix = np.array([
            [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0],
        ], dtype=np.float32)
        self.filter.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.filter.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.filter.errorCovPost = np.eye(8, dtype=np.float32)
        self.filter.statePost[:4, 0] = self._measurement(box)

    @staticmethod
    def _measurement(box):
        x1, y1, x2, y2 = map(float, box)
        return np.array([(x1+x2)/2, (y1+y2)/2, max(1, x2-x1), max(1, y2-y1)], dtype=np.float32)

    @staticmethod
    def _box(measurement):
        cx, cy, w, h = map(float, measurement)
        w, h = max(1.0, w), max(1.0, h)
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dtype=np.float32)

    def predict(self):
        return self._box(self.filter.predict()[:4, 0])

    def correct(self, box):
        measurement = self._measurement(box).reshape(4, 1)
        corrected = self.filter.correct(measurement)
        return self._box(corrected[:4, 0])
