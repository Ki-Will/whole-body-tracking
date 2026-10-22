"""OpenCV overlays for detections, tracks, states, and target centring."""
import time
import cv2
import numpy as np
from .centering import centering_offset


STATE_COLORS = {
    "CONFIRMED": (60, 200, 80),
    "TEMPORARY": (0, 190, 255),
    "UNKNOWN": (160, 160, 160),
}


class FPSMeter:
    def __init__(self):
        self.last = time.perf_counter()
        self.fps = 0.0

    def tick(self):
        now = time.perf_counter()
        dt = now - self.last
        self.last = now
        if dt > 0:
            instant = 1.0 / dt
            self.fps = instant if self.fps == 0 else 0.85 * self.fps + 0.15 * instant
        return self.fps


def draw_overlay(frame, tracks, associations, faces, fps: float, timestamp: float):
    output = frame.copy()
    height, width = output.shape[:2]
    cv2.drawMarker(output, (width // 2, height // 2), (255, 255, 255),
                   cv2.MARKER_CROSS, 18, 1)
    face_by_body = associations
    for face in faces:
        x1, y1, x2, y2 = map(int, face)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 180, 40), 1)
    for index, track in enumerate(tracks):
        x1, y1, x2, y2 = map(int, track.box)
        state = track.identity.state.value
        color = STATE_COLORS[state]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        confidence = track.identity.confidence(timestamp)
        label = f"ID {track.track_id} | {state} | C={confidence:.2f}"
        cv2.putText(output, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        dx, dy = centering_offset(track.box, width, height)
        cv2.putText(output, f"centre offset x={dx:+.2f}, y={dy:+.2f}",
                    (x1, min(height - 6, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.putText(output, f"CPU | {fps:.1f} FPS | q/Esc: quit", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 2, cv2.LINE_AA)
    return output
