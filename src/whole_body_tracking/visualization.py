"""OpenCV overlays for detections, tracks, body visibility, states, and centring."""
import time
import cv2
import numpy as np
from .centering import centering_offset


STATE_COLORS = {
    "CONFIRMED": (60, 200, 80),
    "TEMPORARY": (0, 190, 255),
    "UNKNOWN": (160, 160, 160),
}

BODY_COLORS = {
    "FULL BODY": (60, 200, 80),
    "PARTIAL BODY": (0, 190, 255),
    "UNKNOWN": (160, 160, 160),
}


def body_visibility(box, frame_width: int, frame_height: int, missed_frames: int = 0):
    """Estimate whether a tracked person box is fully visible in the image.

    This is a geometric estimate, not proof that the detector found every body part:
    a box touching the image border is treated as partial, and a predicted-only track
    is unknown because the person was not detected in the current frame.
    """
    if missed_frames > 0:
        return "UNKNOWN"
    x1, y1, x2, y2 = map(float, box)
    margin = max(8.0, 0.015 * min(frame_width, frame_height))
    if (
        x1 <= margin
        or y1 <= margin
        or x2 >= frame_width - margin
        or y2 >= frame_height - margin
    ):
        return "PARTIAL BODY"
    return "FULL BODY"


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
    cv2.drawMarker(
        output, (width // 2, height // 2), (255, 255, 255),
        cv2.MARKER_CROSS, 18, 1
    )

    for face in faces:
        x1, y1, x2, y2 = map(int, face)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 180, 40), 1)

    visibility_states = []
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.box)
        visibility = body_visibility(
            track.box, width, height, getattr(track, "missed_frames", 0)
        )
        visibility_states.append(visibility)
        identity_state = track.identity.state.value
        color = BODY_COLORS[visibility]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        confidence = track.identity.confidence(timestamp)
        label = (
            f"ID {track.track_id} | {visibility} | "
            f"IDENTITY {identity_state} | C={confidence:.2f}"
        )
        cv2.putText(
            output, label, (max(4, x1), max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA
        )
        dx, dy = centering_offset(track.box, width, height)
        cv2.putText(
            output, f"centre offset x={dx:+.2f}, y={dy:+.2f}",
            (max(4, x1), min(height - 6, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA
        )

    # Show a persistent summary even when no person is detected.
    if not tracks:
        summary = "BODY STATUS: UNKNOWN - NO PERSON DETECTED"
        summary_color = BODY_COLORS["UNKNOWN"]
    elif "PARTIAL BODY" in visibility_states:
        summary = "BODY STATUS: PARTIAL BODY DETECTED"
        summary_color = BODY_COLORS["PARTIAL BODY"]
    elif "FULL BODY" in visibility_states:
        summary = "BODY STATUS: FULL BODY DETECTED"
        summary_color = BODY_COLORS["FULL BODY"]
    else:
        summary = "BODY STATUS: UNKNOWN - TRACK NOT CURRENTLY DETECTED"
        summary_color = BODY_COLORS["UNKNOWN"]

    cv2.rectangle(output, (8, 34), (min(width - 8, 590), 66), (25, 25, 25), -1)
    cv2.putText(
        output, summary, (16, 56), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, summary_color, 2, cv2.LINE_AA
    )
    cv2.putText(
        output, f"CPU | {fps:.1f} FPS | q/Esc: quit", (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 2, cv2.LINE_AA
    )
    return output
