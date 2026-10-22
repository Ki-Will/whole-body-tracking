"""Geometry-only face-to-person association."""
from collections.abc import Sequence
import numpy as np


def associate_faces_to_bodies(
    faces: Sequence[np.ndarray], bodies: Sequence[np.ndarray], min_inside_ratio: float = 0.5
) -> dict[int, int]:
    """Return {body_index: face_index}; ambiguous/multiple assignments are rejected.

    A face must have its centre inside the upper portion of a person box. This is
    spatial association only and must not be interpreted as biometric identity.
    """
    candidates: list[tuple[float, int, int]] = []
    for fi, face in enumerate(faces):
        fx1, fy1, fx2, fy2 = map(float, face)
        face_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
        fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        for bi, body in enumerate(bodies):
            bx1, by1, bx2, by2 = map(float, body)
            body_h = max(1.0, by2 - by1)
            # Head/face should be in the top 55% of the person box.
            if bx1 <= fcx <= bx2 and by1 <= fcy <= by1 + 0.55 * body_h:
                ix1, iy1 = max(fx1, bx1), max(fy1, by1)
                ix2, iy2 = min(fx2, bx2), min(fy2, by2)
                inside = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / face_area
                if inside >= min_inside_ratio:
                    # Prefer the face whose centre is closest to the upper-body centre.
                    target_y = by1 + 0.18 * body_h
                    score = abs(fcx - (bx1 + bx2) / 2) / max(1.0, bx2 - bx1)
                    score += abs(fcy - target_y) / body_h
                    candidates.append((score, bi, fi))
    candidates.sort()
    assigned_bodies, assigned_faces, result = set(), set(), {}
    for _, bi, fi in candidates:
        if bi not in assigned_bodies and fi not in assigned_faces:
            result[bi] = fi
            assigned_bodies.add(bi)
            assigned_faces.add(fi)
    return result
