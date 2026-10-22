"""Bounding-box geometry helpers. Boxes use xyxy pixel coordinates."""
from typing import Sequence
import numpy as np


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center(box: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def clamp_box(box: Sequence[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    return np.array([
        np.clip(x1, 0, width - 1), np.clip(y1, 0, height - 1),
        np.clip(x2, 0, width - 1), np.clip(y2, 0, height - 1),
    ], dtype=np.float32)
