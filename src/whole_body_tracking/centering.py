"""Target-centering metrics; this module does not control physical hardware."""
from collections.abc import Sequence


def centering_offset(box: Sequence[float], frame_width: int, frame_height: int) -> tuple[float, float]:
    """Normalized target-centre offset from frame centre, each axis in roughly [-1, 1]."""
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (
        (cx - frame_width / 2) / max(1.0, frame_width / 2),
        (cy - frame_height / 2) / max(1.0, frame_height / 2),
    )
