"""Configuration defaults and environment loading."""
from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    person_model: Path = Path(os.getenv("PERSON_MODEL", "models/yolox_tiny.onnx"))
    face_model: Path = Path(os.getenv("FACE_MODEL", "models/face_detection_yunet_2023mar.onnx"))
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    person_confidence: float = float(os.getenv("PERSON_CONFIDENCE", "0.35"))
    nms_iou: float = float(os.getenv("NMS_IOU", "0.45"))
    face_confidence: float = float(os.getenv("FACE_CONFIDENCE", "0.80"))
    confidence_decay_lambda: float = float(os.getenv("CONFIDENCE_DECAY_LAMBDA", "0.7"))
    temporary_retention_seconds: float = float(os.getenv("TEMPORARY_RETENTION_SECONDS", "2.0"))
