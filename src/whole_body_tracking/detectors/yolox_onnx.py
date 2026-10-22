"""YOLOX ONNX person detector using ONNX Runtime's CPU provider.

Supports common YOLOX exports whose output rows are [cx, cy, w, h, obj, class...]
and whose coordinates are already decoded into input-image pixels. Some custom
exports use a different layout; inspect them with the check-model CLI first.
"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    box: np.ndarray  # xyxy in original-image pixel coordinates
    score: float
    class_id: int


class YOLOXONNXDetector:
    def __init__(self, model_path: str | Path, confidence: float = 0.35,
                 nms_iou: float = 0.45, input_size: int | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Person model not found: {self.model_path}. Download an ONNX model "
                "separately and pass its path with --model."
            )
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input_meta = self.session.get_inputs()[0]
        shape = self.input_meta.shape
        # NCHW is the usual YOLOX export. Use explicit --input-size for dynamic models.
        model_h = shape[2] if len(shape) == 4 and isinstance(shape[2], int) else None
        model_w = shape[3] if len(shape) == 4 and isinstance(shape[3], int) else None
        self.input_size = int(input_size or model_h or model_w or 416)
        if model_h and model_w and model_h != model_w and input_size is None:
            raise ValueError("Non-square model input detected; provide a matching adapter/input size.")
        self.confidence = float(confidence)
        self.nms_iou = float(nms_iou)
        self.output_names = [o.name for o in self.session.get_outputs()]

    def describe(self) -> list[dict]:
        return [{
            "name": item.name,
            "shape": item.shape,
            "type": item.type,
        } for item in self.session.get_inputs()] + [{
            "name": item.name, "shape": item.shape, "type": item.type
        } for item in self.session.get_outputs()]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        original_h, original_w = frame.shape[:2]
        # YOLOX preprocessing uses aspect-preserving resize and top-left padding.
        ratio = min(self.input_size / original_h, self.input_size / original_w)
        resized_w, resized_h = int(original_w * ratio), int(original_h * ratio)
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[:resized_h, :resized_w] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)[None]  # BGR -> RGB
        outputs = self.session.run(self.output_names, {self.input_meta.name: blob})
        if not outputs:
            return []
        pred = np.asarray(outputs[0])
        pred = np.squeeze(pred)
        if pred.ndim != 2:
            raise ValueError(
                f"Unsupported YOLOX output shape {np.asarray(outputs[0]).shape}; "
                "expected one prediction tensor shaped [N, 5 + classes] or [5 + classes, N]."
            )
        # Accommodate exports transposed to [features, predictions].
        if pred.shape[0] <= 128 and pred.shape[1] > pred.shape[0]:
            pred = pred.T
        if pred.shape[1] < 6:
            raise ValueError(f"Unsupported prediction width {pred.shape[1]}; expected at least 6.")

        boxes, scores = [], []
        for row in pred:
            objectness = float(row[4])
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            score = objectness * float(class_scores[class_id])
            # COCO class 0 is person.
            if class_id != 0 or score < self.confidence:
                continue
            cx, cy, bw, bh = map(float, row[:4])
            x1, y1 = (cx - bw / 2) / ratio, (cy - bh / 2) / ratio
            x2, y2 = (cx + bw / 2) / ratio, (cy + bh / 2) / ratio
            x1, y1 = np.clip([x1, y1], [0, 0], [original_w - 1, original_h - 1])
            x2, y2 = np.clip([x2, y2], [0, 0], [original_w - 1, original_h - 1])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores.append(score)

        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.nms_iou)
        if len(indices) == 0:
            return []
        result = []
        for idx in np.asarray(indices).reshape(-1):
            x, y, w, h = boxes[int(idx)]
            result.append(Detection(
                box=np.array([x, y, x + w, y + h], dtype=np.float32),
                score=float(scores[int(idx)]),
                class_id=0,
            ))
        return result
