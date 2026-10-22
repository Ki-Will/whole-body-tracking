
"""YOLOX ONNX person detector using ONNX Runtime's CPU provider.

Supports standard YOLOX raw outputs and already-decoded prediction outputs.
COCO class 0 is person.
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
    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.35,
        nms_iou: float = 0.55,
        input_size: int | None = None,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Person model not found: {self.model_path}. "
                "Provide a valid ONNX model path with --model."
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_meta = self.session.get_inputs()[0]
        shape = self.input_meta.shape

        model_h = (
            shape[2]
            if len(shape) == 4 and isinstance(shape[2], int)
            else None
        )
        model_w = (
            shape[3]
            if len(shape) == 4 and isinstance(shape[3], int)
            else None
        )

        if model_h and model_w and model_h != model_w and input_size is None:
            raise ValueError(
                "Non-square model input detected. "
                "Provide a compatible preprocessing adapter."
            )

        self.input_size = int(
            input_size or model_h or model_w or 416
        )

        self.confidence = float(confidence)
        self.nms_iou = float(nms_iou)
        self.output_names = [
            output.name for output in self.session.get_outputs()
        ]

    def describe(self) -> list[dict]:
        return [
            {
                "name": item.name,
                "shape": item.shape,
                "type": item.type,
            }
            for item in self.session.get_inputs()
        ] + [
            {
                "name": item.name,
                "shape": item.shape,
                "type": item.type,
            }
            for item in self.session.get_outputs()
        ]

    def _decode_raw_yolox(self, pred: np.ndarray) -> np.ndarray:
        """Decode the standard raw YOLOX head into pixel coordinates."""

        if pred.ndim != 2 or pred.shape[1] != 85:
            return pred

        expected_count = sum(
            (self.input_size // stride) ** 2
            for stride in (8, 16, 32)
        )

        if pred.shape[0] != expected_count:
            return pred

        grids = []
        strides = []

        for stride in (8, 16, 32):
            grid_size = self.input_size // stride

            yv, xv = np.meshgrid(
                np.arange(grid_size),
                np.arange(grid_size),
                indexing="ij",
            )

            grid = np.stack((xv, yv), axis=-1).reshape(-1, 2)
            grids.append(grid)

            strides.append(
                np.full(
                    (grid_size * grid_size, 1),
                    stride,
                    dtype=np.float32,
                )
            )

        grids = np.concatenate(grids, axis=0).astype(np.float32)
        strides = np.concatenate(strides, axis=0).astype(np.float32)

        # Standard YOLOX decoding: center = (grid + raw_xy) * stride.
        pred = pred.copy()
        pred[:, :2] = (pred[:, :2] + grids) * strides
        pred[:, 2:4] = (
            np.exp(np.clip(pred[:, 2:4], -10, 10)) * strides
        )

        return pred

    def detect(self, frame: np.ndarray) -> list[Detection]:
        original_h, original_w = frame.shape[:2]

        # Preserve aspect ratio and pad on the bottom/right.
        ratio = min(
            self.input_size / original_h,
            self.input_size / original_w,
        )

        resized_w = int(original_w * ratio)
        resized_h = int(original_h * ratio)

        resized = cv2.resize(
            frame,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.full(
            (self.input_size, self.input_size, 3),
            114,
            dtype=np.uint8,
        )
        canvas[:resized_h, :resized_w] = resized

        # BGR -> RGB, HWC -> NCHW.
        blob = (
            canvas[:, :, ::-1]
            .transpose(2, 0, 1)
            .astype(np.float32)[None, ...]
        )

        outputs = self.session.run(
            self.output_names,
            {self.input_meta.name: blob},
        )

        if not outputs:
            return []

        pred = np.asarray(outputs[0]).squeeze()

        if pred.ndim != 2:
            raise ValueError(
                f"Unsupported YOLOX output shape "
                f"{np.asarray(outputs[0]).shape}. "
                "Expected a 2D prediction tensor after removing "
                "the batch dimension."
            )

        # Handle [features, predictions] output layout.
        if pred.shape[0] <= 128 and pred.shape[1] > pred.shape[0]:
            pred = pred.T

        if pred.shape[1] < 6:
            raise ValueError(
                f"Unsupported prediction width {pred.shape[1]}; "
                "expected at least 6 columns."
            )

        # Decode standard raw YOLOX output where applicable.
        pred = self._decode_raw_yolox(pred)

        boxes = []
        scores = []

        for row in pred:
            objectness = float(row[4])
            class_scores = row[5:]

            class_id = int(np.argmax(class_scores))
            score = objectness * float(class_scores[class_id])

            if class_id != 0 or score < self.confidence:
                continue

            cx, cy, bw, bh = map(float, row[:4])

            # Convert model-input coordinates to original-frame pixels.
            x1 = (cx - bw / 2) / ratio
            y1 = (cy - bh / 2) / ratio
            x2 = (cx + bw / 2) / ratio
            y2 = (cy + bh / 2) / ratio

            x1, y1 = np.clip(
                [x1, y1],
                [0, 0],
                [original_w - 1, original_h - 1],
            )
            x2, y2 = np.clip(
                [x2, y2],
                [0, 0],
                [original_w - 1, original_h - 1],
            )

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1),
            ])
            scores.append(score)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence,
            self.nms_iou,
        )

        if len(indices) == 0:
            return []

        detections = []

        for idx in np.asarray(indices).reshape(-1):
            x, y, w, h = boxes[int(idx)]

            detections.append(
                Detection(
                    box=np.array(
                        [x, y, x + w, y + h],
                        dtype=np.float32,
                    ),
                    score=float(scores[int(idx)]),
                    class_id=0,
                )
            )

        return detections
