"""MediaPipe person detector from the OpenCV Zoo ONNX export.

The model returns 2,254 box/keypoint regressions and score logits. This adapter
uses the model's SSD anchors and official RGB [-1, 1] preprocessing, then emits
the same Detection objects as the YOLOX adapter.
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .yolox_onnx import Detection


class MediaPipePersonDetector:
    INPUT_SIZE = 224
    EXPECTED_ANCHORS = 2254

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.5,
        nms_iou: float = 0.3,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe person model not found: {self.model_path}. "
                "Download person_detection_mediapipe_2023mar.onnx from "
                "the OpenCV Zoo person_detection_mediapipe model directory."
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_meta = self.session.get_inputs()[0]
        self.output_metas = self.session.get_outputs()
        self.output_names = [item.name for item in self.output_metas]
        self.confidence = float(confidence)
        self.nms_iou = float(nms_iou)
        self.anchors = self._generate_anchors()

        input_shape = self.input_meta.shape
        if len(input_shape) != 4 or input_shape[1:] != [3, 224, 224]:
            raise ValueError(
                f"Expected MediaPipe NCHW input [1, 3, 224, 224], "
                f"got {input_shape}"
            )

    @staticmethod
    def _generate_anchors() -> np.ndarray:
        """Generate the 2,254 normalized SSD anchors used by this model."""
        anchors = []
        # These feature maps and repetitions match OpenCV Zoo's model decoder.
        for grid_size, anchors_per_cell in ((28, 2), (14, 2), (7, 6)):
            positions = (np.arange(grid_size, dtype=np.float32) + 0.5) / grid_size
            yv, xv = np.meshgrid(positions, positions, indexing="ij")
            centers = np.stack((xv, yv), axis=-1).reshape(-1, 2)
            anchors.append(np.repeat(centers, anchors_per_cell, axis=0))

        result = np.concatenate(anchors, axis=0).astype(np.float32)
        if result.shape != (MediaPipePersonDetector.EXPECTED_ANCHORS, 2):
            raise RuntimeError(f"Unexpected anchor shape: {result.shape}")
        return result

    def describe(self) -> list[dict]:
        return [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in self.session.get_inputs()
        ] + [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in self.output_metas
        ]

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        logits = np.clip(np.asarray(logits, dtype=np.float64), -100.0, 100.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def _decode(
        self,
        box_delta: np.ndarray,
        score_logits: np.ndarray,
        original_width: int,
        original_height: int,
        pad_bias: np.ndarray,
    ) -> list[Detection]:
        box_delta = np.asarray(box_delta, dtype=np.float32).reshape(-1, 12)
        score_logits = np.asarray(score_logits, dtype=np.float32).reshape(-1)
        if len(box_delta) != len(self.anchors) or len(score_logits) != len(self.anchors):
            raise ValueError(
                "Unexpected MediaPipe output count: "
                f"boxes={len(box_delta)}, scores={len(score_logits)}, "
                f"anchors={len(self.anchors)}"
            )

        scores = self._sigmoid(score_logits)
        scale = float(max(original_width, original_height))

        center_delta = box_delta[:, :2] / self.INPUT_SIZE
        size_delta = box_delta[:, 2:4] / self.INPUT_SIZE
        xy1 = (center_delta - size_delta / 2.0 + self.anchors) * scale
        xy2 = (center_delta + size_delta / 2.0 + self.anchors) * scale
        decoded = np.concatenate((xy1, xy2), axis=1)
        decoded -= np.array(
            [pad_bias[0], pad_bias[1], pad_bias[0], pad_bias[1]],
            dtype=np.float32,
        )

        boxes = []
        valid_scores = []
        for box, score in zip(decoded, scores):
            if not np.isfinite(box).all() or not np.isfinite(score):
                continue
            if float(score) < self.confidence:
                continue

            x1, y1, x2, y2 = map(float, box)
            x1 = float(np.clip(x1, 0, original_width - 1))
            y1 = float(np.clip(y1, 0, original_height - 1))
            x2 = float(np.clip(x2, 0, original_width - 1))
            y2 = float(np.clip(y2, 0, original_height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([
                int(x1),
                int(y1),
                max(1, int(x2 - x1)),
                max(1, int(y2 - y1)),
            ])
            valid_scores.append(float(score))

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            valid_scores,
            self.confidence,
            self.nms_iou,
        )
        detections = []
        for index in np.asarray(indices).reshape(-1) if len(indices) else []:
            x, y, width, height = boxes[int(index)]
            detections.append(
                Detection(
                    box=np.array(
                        [x, y, x + width, y + height],
                        dtype=np.float32,
                    ),
                    score=valid_scores[int(index)],
                    class_id=0,
                )
            )
        return detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        original_height, original_width = frame.shape[:2]
        if original_height <= 0 or original_width <= 0:
            return []

        # Official preprocessing: BGR -> RGB, normalize to [-1, 1],
        # aspect-preserving resize, and centered zero padding.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0 - 0.5) * 2.0
        ratio = min(
            self.INPUT_SIZE / original_height,
            self.INPUT_SIZE / original_width,
        )
        resized_width = max(1, int(original_width * ratio))
        resized_height = max(1, int(original_height * ratio))
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_width = self.INPUT_SIZE - resized_width
        pad_height = self.INPUT_SIZE - resized_height
        left = pad_width // 2
        top = pad_height // 2
        canvas = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype=np.float32)
        canvas[top:top + resized_height, left:left + resized_width] = image
        blob = canvas.transpose(2, 0, 1)[None, ...]

        outputs = self.session.run(
            self.output_names,
            {self.input_meta.name: blob.astype(np.float32)},
        )

        # Do not rely on output names/order: the ONNX Runtime output order
        # differs from OpenCV DNN's unconnected-layer order for this model.
        box_output = None
        score_output = None
        for output in outputs:
            array = np.asarray(output).squeeze()
            if array.ndim == 2 and array.shape == (self.EXPECTED_ANCHORS, 12):
                box_output = array
            elif array.size == self.EXPECTED_ANCHORS:
                score_output = array.reshape(-1)

        if box_output is None or score_output is None:
            shapes = [np.asarray(output).shape for output in outputs]
            raise ValueError(
                f"Unsupported MediaPipe output shapes {shapes}; expected "
                f"[1, 2254, 12] box regressions and [1, 2254, 1] score logits."
            )

        # Convert padding from model-input pixels back to original-image pixels.
        pad_bias = np.array([left / ratio, top / ratio], dtype=np.float32)
        return self._decode(
            box_output,
            score_output,
            original_width,
            original_height,
            pad_bias,
        )
