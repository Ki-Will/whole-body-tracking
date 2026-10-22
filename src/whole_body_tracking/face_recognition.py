"""Local ArcFace enrollment and recognition for face-locked tracking.

Requires an ArcFace-compatible ONNX embedding model supplied by the user.
YuNet supplies five facial landmarks; ArcFace embeddings identify enrolled faces.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Similarity-align YuNet's five landmarks to the ArcFace 112x112 template."""
    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(points, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("Could not align face landmarks")
    return cv2.warpAffine(frame, matrix, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


class ArcFaceEmbedder:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"ArcFace embedding model not found: {self.model_path}. "
                "YuNet only detects faces; provide an ArcFace-compatible ONNX model "
                "to enroll and recognize identities."
            )
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        image = cv2.resize(aligned_bgr, (112, 112))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        tensor = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)
        result = self.session.run([self.output.name], {self.input.name: tensor})[0]
        vector = np.asarray(result, dtype=np.float32).reshape(-1)
        return vector / (np.linalg.norm(vector) + 1e-12)


class FaceIdentityDatabase:
    """Stores one normalized template embedding per enrolled identity in NPZ."""

    def __init__(self, path: str | Path = "data/db/face_db.npz"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.templates: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as data:
                self.templates = {
                    key: np.asarray(data[key], dtype=np.float32).reshape(-1)
                    for key in data.files
                }

    def save_identity(self, name: str, embeddings: list[np.ndarray]) -> None:
        if not name.strip():
            raise ValueError("Identity name cannot be empty")
        if len(embeddings) < 3:
            raise ValueError("Capture at least 3 face samples before saving")
        mean = np.mean(np.stack(embeddings).astype(np.float32), axis=0)
        mean /= np.linalg.norm(mean) + 1e-12
        self.templates[name.strip()] = mean.astype(np.float32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **self.templates)
        metadata = {
            "names": sorted(self.templates),
            "embedding_dim": int(mean.size),
            "updated_at": __import__("time").time(),
        }
        self.path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def match(self, embedding: np.ndarray, threshold: float = 0.55) -> tuple[Optional[str], float]:
        if not self.templates:
            return None, 0.0
        scores = {
            name: float(np.dot(embedding, template) /
                        ((np.linalg.norm(embedding) * np.linalg.norm(template)) + 1e-12))
            for name, template in self.templates.items()
            if template.shape == embedding.shape
        }
        if not scores:
            return None, 0.0
        name, score = max(scores.items(), key=lambda item: item[1])
        # Threshold is configurable: cosine-similarity thresholds depend on the
        # exact ArcFace export and must be calibrated with real positive/negative samples.
        return (name, score) if score >= threshold else (None, score)
