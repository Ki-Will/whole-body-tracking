"""Conservative three-state identity evidence model."""
from dataclasses import dataclass
from enum import Enum
import math


class IdentityState(str, Enum):
    CONFIRMED = "CONFIRMED"
    TEMPORARY = "TEMPORARY"
    UNKNOWN = "UNKNOWN"


@dataclass
class IdentityEvidence:
    decay_lambda: float = 0.7
    retention_seconds: float = 2.0
    state: IdentityState = IdentityState.UNKNOWN
    confidence_at_last_face: float = 0.0
    last_face_time: float | None = None

    def observe_face(self, timestamp: float, confidence: float) -> None:
        self.last_face_time = timestamp
        self.confidence_at_last_face = max(0.0, min(1.0, confidence))
        self.state = IdentityState.CONFIRMED

    def update(self, timestamp: float, body_track_valid: bool) -> IdentityState:
        if self.last_face_time is None:
            self.state = IdentityState.UNKNOWN
            return self.state
        elapsed = max(0.0, timestamp - self.last_face_time)
        decayed = self.confidence_at_last_face * math.exp(-self.decay_lambda * elapsed)
        if body_track_valid and elapsed <= self.retention_seconds and decayed > 0.05:
            self.state = IdentityState.TEMPORARY
        else:
            self.state = IdentityState.UNKNOWN
        return self.state

    def confidence(self, timestamp: float) -> float:
        if self.last_face_time is None:
            return 0.0
        elapsed = max(0.0, timestamp - self.last_face_time)
        return self.confidence_at_last_face * math.exp(-self.decay_lambda * elapsed)
