"""Greedy IoU track manager with motion prediction and explicit uncertainty state."""
from dataclasses import dataclass, field
import numpy as np
from ..geometry import box_iou
from ..identity_state import IdentityEvidence, IdentityState
from .kalman import BoxKalman


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    score: float
    last_seen: float
    missed_frames: int = 0
    filter: BoxKalman = field(init=False)
    identity: IdentityEvidence = field(default_factory=IdentityEvidence)

    def __post_init__(self):
        self.filter = BoxKalman(self.box)


class TrackManager:
    def __init__(self, iou_threshold: float = 0.15, max_missed_frames: int = 20,
                 decay_lambda: float = 0.7, retention_seconds: float = 2.0):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.decay_lambda = decay_lambda
        self.retention_seconds = retention_seconds
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections, timestamp: float) -> list[Track]:
        track_ids = list(self.tracks)
        predicted = {tid: self.tracks[tid].filter.predict() for tid in track_ids}
        pairs = []
        for di, det in enumerate(detections):
            for tid in track_ids:
                overlap = box_iou(det.box, predicted[tid])
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, di, tid))
        pairs.sort(reverse=True)
        matched_dets, matched_tracks = set(), set()
        for _, di, tid in pairs:
            if di in matched_dets or tid in matched_tracks:
                continue
            det = detections[di]
            track = self.tracks[tid]
            track.box = track.filter.correct(det.box)
            track.score = det.score
            track.last_seen = timestamp
            track.missed_frames = 0
            matched_dets.add(di)
            matched_tracks.add(tid)

        for tid in track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track.box = predicted[tid]
                track.missed_frames += 1
                track.identity.update(timestamp, track.missed_frames <= self.max_missed_frames)

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            track = Track(
                track_id=self.next_id, box=np.asarray(det.box, dtype=np.float32),
                score=float(det.score), last_seen=timestamp,
                identity=IdentityEvidence(
                    decay_lambda=self.decay_lambda,
                    retention_seconds=self.retention_seconds,
                ),
            )
            self.tracks[self.next_id] = track
            self.next_id += 1

        expired = [
            tid for tid, track in self.tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]
        for tid in expired:
            del self.tracks[tid]
        return list(self.tracks.values())
