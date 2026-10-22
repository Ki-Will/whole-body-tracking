"""Command-line interface for model checks and webcam/video inference."""
import argparse
from pathlib import Path
import sys
import time
import cv2
from .config import Settings
from .detectors.yolox_onnx import YOLOXONNXDetector
from .detectors.yunet import YuNetFaceDetector
from .detectors.haar import HaarFaceDetector
from .association import associate_faces_to_bodies
from .identity_state import IdentityState
from .tracking.manager import TrackManager
from .visualization import FPSMeter, draw_overlay
from .face_landmarks import detect_faces_with_landmarks
from .face_recognition import ArcFaceEmbedder, FaceIdentityDatabase, align_face


def build_parser():
    parser = argparse.ArgumentParser(prog="whole-body-tracking")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-model", help="print ONNX model tensor metadata")
    check.add_argument("--model", default=str(Settings().person_model))
    check.add_argument("--input-size", type=int, default=None)

    run = sub.add_parser("run", help="run person tracking on webcam or video")
    run.add_argument("--model", default=str(Settings().person_model))
    run.add_argument("--face-model", default=str(Settings().face_model))
    run.add_argument("--arcface-model", default=None, help="ArcFace ONNX model for enrolled identity recognition")
    run.add_argument("--face-database", default="data/db/face_db.npz", help="local enrolled face database")
    run.add_argument("--match-threshold", type=float, default=0.45, help="cosine similarity threshold; calibrate for your model")
    run.add_argument("--camera", type=int, default=Settings().camera_index)
    run.add_argument("--source", default=None, help="video path; takes precedence over --camera")
    run.add_argument("--input-size", type=int, default=None)
    run.add_argument("--confidence", type=float, default=Settings().person_confidence)
    run.add_argument("--nms-iou", type=float, default=Settings().nms_iou)
    run.add_argument("--face-confidence", type=float, default=Settings().face_confidence)
    run.add_argument("--face-backend", choices=("yunet", "haar"), default="yunet")
    run.add_argument("--no-face", action="store_true", help="disable face detection/association")
    run.add_argument("--retention-seconds", type=float, default=Settings().temporary_retention_seconds)
    run.add_argument("--decay-lambda", type=float, default=Settings().confidence_decay_lambda)
    run.add_argument("--no-display", action="store_true", help="process frames without a display window")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "check-model":
        detector = YOLOXONNXDetector(args.model, input_size=args.input_size)
        for tensor in detector.describe():
            print(f"{tensor['name']}: shape={tensor['shape']} type={tensor['type']}")
        print("Execution providers:", detector.session.get_providers())
        return 0

    detector = YOLOXONNXDetector(
        args.model, confidence=args.confidence, nms_iou=args.nms_iou,
        input_size=args.input_size,
    )
    face_detector = None
    if not args.no_face:
        face_detector = (
            YuNetFaceDetector(args.face_model, confidence=args.face_confidence)
            if args.face_backend == "yunet" else HaarFaceDetector()
        )

    embedder = ArcFaceEmbedder(args.arcface_model) if getattr(args, "arcface_model", None) else None
    face_db = FaceIdentityDatabase(args.face_database) if embedder else None
    if embedder and not face_db.templates:
        raise RuntimeError(f"No enrolled identities in {args.face_database}. Enroll first with: python -m whole_body_tracking.enrollment --name YOUR_NAME --camera {args.camera} --face-model {args.face_model} --arcface-model {args.arcface_model}")

    source = args.source if args.source is not None else args.camera
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source {source!r}. Check camera permissions or video path.")

    manager = TrackManager(
        decay_lambda=args.decay_lambda,
        retention_seconds=args.retention_seconds,
    )
    fps_meter = FPSMeter()
    started = time.perf_counter()
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = time.perf_counter() - started
            detections = detector.detect(frame)
            tracks = manager.update(detections, timestamp)
            face_records = detect_faces_with_landmarks(face_detector, frame) if face_detector and embedder and args.face_backend == "yunet" else []
            faces = [record["box"] for record in face_records] if face_records else (face_detector.detect(frame) if face_detector else [])
            # Only associate faces with bodies detected in this frame.
            # Stale predicted boxes remain visible but cannot claim a face.
            active_indices = [i for i, track in enumerate(tracks) if track.missed_frames == 0]
            local_associations = associate_faces_to_bodies(
                faces, [tracks[i].box for i in active_indices]
            ) if face_detector else {}
            associations = {
                active_indices[body_index]: face_index
                for body_index, face_index in local_associations.items()
            }

            associated_indices = set(associations)
            for body_index in associated_indices:
                track = tracks[body_index]
                face_index = associations[body_index]
                if embedder is not None and face_db is not None and face_records and args.face_backend == "yunet":
                    try:
                        vector = embedder.embed(align_face(frame, face_records[face_index]["landmarks"]))
                        matched_name, similarity = face_db.match(vector, threshold=args.match_threshold)
                        if matched_name is not None:
                            track.identity.observe_face(timestamp, max(0.0, min(1.0, similarity)))
                            track.identity_name = matched_name
                        else:
                            track.identity.update(timestamp, body_track_valid=True)
                    except (ValueError, cv2.error):
                        track.identity.update(timestamp, body_track_valid=True)
                else:
                    # A face detection or geometric association alone cannot confirm identity.
                    track.identity.update(timestamp, body_track_valid=True)
            for index, track in enumerate(tracks):
                if index not in associated_indices:
                    track.identity.update(
                        timestamp,
                        body_track_valid=track.missed_frames <= manager.max_missed_frames,
                    )

            fps = fps_meter.tick()
            if not args.no_display:
                view = draw_overlay(frame, tracks, associations, faces, fps, timestamp)
                cv2.imshow("Whole-Body Tracking (CPU)", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            frame_count += 1
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
