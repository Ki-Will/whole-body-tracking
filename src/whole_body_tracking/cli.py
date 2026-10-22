"""Command-line interface for model checks and webcam/video inference."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np

import cv2

from .config import Settings
from .detectors.mediapipe_person import MediaPipePersonDetector
from .detectors.yolox_onnx import YOLOXONNXDetector
from .detectors.yunet import YuNetFaceDetector
from .detectors.haar import HaarFaceDetector
from .association import associate_faces_to_bodies
from .identity_state import IdentityState
from .tracking.manager import TrackManager
from .visualization import FPSMeter, draw_overlay
from .face_recognition import ArcFaceEmbedder, FaceIdentityDatabase


def _add_person_detector_arguments(parser):
    parser.add_argument(
        "--person-backend",
        choices=("yolox", "mediapipe"),
        default="yolox",
        help="person detector backend (default: yolox)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="person detector ONNX path; defaults according to --person-backend",
    )
    parser.add_argument("--input-size", type=int, default=None)


def _make_person_detector(args):
    settings = Settings()
    model_path = args.model
    if model_path is None:
        model_path = (
            "models/person_detection_mediapipe_2023mar.onnx"
            if args.person_backend == "mediapipe"
            else str(settings.person_model)
        )

    if args.person_backend == "mediapipe":
        confidence_arg = getattr(args, "confidence", None)
        confidence = confidence_arg if confidence_arg is not None else 0.5
        nms_iou_arg = getattr(args, "nms_iou", None)
        nms_iou = nms_iou_arg if nms_iou_arg is not None else 0.3
        if args.input_size is not None and args.input_size != 224:
            raise ValueError("MediaPipe person detection uses a fixed 224x224 input.")
        return MediaPipePersonDetector(
            model_path,
            confidence=confidence,
            nms_iou=nms_iou,
        )

    confidence_arg = getattr(args, "confidence", None)
    confidence = (
        confidence_arg if confidence_arg is not None else settings.person_confidence
    )
    nms_iou_arg = getattr(args, "nms_iou", None)
    nms_iou = nms_iou_arg if nms_iou_arg is not None else settings.nms_iou
    return YOLOXONNXDetector(
        model_path,
        confidence=confidence,
        nms_iou=nms_iou,
        input_size=args.input_size,
    )


def build_parser():
    parser = argparse.ArgumentParser(prog="whole-body-tracking")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check-model", help="print ONNX model tensor metadata")
    _add_person_detector_arguments(check)

    run = sub.add_parser("run", help="run person tracking on webcam or video")
    _add_person_detector_arguments(run)
    run.add_argument("--face-model", default=str(Settings().face_model))
    run.add_argument("--arcface-model", default="models/embedder_arcface.onnx")
    run.add_argument("--recognition-threshold", type=float, default=0.60)
    run.add_argument("--enrollment-db", default="data/db/face_db.npz")
    run.add_argument("--enroll-name", default=None, help="identity name to enroll when pressing e")
    run.add_argument("--camera", type=int, default=Settings().camera_index)
    run.add_argument("--source", default=None, help="video path; takes precedence over --camera")
    run.add_argument("--confidence", type=float, default=None)
    run.add_argument("--nms-iou", type=float, default=None)
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
        detector = _make_person_detector(args)
        for tensor in detector.describe():
            print(f"{tensor['name']}: shape={tensor['shape']} type={tensor['type']}")
        print("Execution providers:", detector.session.get_providers())
        return 0

    detector = _make_person_detector(args)
    face_detector = None
    if not args.no_face:
        face_detector = (
            YuNetFaceDetector(args.face_model, confidence=args.face_confidence)
            if args.face_backend == "yunet" else HaarFaceDetector()
        )

    embedder = None
    identity_db = None
    if not args.no_face and Path(args.arcface_model).is_file():
        embedder = ArcFaceEmbedder(args.arcface_model)
        identity_db = FaceIdentityDatabase(args.enrollment_db)
        print(f"[FACE] loaded {len(identity_db.templates)} enrolled identities")
    elif not args.no_face:
        print(f"[FACE] ArcFace model not found: {args.arcface_model}; recognition is disabled.")

    source = args.source if args.source is not None else args.camera
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source {source!r}. "
            "Check camera permissions or video path."
        )

    manager = TrackManager(
        decay_lambda=args.decay_lambda,
        retention_seconds=args.retention_seconds,
    )
    fps_meter = FPSMeter()
    started = time.perf_counter()
    frame_count = 0
    locked_track_id = None
    recognized_names = {}
    latest_embeddings = {}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = time.perf_counter() - started
            detections = detector.detect(frame)
            tracks = manager.update(detections, timestamp)
            faces = face_detector.detect(frame) if face_detector else []

            # Associate faces only with bodies detected in this frame.
            active_indices = [
                i for i, track in enumerate(tracks)
                if track.missed_frames == 0
            ]
            local_associations = associate_faces_to_bodies(
                faces,
                [tracks[i].box for i in active_indices],
            ) if face_detector else {}
            associations = {
                active_indices[body_index]: face_index
                for body_index, face_index in local_associations.items()
            }

            associated_indices = set(associations)
            latest_embeddings = {}
            for body_index in associated_indices:
                track = tracks[body_index]
                face_confidence = max(0.0, min(1.0, float(track.score)))
                track.identity.observe_face(timestamp, face_confidence)
                if embedder is not None and identity_db is not None:
                    fx1, fy1, fx2, fy2 = map(int, faces[associations[body_index]])
                    pad = int(max(fx2-fx1, fy2-fy1) * 0.20)
                    fx1, fy1 = max(0, fx1-pad), max(0, fy1-pad)
                    fx2, fy2 = min(frame.shape[1], fx2+pad), min(frame.shape[0], fy2+pad)
                    face_crop = frame[fy1:fy2, fx1:fx2]
                    if face_crop.size:
                        embedding = embedder.embed(face_crop)
                        latest_embeddings[track.track_id] = embedding
                        name, similarity = identity_db.match(
                            embedding, threshold=args.recognition_threshold
                        )
                        if name:
                            recognized_names[track.track_id] = (name, similarity)
            for index, track in enumerate(tracks):
                if index not in associated_indices:
                    track.identity.update(
                        timestamp,
                        body_track_valid=track.missed_frames <= manager.max_missed_frames,
                    )

            fps = fps_meter.tick()
            if not args.no_display:
                view = draw_overlay(frame, tracks, associations, faces, fps, timestamp)
                for track in tracks:
                    x1, y1, x2, y2 = map(int, track.box)
                    if track.track_id in recognized_names:
                        name, similarity = recognized_names[track.track_id]
                        cv2.putText(view, f"{name} {similarity:.2f}",
                                    (max(4,x1), min(view.shape[0]-8,y2+34)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,220,90), 2, cv2.LINE_AA)
                    if locked_track_id == track.track_id:
                        cv2.rectangle(view, (x1-4,y1-4), (x2+4,y2+4), (0,0,255), 3)
                        cv2.putText(view, "LOCKED TARGET", (max(4,x1), max(18,y1-28)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(view, "e=enroll  r=recognize  l=lock/unlock  q=quit",
                            (12, view.shape[0]-14), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (255,255,255), 1, cv2.LINE_AA)
                cv2.imshow("Whole-Body Tracking (CPU)", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("l"):
                    if locked_track_id is not None:
                        print("[LOCK] target unlocked")
                        locked_track_id = None
                    else:
                        cx, cy = frame.shape[1]/2, frame.shape[0]/2
                        target = min((t for t in tracks if t.missed_frames == 0),
                            key=lambda t: ((float(t.box[0]+t.box[2])/2-cx)**2 +
                                           (float(t.box[1]+t.box[3])/2-cy)**2), default=None)
                        if target is not None:
                            locked_track_id = target.track_id
                            print(f"[LOCK] body track {locked_track_id} selected; visual lock only")
                if key in (ord("e"), ord("r")):
                    target_id = locked_track_id
                    if target_id is None and tracks:
                        cx, cy = frame.shape[1]/2, frame.shape[0]/2
                        target = min((t for t in tracks if t.missed_frames == 0),
                            key=lambda t: ((float(t.box[0]+t.box[2])/2-cx)**2 +
                                           (float(t.box[1]+t.box[3])/2-cy)**2), default=None)
                        target_id = target.track_id if target else None
                    embedding = latest_embeddings.get(target_id)
                    if embedding is None or identity_db is None:
                        print("[FACE] no current face embedding or ArcFace model/database unavailable")
                    elif key == ord("e"):
                        name = args.enroll_name
                        if not name:
                            try:
                                name = input("Name to enroll: ").strip()
                            except EOFError:
                                name = ""
                        if name:
                            # Collect samples across successive frames while key is held
                            # is not implemented; require repeated presses and average later.
                            identity_db.templates[name] = embedding
                            identity_db.path.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(identity_db.path, **identity_db.templates)
                            identity_db.path.with_suffix(".json").write_text(
                                json.dumps({"names": sorted(identity_db.templates)}, indent=2),
                                encoding="utf-8")
                            print(f"[ENROLL] saved single template for {name!r}")
                    else:
                        name, similarity = identity_db.match(
                            embedding, threshold=args.recognition_threshold
                        )
                        if name:
                            recognized_names[target_id] = (name, similarity)
                            print(f"[RECOGNITION] {name}: {similarity:.3f}")
                        else:
                            print(f"[RECOGNITION] no match; best similarity={similarity:.3f}")
            frame_count += 1
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
