"""Local webcam enrollment for an ArcFace-compatible ONNX model."""
import argparse
import cv2
from .detectors.yunet import YuNetFaceDetector
from .face_landmarks import detect_faces_with_landmarks
from .face_recognition import ArcFaceEmbedder, FaceIdentityDatabase, align_face


def enroll(name, camera, face_model, arcface_model, database_path, samples_needed=15):
    detector = YuNetFaceDetector(face_model, confidence=0.75)
    embedder = ArcFaceEmbedder(arcface_model)
    database = FaceIdentityDatabase(database_path)
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera}")
    samples = []
    print("Look at the camera and slowly turn your head. SPACE captures; Q/Esc cancels.")
    try:
        while len(samples) < samples_needed:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a camera frame")
            faces = detect_faces_with_landmarks(detector, frame)
            preview = frame.copy()
            for face in faces:
                x1, y1, x2, y2 = map(int, face['box'])
                cv2.rectangle(preview, (x1, y1), (x2, y2), (60, 200, 80), 2)
            cv2.putText(preview, f"{name}: {len(samples)}/{samples_needed} | SPACE capture | Q quit", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Local face enrollment", preview)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                print("Enrollment cancelled; database unchanged.")
                return 1
            if key == ord(' '):
                if len(faces) != 1:
                    print("Ensure exactly one face is visible before capture.")
                    continue
                try:
                    samples.append(embedder.embed(align_face(frame, faces[0]['landmarks'])))
                    print(f"Captured {len(samples)}/{samples_needed}")
                except (ValueError, cv2.error) as exc:
                    print(f"Could not align face: {exc}")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    database.save_identity(name, samples)
    print(f"Saved local face template for {name!r} to {database.path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Enroll a consenting person locally using a webcam")
    parser.add_argument('--name', required=True)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--face-model', required=True)
    parser.add_argument('--arcface-model', required=True)
    parser.add_argument('--database', default='data/db/face_db.npz')
    parser.add_argument('--samples', type=int, default=15)
    args = parser.parse_args()
    if args.samples < 3:
        parser.error('--samples must be at least 3')
    return enroll(args.name, args.camera, args.face_model, args.arcface_model, args.database, args.samples)


if __name__ == '__main__':
    raise SystemExit(main())
