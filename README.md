# Whole-Body Tracking & Face Association

CPU-first webcam computer vision for person detection, face-to-body association, temporal track continuity, and explicit identity-confidence states.

> **Status:** Initial implementation scaffold. Model files are intentionally not included. Detection quality and runtime must be validated with the exact ONNX exports you download.

## Goals

- Detect people with a pretrained YOLOX ONNX model.
- Detect faces with OpenCV YuNet; optionally compare against Haar Cascade.
- Associate faces with person boxes using spatial geometry, not face recognition.
- Track people between frames using IoU assignment and Kalman motion prediction.
- Expose honest `CONFIRMED`, `TEMPORARY`, and `UNKNOWN` states.
- Decay confidence exponentially while face evidence is missing.
- Visualize target centring and evaluate tracking behaviour on consented webcam/video scenarios.

This project does **not** identify people by name or perform biometric face recognition. A track ID is a temporary video-session identifier, not proof of real-world identity.

## Requirements

- Python 3.10 or 3.11 recommended
- CPU with AVX support recommended; no GPU required
- Webcam or video file for testing
- A pretrained YOLOX ONNX file downloaded separately
- A YuNet ONNX face detector file downloaded separately

See [Model setup](#model-setup). Model weights are excluded from Git and are not downloaded by the initializer.

## Quick start (Linux / Kali Linux)

```bash
git clone https://github.com/Ki-Will/whole-body-tracking.git
cd whole-body-tracking

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

cp .env.example .env  # optional; edit paths/settings if desired
```

On Debian/Kali, if OpenCV cannot open a camera because GUI/video dependencies are missing, install the system packages available for your desktop environment (for example `libgl1`, `libglib2.0-0`, and the appropriate V4L2 tools). Do not install multiple OpenCV wheel variants into the same virtual environment.

## Model setup

Download model files from their official project sources and put them under `models/` (the directory is ignored by Git).

- **YOLOX person detector:** choose a pretrained YOLOX ONNX export compatible with this project's YOLOX output decoder. YOLOX-Nano/Tiny are the first CPU benchmarks; YOLOX-S is a heavier alternative. The legacy release asset named `yolox.onnx` is a large YOLOX-X model, not a generic modern `yolo.onnx` model.
- **YuNet face detector:** obtain the ONNX file from the [OpenCV Zoo YuNet model directory](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
- Optional Haar baseline uses the frontal-face cascade bundled with OpenCV; no extra model download is needed.

Expected default paths:

```text
models/yolox_tiny.onnx
models/face_detection_yunet_2023mar.onnx
```

Override them using CLI flags if your filenames differ. Do not rename an incompatible model merely to match the example path. ONNX exports can differ in output decoding, input dimensions, and output tensor shapes; run the model check before live inference.

## Run

Inspect the selected model's input/output tensors:

```bash
python -m whole_body_tracking.cli check-model --model models/yolox_tiny.onnx
```

Start webcam detection (YuNet face detection and association are enabled by default):

```bash
python -m whole_body_tracking.cli run \
  --model models/yolox_tiny.onnx \
  --face-model models/face_detection_yunet_2023mar.onnx \
  --camera 0
```

Use a video file instead of a webcam:

```bash
python -m whole_body_tracking.cli run --model models/yolox_tiny.onnx --source ./sample.mp4
```

For a different model input size, pass `--input-size 416`; set `--no-face` to isolate person detection. Press `q` or `Esc` to quit.

## State model

- **CONFIRMED** — a current face detection is geometrically associated with a tracked body and passes association gates.
- **TEMPORARY** — face evidence is currently absent, but the body track remains plausible within its configured retention window. Confidence decays with elapsed time.
- **UNKNOWN** — the track is new, evidence is ambiguous, or the retention window has expired. The system does not claim identity continuity without sufficient evidence.

Confidence decay is time-based:

`confidence(t) = confidence_at_last_face * exp(-lambda * elapsed_seconds)`

A Kalman prediction is only motion evidence; it is not identity proof. When crossing tracks cannot be disambiguated, the system should prefer `UNKNOWN` over silently transferring identity.

## Project layout

```text
whole-body-tracking/
├── models/                     # local model files; intentionally not committed
├── src/whole_body_tracking/
│   ├── cli.py                  # command-line entry point
│   ├── config.py               # runtime settings
│   ├── detectors/yolox_onnx.py # CPU ONNX person detector
│   ├── detectors/yunet.py      # YuNet face detector
│   ├── detectors/haar.py       # optional face-detection baseline
│   ├── tracking/kalman.py      # constant-velocity box filter
│   ├── tracking/manager.py     # track lifecycle and IoU assignment
│   ├── association.py          # face-to-body geometry
│   ├── identity_state.py       # three-state model and decay
│   ├── centering.py            # normalized target-centre offset
│   └── visualization.py        # overlay and FPS
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Tests and checks

```bash
python -m pytest
python -m compileall -q src tests
```

The unit tests do not require model weights or a webcam. Full integration checks require valid local ONNX files and camera access.

## Evaluation plan

Use consented participants and repeat each scenario several times:

1. Frontal face and normal walking.
2. Head turn / face detector loss while the body remains visible.
3. Brief body occlusion and reappearance.
4. Two people crossing paths.
5. Lighting and distance changes.

Record detection FPS, track fragmentation, ID switches, face/body association precision, time spent in each state, and confidence decay. Treat `UNKNOWN` during ambiguous crossings as a valid conservative outcome, not automatically as a failure.

## Limitations

- A person detector and a face detector do not provide biometric identity.
- Geometry-only face/body matching can be ambiguous when people overlap or cross.
- YOLOX ONNX output formats vary; the current decoder expects decoded `[cx, cy, width, height, objectness, class scores...]` rows. Raw grid outputs or custom layouts need an export-specific decoder adapter.
- CPU throughput depends on model size, input resolution, camera resolution, and system load.

## License

Project code is provided under the MIT License unless a file states otherwise. Third-party model weights and dependencies retain their own licenses; check those terms before redistribution.
