# Face Detection Pipeline

A modular face processing pipeline for image, video, stream URL, and webcam input.

This is **Component 1** of the deepfake-verification system under `WORKING/`:
`frame` (this repo, stage 1) -> `RPPG/` (stage 2) -> `quantum/` (stage 3).
It is invoked automatically by `WORKING/run_pipeline.py` (stage 1: frame sampling
+ quality assessment feeding the rPPG and quantum stages).

## Project Structure

project_root/
- app/
  - config.py
  - detector.py
  - extract_frames.py
  - ingest.py
  - logger.py
  - main.py
  - processor.py
  - quality.py
  - storage.py
- weights/
  - yolov8n-face-lindevs.pt
  - yolov8x-face-lindevs.pt
  - yolov9e-face-lindevs.pt
  - yolov9t-face-lindevs.pt
- output/
  - full_frames/
  - cropped_faces/
  - metadata.jsonl
- requirements.txt
- README.md

## Module Responsibilities

- app/main.py
  - Entry point. Runs the face detection/cropping/storage pipeline and then triggers the frame sampling + quality layer for local video sources.
- app/config.py
  - Stores model preset mapping and builds normalized runtime configuration paths.
- app/ingest.py
  - Handles source input (image/video/webcam/stream) and yields ordered frame packets with timestamps.
- app/detector.py
  - Loads YOLO face model weights, resolves device (CPU/GPU), and returns face detections.
- app/extract_frames.py
  - Implements the Frame Sampling Layer and orchestrates Frame Quality Assessment, reporting, and metadata export.
- app/processor.py
  - Draws detection annotations and extracts face crops from each frame.
- app/quality.py
  - Applies frame quality rules (blur/brightness/face checks), rejection taxonomy, and composite quality scoring.
- app/storage.py
  - Saves annotated frames, face crops, and metadata asynchronously.
- app/logger.py
  - Provides shared logger setup for consistent console logs.

### App File Usage Summary

- All files in app are part of the pipeline.
- Running `python app/main.py ...` uses: config.py, detector.py, ingest.py, processor.py, storage.py, logger.py directly.
- Running `python app/main.py ...` also invokes extract_frames.py, which uses quality.py for quality scoring and filtering.

### Architecture Diagram (ASCII)

```text
        +----------------+
        |   app/main.py  |
        +----------------+
         | direct calls
         v
  +-----------+  +------------+  +-----------+  +-------------+  +-----------+
  | config.py |  | ingest.py  |  |detector.py|  | processor.py |  | storage.py|
  +-----------+  +------------+  +-----------+  +-------------+  +-----------+
     ^               |              |               |               |
     |               +--------------+---------------+---------------+
     |                              data flow
     |
  +----------------+
  |   logger.py    |
  +----------------+

        indirect call from main.py
          |
          v
        +----------------------+
        | app/extract_frames.py|
        +----------------------+
           | uses          | uses
           v               v
         +------------+   +-----------+
         | ingest.py  |   | quality.py|
         +------------+   +-----------+
           | uses          | uses
           v               v
         +-----------+   +-----------+
         |detector.py|   | logger.py |
         +-----------+   +-----------+
```

## Pipeline Flow

Input (image/video/webcam)
-> Frame Extraction Layer
-> Face Detection Engine (YOLO face)
-> Face Cropping Module
-> Annotation Module
-> Storage Manager
-> Structured Output

## Output Contract

- Annotated frames:
  - output/full_frames/frame_<timestamp>.jpg
- Cropped faces:
  - output/cropped_faces/face_<frameID>_<faceID>.jpg
- Optional metadata (JSONL):
  - output/metadata.jsonl

## Install

Install all required Python dependencies from requirements.txt.

```bash
pip install -r requirements.txt
```

Requirements include CUDA 12.1 PyTorch wheels in requirements.txt. If CUDA is unavailable, the pipeline automatically falls back to CPU when --device auto is used.

## Run

### Webcam

Run live webcam detection, display annotated frames, and save JSONL metadata.

```bash
python app/main.py --source 1 --display --save-metadata
```

### Video File

Process a local video file and save annotated outputs plus metadata.

```bash
python app/main.py --source test.mp4 --save-metadata
```

### Choose Model While Running

Run video processing with the yolov8x preset model.

```bash
python app/main.py --source test.mp4 --model yolov8x --save-metadata
```

Run video processing with the yolov9e preset model.

```bash
python app/main.py --source test.mp4 --model yolov9e --save-metadata
```

Run video processing with a custom weights file path.

```bash
python app/main.py --source test.mp4 --weights weights/yolov9t-face-lindevs.pt --save-metadata
```

### Custom Output Folder

Save all generated outputs into a custom output root directory.

```bash
python app/main.py --source test.mp4 --output-root output_run_01 --save-metadata
```

### Image File

Process a single image and save face detections and metadata.

```bash
python app/main.py --source frames/frame_0.jpg --save-metadata
```

### Frame Sampling + Quality Assessment (Layer 2 + Layer 3)

Extract sampled frames, apply quality filtering, and generate quality reports.

```bash
python app/extract_frames.py --source test.mp4 --sample-fps 10 --save-quality-examples
```

This command produces (under `WORKING/output/frames/`):
- Ordered extracted frame sequence per video: frame_sequences/<video_name>/frames/
- Frame metadata with quality flags: frame_sequences/<video_name>/frame_metadata.jsonl
- Extraction success/failure log: frame_extraction_log.jsonl
- Frame count and sequence readiness summary: frame_extraction_summary.json
- Sampling-rate comparison note: docs/frame_sampling_rate_comparison.md
- Frame-quality checklist sheet: docs/frame_quality_checklist.md
- Good vs bad frame marking report: docs/frame_quality_examples_report.md

## Main Options

- --source: image path, video path, stream URL, or webcam index (default: 1)
- --model: model preset one of yolov8n, yolov8x, yolov9e, yolov9t (default: yolov8n)
- --weights: custom YOLO weights path, overrides --model
- --conf: minimum face confidence threshold (default: 0.35)
- --imgsz: inference image size (default: 320)
- --device: auto, cpu, gpu, cuda, or explicit index like 0 (default: auto)
- --half: enable FP16 inference on CUDA
- --output-root: root output folder (default: WORKING/output/frames/annotated)
- --display: show live annotated window
- --save-metadata: write one JSONL record per frame
- --max-io-workers: storage thread pool size (default: 4)

## Extraction Script Options

- --source: single video path
- --input-dir: directory of videos (batch mode)
- --sample-fps: controlled frame sampling rate (project baseline default: 10)
- --min-seq-len: minimum accepted frames required for temporal modeling (default: 64)
- --blur-threshold: blur rejection threshold (default: 8)
- --dark-threshold: dark-frame rejection threshold (default: 45)
- --bright-threshold: overexposed-frame rejection threshold (default: 220)
- --min-face-area-ratio: minimum image-area ratio for largest face (default: 0.005)
- --compare-rates: sampling-rate comparison list for note generation (default: 5,10,15)
- --save-quality-examples: save one accepted and one rejected example frame per video

Quality rejection categories:
- blur
- dark_frame
- overexposed_frame
- no_face
- face_too_small
- extreme_pose

## Model Weight Behavior

- For preset models, weight files are expected under weights/.
- If the selected file is yolov8n-face-lindevs.pt and it is missing, the detector downloads it automatically.
- For other missing weight files, the app raises a file not found error.

## Metadata Record Example

```json
{
  "frame_id": 123,
  "timestamp": "1713921112345678901",
  "faces": [
    {
      "x": 10,
      "y": 20,
      "w": 90,
      "h": 110,
      "confidence": 0.9123
    }
  ]
}
```
