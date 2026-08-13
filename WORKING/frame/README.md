# Frame Sampling & Quality Layer (Stage 1)

**Component 1** of the deepfake-verification system under `WORKING/`:
`frame` (this directory, stage 1) -> `RPPG/` (stage 2) -> `quantum/` (stage 3).

It is invoked automatically by `WORKING/run_pipeline.py` (stage 1: frame sampling
+ quality assessment feeding the rPPG and quantum stages).

## Project Structure

```
WORKING/frame/
├── app/
│   ├── config.py           # Model presets, runtime configuration
│   ├── detector.py         # YOLO face detection wrapper
│   ├── pipeline.py         # Single CLI: face pipeline AND frame extraction (was app/main.py)
│   └── processing.py       # FrameIngestor, FrameQualityAssessor, StorageManager
├── weights/
│   └── yolov8n-face-lindevs.pt   # Auto-downloaded if missing
├── requirements.txt
└── README.md
```

## Module Responsibilities

| File | Responsibility |
|------|----------------|
| `app/pipeline.py` | **Single CLI entry** - `run_frame_sampling_quality_layer()` (used by `run_pipeline.py`) plus the standalone face pipeline and frame-extraction modes (the legacy `app/main.py` / `app/extract_frames.py`) |
| `app/detector.py` | `FaceDetector` - loads YOLO weights, runs inference, returns face boxes + confidences |
| `app/processing.py` | `FrameIngestor`, `FrameQualityAssessor`, `StorageManager` implementations |
| `app/config.py` | Model preset mapping (`MODEL_WEIGHTS`), `build_config()` for normalized paths |

## Pipeline Flow

```
Input video
  -> FrameIngestor (sample at --sample-fps, default 10 FPS)
  -> FaceDetector (YOLO face detection)
  -> FrameQualityAssessor (blur/dark/bright/face-size/pose gates)
  -> Accepted frames saved as JPEGs to output/frames/frame_sequences/<video>/frames/
  -> Per-frame metadata JSONL to output/frames/frame_sequences/<video>/frame_metadata.jsonl
  -> Summary JSON to output/frames/frame_extraction_summary.json
```

## Output Contract (for Stage 2 rPPG)

| Artifact | Path | Description |
|----------|------|-------------|
| **Accepted frames** | `output/frames/frame_sequences/<video>/frames/frame_XXXXXX_tYYYYYYYYYY.jpg` | Sampled frames passing all quality gates |
| **Frame metadata** | `output/frames/frame_sequences/<video>/frame_metadata.jsonl` | Per-frame: `frame_id`, `source_frame_id`, `timestamp_ms`, `quality` (accepted, score, rejection_reasons, blur_score, brightness_score, face_confidence, max_face_area_ratio, face_too_small, extreme_pose) |
| **Extraction log** | `output/frames/frame_extraction_log.jsonl` | Per-video success/failure + summary |
| **Summary** | `output/frames/frame_extraction_summary.json` | Aggregate: `sampled_frames`, `accepted_frames`, `rejected_frames`, `rejections` (by category), `rppg_checks` (face resolution, color variation), `rppg_input_ready` boolean |
| **Docs** | `output/frames/docs/` | Sampling rate comparison, quality checklist, examples report |

## Quality Rejection Categories

| Category | Condition |
|----------|-----------|
| `blur` | Laplacian variance < `--blur-threshold` (default 8.0) |
| `dark_frame` | Gray mean < `--dark-threshold` (default 45.0) |
| `overexposed_frame` | Gray mean > `--bright-threshold` (default 220.0) |
| `no_face` | YOLO detects no face above `--conf` (default 0.35) |
| `face_too_small` | Largest face area ratio < `--min-face-area-ratio` (default 0.005) |
| `extreme_pose` | Face touches frame edge or unusual aspect ratio |

## Install

```bash
pip install -r requirements.txt
```

Requires CUDA 12.1 PyTorch wheels (in requirements.txt). Falls back to CPU when `--device auto` is used and CUDA unavailable.

## Run

### Via End-to-End Pipeline (Recommended)

```bash
# From WORKING/
python run_pipeline.py --source path/to/video.mp4 --method POS
```

This automatically runs Stage 1 (frame sampling + quality) and passes accepted frames to Stage 2.

### Standalone Frame Stage

```bash
# From WORKING/frame/
python app/pipeline.py --source test.mp4 --save-metadata
```

Or extraction-only mode (same entry point; extraction flags switch the mode):

```bash
python app/pipeline.py --source test.mp4 --sample-fps 10 --save-quality-examples
```

### Main Options (app/pipeline.py — face pipeline mode)

| Option | Description | Default |
|--------|-------------|---------|
| `--source` | Video path, stream URL, or webcam index | `1` |
| `--model` | YOLO preset: `yolov8n`, `yolov8x`, `yolov9e`, `yolov9t` | `yolov8n` |
| `--weights` | Custom weights path (overrides `--model`) | - |
| `--conf` | Minimum face confidence threshold | `0.35` |
| `--imgsz` | Inference image size | `320` |
| `--device` | `auto`, `cpu`, `gpu`, `cuda`, or GPU index | `auto` |
| `--half` | Enable FP16 on CUDA | off |
| `--output-root` | Root output folder | `WORKING/output/frames/annotated` |
| `--display` | Show live annotated window | off |
| `--save-metadata` | Write JSONL metadata per frame | off |
| `--max-io-workers` | Storage thread pool size | `4` |

### Extraction Script Options (app/pipeline.py — extraction mode)

| Option | Description | Default |
|--------|-------------|---------|
| `--source` | Single video path | required |
| `--input-dir` | Directory of videos (batch mode) | - |
| `--sample-fps` | Controlled frame sampling rate | `10.0` |
| `--min-seq-len` | Minimum accepted frames for temporal modeling | `64` |
| `--blur-threshold` | Blur rejection threshold | `8.0` |
| `--dark-threshold` | Dark-frame rejection threshold | `45.0` |
| `--bright-threshold` | Overexposed-frame rejection threshold | `220.0` |
| `--min-face-area-ratio` | Minimum face area ratio | `0.005` |
| `--compare-rates` | Sampling-rate comparison list for docs | `5,10,15` |
| `--save-quality-examples` | Save accepted/rejected example frames | off |

## Model Weight Behavior

- For preset models, weight files are expected under `weights/`.
- If `yolov8n-face-lindevs.pt` is missing, the detector downloads it automatically.
- For other missing weight files, the app raises `FileNotFoundError`.

## Metadata Record Example

```json
{
  "video": "test.mp4",
  "frame_id": 123,
  "source_frame_id": 1230,
  "timestamp_ms": 12345,
  "quality": {
    "accepted": true,
    "quality_flag": "accepted",
    "score": 0.8542,
    "rejection_reasons": [],
    "blur_score": 42.17,
    "brightness_score": 128.3,
    "face_detected": true,
    "face_confidence": 0.9123,
    "max_face_area_ratio": 0.12,
    "face_too_small": false,
    "extreme_pose": false
  },
  "saved_frame": "frame_000123_t0000012345.jpg"
}
```