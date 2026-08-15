# rPPG Physiological Feature Extraction (Stage 2)

**Component 2** of the deepfake-verification system under `WORKING/`:
`frame` (stage 1) -> `RPPG/` (this directory, stage 2) -> `quantum/` (stage 3).

Extracts 10 physiological features from facial video via remote photoplethysmography (rPPG).
The feature table `output/rppg/dataset_features.csv` is the **direct data source** for the quantum
decision layer (`WORKING/quantum/`): it consumes the 10 rPPG features as-is (no synthetic data),
flips the label convention (CSV 1 = fake -> quantum 0 = fake), and builds its
training/eval splits from this table.

## Project Structure

```
WORKING/RPPG/
├── rppg/                      # Core rPPG library
│   ├── __init__.py
│   ├── face_roi.py           # MediaPipe face landmarks -> ROI extraction (cheeks, forehead)
│   ├── pipeline.py           # RPPGPipeline: video/frames -> features + signals
│   ├── signal_extraction.py  # POS/CHROM pulse reconstruction (vectorized)
│   ├── preprocessing.py      # Detrend, bandpass, normalize (cached filters)
│   ├── features.py           # 10-feature computation (HR, SNR, PRV, entropy, MAD, SQI, correlations, HR half-diff, peak prominence)
│   └── model_utils.py        # MediaPipe Face Landmarker + Haar cascade download/caching (SHA-256 verified)
├── rppg-pipeline/            # Training & demo scripts
│   ├── extract_dataset_features.py  # Build dataset_features.csv from archive/ (+ FF++ via --include-ffpp)
│   ├── probe_features.py            # Per-feature AUC probe on a dataset (signal-level study)
│   ├── train_classifier.py         # Train rPPG RandomForest (side path)
│   ├── run_on_video.py             # Single video inference
│   ├── batch_run.py                # Batch processing
│   ├── run_live_webcam.py          # Webcam demo
│   ├── streamlit_app.py            # Streamlit UI
│   ├── debug_run.py                # Debug utilities
│   ├── _check_video.py             # Video validation
│   ├── _make_test_video.py         # Synthetic test video generator
│   └── retrain_dfdc.py             # DFDC retraining script
├── requirements.txt
└── README.md
```

## Core Pipeline (`rppg/pipeline.py`)

### `RPPGPipeline` Class

```python
pipeline = RPPGPipeline(
    method="POS",              # "POS" or "CHROM"
    target_fps=None,           # Resample to this FPS (None = native)
    blur_threshold=15.0,       # Laplacian variance gate
    brightness_range=(25, 230),# Gray mean gate
    low_hz=0.7, high_hz=4.0,   # Physiological band (Hz)
    min_usable_frames=48,      # Minimum frames for feature extraction
    roi_weights=(0.35, 0.35, 0.30)  # Left cheek, right cheek, forehead
)
```

### Two Entry Points

| Method | Input | Use Case |
|--------|-------|----------|
| `process_video(video_path)` | Raw video file | Standalone scripts, direct video read |
| `process_frames(frames_dir, metadata_path, fps)` | Stage-1 accepted frames + JSONL | **Called by `run_pipeline.py`** (stage-1 handoff) |

**Stage-1 handoff path** (used by `run_pipeline.py`):
- Receives accepted JPEGs from `output/frames/frame_sequences/<video>/frames/`
- Receives `frame_metadata.jsonl` with per-frame timestamps
- Runs at stage-1 sample rate (30 fps)
- **Skips blur/brightness re-gating** (stage 1 already did this)
- Still runs MediaPipe per frame; frames with no face → interpolated

**Fallback**: If stage-1 frames unavailable, falls back to `process_video()`.

### Output: `RPPGResult` Dataclass

| Field | Type | Description |
|-------|------|-------------|
| `fps` | float | Effective sampling rate |
| `n_frames_total` | int | Total frames processed |
| `n_frames_usable` | int | Frames with face detected |
| `features` | `RPPGFeatures \| None` | **10-feature vector (or None if < 48 usable frames / degenerate signal)** |
| `combined_signal` | `np.ndarray \| None` | Cleaned combined pulse waveform |
| `left_cheek_signal` | `np.ndarray \| None` | Cleaned left cheek pulse |
| `right_cheek_signal` | `np.ndarray \| None` | Cleaned right cheek pulse |
| `forehead_signal` | `np.ndarray \| None` | Cleaned forehead pulse |
| `quality_log` | `List[FrameQuality]` | Per-frame quality records |
| `warnings` | `List[str]` | Runtime warnings |

### 10 Physiological Features (`rppg/features.py`)

| Feature | Description | Units/Range |
|---------|-------------|-------------|
| `heart_rate_bpm` | Dominant pulse frequency from PSD peak | ~40-180 BPM |
| `snr_db` | Signal-to-noise ratio (HR band vs rest) | dB (higher = cleaner) |
| `prv_std_ms` | Pulse rate variability (std of inter-beat intervals) | ms |
| `spectral_entropy` | Normalized Shannon entropy of in-band PSD | 0-1 (lower = more periodic) |
| `mad` | Mean absolute deviation of waveform | Amplitude units |
| `signal_quality_index` | Beat regularity + spectral concentration | 0-1 |
| `cheek_forehead_correlation` | Pearson r between cheek & forehead pulses | -1 to 1 |
| `left_right_cheek_correlation` | Pearson r between left & right cheek pulses | -1 to 1 |
| `hr_half_diff` | Abs. HR difference between first/second halves of the pulse (stability marker) | BPM (0 = stable) |
| `peak_prominence` | Spectral peak-to-mean ratio of the in-band PSD (dominance of the pulse peak) | ratio (>>1 = strong pulse) |

**Feature order is fixed** - must match `FEATURE_NAMES` in `quantum/config.py` and `RPPGFeatures.feature_names()`.

### Failure Mode

Returns `features=None` when `n_frames_usable < min_usable_frames` (48) **or** when the
raw features are degenerate (≥ 2 NaN values or `signal_quality_index == 0.0` — a flat/dead
pulse). Degenerate signals are rejected at the pipeline level instead of being filled with
"average human" constants (H1). `run_pipeline.py` then emits `INCONCLUSIVE` verdict and
exits with code 3.

## Data Generation

### Build Training Feature Table

```bash
# From WORKING/RPPG/
python rppg-pipeline/extract_dataset_features.py \
  --workers 0            # 0 = all CPU cores (~12 vid/s on this host)
  --target-fps 10        # stage-1 sample rate the pipeline is designed for
  --min-usable-frames 30 # training-table gate (inference keeps 48)
  --method POS

# Optional flags
#   --include-ffpp    include FaceForensics++ clips (FF-synthesis fakes, FF-real/YouTube-real reals)
#   --max-per-class N cap samples per class (smoke test)
#   --output <path>   redirect the output CSV (probe runs)
```

Reads videos from (see `collect_samples()`):
- `archive/DFDC_Dataset/Fake/` (DFDC phone-style face-swaps)
- `archive/DFDC_Dataset/Real/`
- `FF++/` at the repo root, only with `--include-ffpp`
- legacy `archive (1)` CSV layout

MediaPipe/TFLite C++ stderr logging is silenced in worker processes, so the
progress bar stays readable during bulk extraction.

Writes: `WORKING/output/rppg/dataset_features.csv` (relative to `WORKING/`)

### Featurability Probe (`probe_features.py`)

Signal-level study of how much class signal each of the 10 features actually
carries on a dataset — no classifier involved. Emits per-feature AUC (real vs
fake) per method (POS/CHROM) per target fps, plus an all-10-feature oracle AUC,
and a table ranking features.

```bash
python rppg-pipeline/probe_features.py --ffpp --max-per-class 120 --workers 8
```

### Dataset Findings (Validated Aug 2026)

- **Current training table** (`output/rppg/dataset_features.csv`): **16 labeled DFDC clips
  (9 real / 7 fake)** — `archive/DFDC_Dataset/Real|Fake`, DFDC archive only. This is the
  **direct, exclusive data source for the quantum layer**; FF++ is NOT used (gitignored,
  `--include-ffpp` off by default).
- **Quality caveats**: all 16 rows carry **negative SNR** (−1.4 to −8.1 dB) — the pulse is
  buried in noise across the whole set; HR features lie on a coarse ~24.3 BPM grid
  (short compressed clips + coarse PSD binning at 10 fps). With a 10-row train set, all
  metrics are indicative only.
- Historical probe runs (2,797 DFDC preview clips) showed rPPG features at **chance level**
  (AUC 0.47–0.53) — face-swap fakes transplant the source person's genuine skin/vascular
  signal. That probe motivated the current small, honest DFDC table; it does not change
  the fact that the trained model consumes exactly the 16 rows above.

### Train rPPG Classifier (Side Path - Not Used for Final Verdict)

```bash
python rppg-pipeline/train_classifier.py
```

Saves to `WORKING/output/rppg/`:
- `rppg_classifier.pkl` - RandomForest model
- `rppg_classifier_metadata.json` - Training summary, column metadata

**Label convention**: CSV uses `1 = fake, 0 = real`. This is flipped in quantum layer.

## Run Inference

### Single Video (Direct Video Read)

```bash
python rppg-pipeline/run_on_video.py path/to/video.mp4 --method POS --plot
```

### Batch Processing

```bash
python rppg-pipeline/batch_run.py
```

### Webcam Demo

```bash
python rppg-pipeline/run_live_webcam.py
```

### Streamlit App

```bash
streamlit run rppg-pipeline/streamlit_app.py
```

## Key Optimizations (Completed)

- **POS vectorization** (`signal_extraction.py`): `sliding_window_view` + batched matmul + `np.add.at` overlap-add — bit-identical to original loop, removes Python-level window iteration
- **Shared Welch PSD** (`features.py`): Single periodogram for HR, SNR, entropy, SQI — 3 fewer `scipy.signal.welch` calls per video
- **Cached filter/detrend** (`preprocessing.py`): `@lru_cache` on Butterworth coefficients + detrend sparse matrix
- **Skin-mask hoist** (`face_roi.py`): Compute YCrCb+inRange once/frame instead of 3×
- **Discarded quality work** (`pipeline.py`): Stage-1 path still computes Laplacian/brightness (recorded in metadata) but the rPPG gate uses only `face.found`

## Install

```bash
pip install -r requirements.txt
```

### First-Run Face Model

Uses MediaPipe's Face Landmarker Tasks API. On first run it downloads a small model file automatically and caches it locally. Internet required on first run.

If no internet access, place the model manually or provide a custom path through `FaceROIExtractor`.

## MediaPipe Face Landmarker

The `FaceROIExtractor` in `face_roi.py` uses MediaPipe to detect 468 facial landmarks, then extracts three ROIs:
- **Left cheek**: Landmarks around left cheek region
- **Right cheek**: Landmarks around right cheek region
- **Forehead**: Landmarks above eyebrows

Per-ROI mean RGB traces are accumulated, then fed to POS/CHROM for pulse reconstruction.

## Signal Extraction Methods

### POS (Plane-Orthogonal-to-Skin)
- Projects RGB into a plane orthogonal to the skin-tone direction
- More robust to illumination changes
- Default method

### CHROM (Chrominance-based)
- Uses chrominance ratio (R-G, R-B) for pulse extraction
- Alternative when POS struggles

Both implemented in `signal_extraction.py` with vectorized sliding-window processing.

## Feature Computation Details

All spectral features share **one Welch periodogram** per video:
- `heart_rate_bpm`: Peak frequency in 0.7-4.0 Hz band × 60
- `snr_db`: Power at HR fundamental + 1st harmonic vs rest of band
- `spectral_entropy`: Normalized Shannon entropy of in-band PSD
- `signal_quality_index`: Beat regularity (peak spacing) + spectral concentration

Time-domain features:
- `prv_std_ms`: Std of inter-beat intervals from peak detection
- `mad`: Mean absolute deviation of cleaned waveform
- Correlations: Pearson r between per-ROI cleaned signals

NaN features are filled with physiologically neutral fallbacks (HR=72, SNR=0, etc.)
**only for the rare single-NaN case** — degenerate signals (≥ 2 NaN or zero SQI) are
rejected as `features=None` before the fallback runs, and `estimate_snr` returns NaN
(not `-inf`) for zero-power signals so NaN is the single missing-value convention.

## Output Files (in `WORKING/output/rppg/`)

| File | Description |
|------|-------------|
| `dataset_features.csv` | 10 features + label per video (1=fake, 0=real) |
| `rppg_classifier.pkl` | Trained RandomForest (side path) |
| `rppg_classifier_metadata.json` | Training summary, feature columns, metrics |
| `plots/` | Diagnostic plots (PSD, waveforms, feature distributions) |

## Validation Strategy

Before using as research result, validate rPPG stage on ground-truth datasets:

- **UBFC-rPPG**: https://sites.google.com/view/ybenezeth/ubfcrppg
- **PURE**: https://www.tu-ilmenau.de/.../pulse-rate-detection-dataset-pure

Measure heart-rate error against contact-PPG reference; report MAE/RMSE.

## Limitations

- Strong occlusion, masks, or very poor lighting → unreliable features
- Very low frame rates reduce usable heart-rate range
- Heavy compression distorts skin-tone cues, weakens rPPG
- Classifier only as good as extracted features and training data quality
- The training table is only 16 rows (9 real / 7 fake, all negative-SNR DFDC clips) —
  metrics are smoke tests, not deployment guarantees; QAOA mutual-information weights
  go degenerate on tiny tables (data-quantity issue, not a code bug)

## Recommended Workflow

1. Add new videos to `archive/DFDC_Dataset/Fake|Real` (the current quantum-relevant data source)
2. Probe the dataset first: `python rppg-pipeline/probe_features.py --max-per-class N --workers 8`
3. Regenerate features: `python rppg-pipeline/extract_dataset_features.py --workers 0 --target-fps 10 --min-usable-frames 30`
4. Retrain classifier (optional): `python rppg-pipeline/train_classifier.py`
5. Rerun quantum pipeline: `python -m quantum.pipeline --all` (from `WORKING/`)
6. Test end-to-end: `python run_pipeline.py --source <video> --method POS`