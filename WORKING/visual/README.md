# Visual Feature Extraction (Stage 3)

> **Stage 3 of 4:** `frame/` (stage 1) → `RPPG/` (stage 2) → `visual/` (this directory, appearance features) → `quantum/` (stage 4).

Extracts **39** visual features from face crops saved by Stage 1:

| Group                                 | Count |
| ------------------------------------- | ----- |
| Deep features (PCA-reduced ResNet50 global average pooling) | 16 |
| LBP histogram bins (uniform patterns) | 10    |
| GLCM texture features (contrast, energy, homogeneity, correlation) | 4 |
| Color statistics (mean/std per RGB channel) | 6 |
| DCT frequency domain features (low/mid/high energy bands) | 3 |

Fuses with rPPG features (20) to create the **fused feature set (59 features)** used for training.

## Contents

- [Project Structure](#project-structure)
- [Module Responsibilities](#module-responsibilities)
- [Feature Breakdown (39 Total)](#feature-breakdown-39-total)
- [Pipeline Flow](#pipeline-flow)
- [Install](#install)
- [Run](#run)
- [Output Files](#output-files)
- [Feature Contract](#feature-contract)
- [Integration with End-to-End Pipeline](#integration-with-end-to-end-pipeline)
- [VisualFeatureExtractor Details](#visualfeatureextractor-details)
- [Limitations](#limitations)

## Project Structure

```text
WORKING/visual/
├── __init__.py
├── extractor.py          # VisualFeatureExtractor, VisualFeatures, VISUAL_FEATURE_NAMES
├── pipeline.py           # Batch extraction + fusion + experiment splits CLI
└── README.md
```

## Module Responsibilities

| File           | Responsibility |
| -------------- | -------------- |
| `extractor.py` | `VisualFeatureExtractor` class (ResNet50 + handcrafted features), `VisualFeatures` dataclass (39 fields) + `VISUAL_FEATURE_NAMES` (fixed order), `compute_visual_features()` (stage-1 handoff), `iter_video_center_crops()` shared video sampler |
| `pipeline.py`  | `extract_visual_features_dataset()`, `fuse_features()`, `create_experiment_splits()`, CLI entry point |

## Feature Breakdown (39 Total)

| Category                  | Count | Features                                                     |
| ------------------------- | ----- | ------------------------------------------------------------ |
| Deep (ResNet50 GAP → PCA) | 16    | `deep_feat_0` … `deep_feat_15`                               |
| LBP (uniform, 8 pts, r=1) | 10    | `lbp_uniform_hist_0` … `lbp_uniform_hist_9`                  |
| Texture (GLCM)            | 4     | `texture_contrast`, `texture_energy`, `texture_homogeneity`, `texture_correlation` |
| Color (RGB mean/std)      | 6     | `color_mean_r/g/b`, `color_std_r/g/b`                        |
| Frequency (DCT)           | 3     | `freq_low_energy`, `freq_mid_energy`, `freq_high_energy`      |

**Feature order is fixed** — must match `VISUAL_FEATURE_NAMES` in `extractor.py` and the fused order in `quantum/config.py`.

## Pipeline Flow

```text
Stage-1 frames (JPEGs) + cropped_faces/
  → VisualFeatureExtractor (ResNet50 + handcrafted)
  → Shared training PCA (fit once on training videos) → averaged across frames
  → VisualFeatures dataclass (39-dim vector)
  → CSV: MAJ_OUTPUT_ROOT/visual/visual_features.csv
  → fuse_features() with rPPG CSV → MAJ_OUTPUT_ROOT/visual/fused_features.csv (59 features)
  → create_experiment_splits() → metadata-aware grouped train/val/test splits
```

### Split integrity

> All experiment splits use the canonical grouped splitter from
> `quantum.data`; ordinary row/video-level random splitting is not permitted.
> Rows are grouped by the first available `subject_id`, `identity_id`,
> `original_video_id`, `source_video_id`, `source_id`, `group_id`, or content
> hash. When none is present, the normalized clip path is used and the
> manifest records that clip-level fallback explicitly. The split manifest
> stores every path and group and is written before training artifacts are
> consumed. Duplicate paths fail the build, and group overlap across train,
> validation, and test is a hard error.

## Install

```bash
# From WORKING/
pip install torchvision scikit-image
```

## Run

### Extract Visual Features (from Stage-1 frames)

```bash
# From WORKING/
python visual/pipeline.py \
  --frames-root MAJ_OUTPUT_ROOT/frames/frame_sequences \
  --rppg-csv MAJ_OUTPUT_ROOT/rppg/dataset_features.csv \
  --fuse \
  --create-splits
```

| Option            | Description                                              |
| ---------------- | -------------------------------------------------------- |
| `--frames-root`  | Path to `frame_sequences/` directory (required)          |
| `--rppg-csv`     | Path to rPPG features CSV (for label join + fusion)      |
| `--max-frames`   | Max frames per video (default 100)                       |
| `--backbone`     | CNN backbone (default `resnet50`)                        |
| `--device`       | `auto`, `cpu`, `cuda` (default `auto`)                   |
| `--deep-dim`     | PCA dimension for deep features (default 16)             |
| `--fuse`         | Fuse with rPPG features into `fused_features.csv`        |
| `--create-splits`| Create train/val/test splits for quantum layer           |

### Fallback: Extract Directly from Videos

```bash
python visual/pipeline.py \
  --video-root archive/DFDC_Dataset \
  --rppg-csv MAJ_OUTPUT_ROOT/rppg/dataset_features.csv \
  --fuse \
  --create-splits
```

## Output Files

In `MAJ_OUTPUT_ROOT/visual/`:

| File                          | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| `visual_features.csv`          | 39 features + label per video                                |
| `visual_pca.npz`               | Training-only 2048→16 PCA parameters and schema metadata     |
| `fused_features.csv`           | 59 features (20 rPPG + 39 visual) + label                    |
| `experiments/fused_split.npz`  | Train/val/test arrays for quantum layer                      |
| `experiments/split_indices.npz`| Split indices for reproducibility                            |

## Feature Contract

> `VISUAL_FEATURE_NAMES` in `extractor.py` (39 features) must stay identical in name AND order to the visual portion of `FUSED_FEATURE_NAMES` in `quantum/config.py`. The test `test_feature_contract_sync` in `quantum/tests.py` guards this.

## Integration with End-to-End Pipeline

`run_pipeline.py` calls `compute_visual_features()` with:

- `frames_dir`: `MAJ_OUTPUT_ROOT/frames/frame_sequences/<video>/frames/`
- Uses face crops from `MAJ_OUTPUT_ROOT/frames/frame_sequences/<video>/cropped_faces/`
- Falls back to full frames + center crop if crops unavailable
- Loads `MAJ_OUTPUT_ROOT/visual/visual_pca.npz`; inference never fits PCA or falls back to
  raw-dimension truncation. Generate this artifact with the batch extraction command
  before serving inference.

## VisualFeatureExtractor Details

### Deep Features

- Backbone: ResNet50 (ImageNet-1K V2 weights)
- Layer: Global Average Pooling (2048-dim)
- Reduction: PCA to 16 components, fitted exactly once on pooled training-partition
  ResNet50 frame features and reused for validation, test, and inference
- PCA input contract: exactly 2048 ResNet50 GAP features; missing, incompatible, or
  corrupted artifacts fail explicitly
- Device: CUDA when available, else CPU

### Handcrafted Features

- **LBP**: `skimage.feature.local_binary_pattern` (uniform, n_points=8, radius=1) → 10-bin histogram
- **Texture**: GLCM (distances=[1], angles=[0, π/4, π/2, 3π/4], levels=32) → contrast, energy, homogeneity, correlation
- **Color**: Mean/std per RGB channel on face crop (6 values)
- **Frequency**: DCT on 64×64 normalized patch → low (8×8), mid (16×16), high (remaining) energy fractions

### Aggregation

All per-frame features are **averaged across frames** to produce a single 39-dim vector per video.

## Limitations

- Requires Stage-1 face crops for best results (falls back to center crop)
- ResNet50 features may not generalize to unseen deepfake generators
- Handcrafted features sensitive to compression/lighting
- 39 visual features alone don't separate classes well (AUC ~0.55); fusion with rPPG is essential
