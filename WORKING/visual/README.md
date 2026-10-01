# Visual Feature Extraction (Stage 3)

**Component 3** of the deepfake-verification system under `WORKING/`:
`frame/` (stage 1) → `RPPG/` (stage 2) → `visual/` (this directory, stage 3) → `quantum/` (stage 4).

Extracts **39** visual features from face crops saved by Stage 1:
- 16 deep features (PCA-reduced ResNet50 global average pooling)
- 10 LBP histogram bins (uniform patterns)
- 4 GLCM texture features (contrast, energy, homogeneity, correlation)
- 6 color statistics (mean/std per RGB channel)
- 3 DCT frequency domain features (low/mid/high energy bands)

Fuses with rPPG features (24) to create the **fused feature set (63 features)** used for training.

## Project Structure

```
WORKING/visual/
├── __init__.py
├── extractor.py          # VisualFeatureExtractor + compute_visual_features()
├── features.py           # VisualFeatures dataclass + VISUAL_FEATURE_NAMES
├── pipeline.py           # Batch extraction + fusion + experiment splits CLI
└── README.md
```

## Module Responsibilities

| File | Responsibility |
|------|----------------|
| `extractor.py` | `VisualFeatureExtractor` class (ResNet50 + handcrafted features), `compute_visual_features()` (stage-1 handoff), `compute_visual_features_from_video()` (fallback) |
| `features.py` | `VisualFeatures` dataclass (39 fields), `feature_names()` (fixed order), `VISUAL_FEATURE_NAMES` list |
| `pipeline.py` | `extract_visual_features_dataset()`, `fuse_features()`, `create_experiment_splits()`, CLI entry point |

## Feature Breakdown (39 Total)

| Category | Count | Features |
|----------|-------|----------|
| Deep (ResNet50 GAP → PCA) | 16 | `deep_feat_0` … `deep_feat_15` |
| LBP (uniform, 8 pts, r=1) | 10 | `lbp_uniform_hist_0` … `lbp_uniform_hist_9` |
| Texture (GLCM) | 4 | `texture_contrast`, `texture_energy`, `texture_homogeneity`, `texture_correlation` |
| Color (RGB mean/std) | 6 | `color_mean_r/g/b`, `color_std_r/g/b` |
| Frequency (DCT) | 3 | `freq_low_energy`, `freq_mid_energy`, `freq_high_energy` |

**Feature order is fixed** — must match `VISUAL_FEATURE_NAMES` in `features.py` and the fused order in `quantum/config.py`.

## Pipeline Flow

```
Stage-1 frames (JPEGs) + cropped_faces/
  → VisualFeatureExtractor (ResNet50 + handcrafted)
  → Per-frame features → averaged across frames
  → VisualFeatures dataclass (39-dim vector)
  → CSV: output/visual/visual_features.csv
  → fuse_features() with rPPG CSV → fused_features.csv (63 features)
  → create_experiment_splits() → train/val/test splits for quantum layer
```

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
  --frames-root output/frames/frame_sequences \
  --rppg-csv ../output/rppg/dataset_features.csv \
  --fuse \
  --create-splits
```

Options:
- `--frames-root`: Path to `frame_sequences/` directory (required)
- `--rppg-csv`: Path to rPPG features CSV (for label join + fusion)
- `--max-frames`: Max frames per video (default 100)
- `--backbone`: CNN backbone (default `resnet50`)
- `--device`: `auto`, `cpu`, `cuda` (default `auto`)
- `--deep-dim`: PCA dimension for deep features (default 16)
- `--fuse`: Fuse with rPPG features into `fused_features.csv`
- `--create-splits`: Create train/val/test splits for quantum layer

### Fallback: Extract Directly from Videos

```bash
python visual/pipeline.py \
  --video-root archive/DFDC_Dataset \
  --rppg-csv ../output/rppg/dataset_features.csv \
  --fuse \
  --create-splits
```

## Output Files (in `WORKING/output/visual/`)

| File | Description |
|------|-------------|
| `visual_features.csv` | 39 features + label per video |
| `fused_features.csv` | 63 features (24 rPPG + 39 visual) + label |
| `experiments/fused_split.npz` | Train/val/test arrays for quantum layer |
| `experiments/split_indices.npz` | Split indices for reproducibility |

## Feature Contract

`VISUAL_FEATURE_NAMES` in `features.py` (39 features) must stay identical in name AND order to the visual portion of `FUSED_FEATURE_NAMES` in `quantum/config.py`. The test `test_feature_contract_sync` in `quantum/tests.py` guards this.

## Integration with End-to-End Pipeline

`run_pipeline.py` calls `compute_visual_features()` with:
- `frames_dir`: `output/frames/frame_sequences/<video>/frames/`
- Uses face crops from `output/frames/frame_sequences/<video>/cropped_faces/`
- Falls back to full frames + center crop if crops unavailable

## VisualFeatureExtractor Details

### Deep Features
- Backbone: ResNet50 (ImageNet-1K V2 weights)
- Layer: Global Average Pooling (2048-dim)
- Reduction: PCA to 16 components (fitted on first batch, random_state=42)
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