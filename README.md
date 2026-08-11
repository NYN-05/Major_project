# DeepFake Video Detection System for Video KYC (rPPG + Quantum ML)

Detection of deepfake videos for video KYC in financial systems using
**rPPG (remote photoplethysmography)** as the primary physiological evidence layer and
**quantum machine learning** as the final decision layer, designed for low-resolution videos.

## Architecture (Two-Stage Pipeline)

```
Input KYC video
  -> Frame separation + quality filtering (WORKING/frame)
  -> Face detection, alignment, low-res preprocessing
  -> rPPG signal extraction + feature engineering (WORKING/RPPG)
  -> Hybrid quantum classifier (WORKING/quantum)
  -> Decision output: real / fake / uncertain
```

The end-to-end flow is orchestrated by `WORKING/run_pipeline.py`:

```
frames  ->  rPPG  ->  quantum  ->  final verdict (REAL / FAKE / UNCERTAIN)
```

## Repository Layout

| Path | Description |
|---|---|
| `WORKING/` | Active project root (all three pipeline stages + end-to-end runner) |
| `WORKING/frame/` | Component 1 - frame extraction, YOLO face detection, cropping, frame quality assessment |
| `WORKING/RPPG/` | Component 2 - rPPG signal extraction + feature engineering, classifier |
| `WORKING/quantum/` | Component 3 - QAOA feature selection + hybrid Variational Quantum Classifier (PennyLane + PyTorch) + classical baselines |
| `WORKING/run_pipeline.py` | End-to-end runner: frames -> rPPG -> quantum -> verdict |
| `WORKING/output/` | Pipeline/quantum artifacts (results, plots, checkpoints) |
| `Docs/` | Project docs, report drafts, research papers |
| `FF++/` | FaceForensics++ dataset (gitignored) |

## Components

### 1. Frame Separation (`WORKING/frame/`)

Modular face pipeline for image/video/webcam sources: frame sampling at a controlled
FPS, face detection (YOLO), cropping, storage, and frame quality assessment
(blur / dark / overexposed / no-face / face-too-small / extreme-pose rules).

```bash
python app/main.py --source test.mp4 --save-metadata
python app/extract_frames.py --source test.mp4 --sample-fps 10 --save-quality-examples
```

### 2. rPPG Pipeline (`WORKING/RPPG/`)

POS/CHROM pulse reconstruction from facial ROIs (left cheek, right cheek, forehead)
and an 8-feature physiological vector per video: heart rate, SNR, PRV, spectral
entropy, MAD, signal quality index, and inter-region correlations. Features are
persisted with labels (1 = fake, 0 = real) in `dataset_features.csv`, which is the
direct data source for the quantum layer.

```bash
python rppg-pipeline/extract_dataset_features.py   # rebuild dataset_features.csv
```

### 3. Quantum Model (`WORKING/quantum/`)

Hybrid classical-quantum decision stage built with PennyLane + PyTorch. It consumes
the rPPG 8-feature vector **directly** (same names/order as `RPPGFeatures.feature_names()`);
no synthetic data is generated anywhere in the pipeline.

- **Data** - `data.py` builds `output/data.npz` from the real labeled rPPG feature
  table `RPPG/dataset_features.csv` (rPPG label 1 = fake is flipped to the quantum
  convention LABEL_REAL = 1, LABEL_FAKE = 0) with a stratified train/val/test split
- **QAOA feature selection** - selects the most informative rPPG features using a
  cost Hamiltonian (mutual information weights + correlation redundancy penalty +
  cardinality constraint), optimized with COBYLA
- **Hybrid VQC** - angle-encoded features into a parameterized quantum circuit
  (`StronglyEntanglingLayers`) feeding a small classical head
- **Balanced focal loss** with label smoothing and confidence penalty
- **Evaluation** - accuracy / precision / recall / F1 / AUC-ROC / ECE, KYC-friendly
  decision bins (real / uncertain / fake), ROC and confusion-matrix plots
- **Classical baselines** - MLP, RandomForest on the same features

```bash
# From WORKING/: build data, QAOA selection, train VQC, evaluate, baselines
python -m quantum.run --all
```

## Install

```bash
pip install -r WORKING/frame/requirements.txt     # Component 1 (CUDA PyTorch)
pip install -r WORKING/RPPG/requirements.txt      # Component 2 (rPPG / MediaPipe)
pip install pennylane torch numpy scikit-learn scipy matplotlib   # Component 3
```

## Run End-to-End

```bash
# From WORKING/
python run_pipeline.py --source path/to/video.mp4 --method POS --out result.json
```

See `Docs/` and the PDFs at the repo root for the full system documentation and
work-division checklist.

## Team

- **Jhashank** - rPPG-to-classifier pipeline, quantum model (Component 3)
- **Sumit** - input-to-ROI pipeline (data, frames, face detection)
- **Aswin** - preprocessing policy, evaluation, integration