# DeepFake Video Detection System for Video KYC (rPPG + Quantum ML)

Detection of deepfake videos for video KYC in financial systems using
**rPPG (remote photoplethysmography)** as the primary physiological evidence layer and
**quantum machine learning** as the final decision layer, designed for low-resolution videos.

## Architecture (Two-Stage Pipeline)

```
Input KYC video
  -> Frame separation + quality filtering (Component 1: Implementation/)
  -> Face detection, alignment, low-res preprocessing
  -> rPPG signal extraction + feature engineering (Component 3: friend's pipeline)
  -> Quantum input conditioning + hybrid quantum classifier (Component 2: quantum/)
  -> Decision output: real / fake / uncertain
```

## Repository Layout

| Path | Description |
|---|---|
| `Implementation/` | Component 1 (done) - frame extraction, YOLO face detection, cropping, frame quality assessment |
| `quantum/` | Component 2 (this repo's quantum work) - QAOA feature selection + hybrid Variational Quantum Classifier (PennyLane + PyTorch) + classical baselines |
| `Quantum_ML_Implementation/tools/` | Dataset utilities: video frame extraction, dataset audit, face ROI pipeline |
| `Docs/` | Project docs, report drafts, research papers, work-division checklist |
| `FF++/` | FaceForensics++ dataset (gitignored) |

## Components

### 1. Frame Separation (`Implementation/`)

Modular face pipeline for image/video/webcam sources: frame sampling at a controlled
FPS, face detection (YOLO), cropping, storage, and frame quality assessment
(blur / dark / overexposed / no-face / face-too-small / extreme-pose rules).

```bash
python app/main.py --source test.mp4 --save-metadata
python app/extract_frames.py --source test.mp4 --sample-fps 10 --save-quality-examples
```

### 2. Quantum Model (`quantum/`)

Hybrid classical-quantum decision stage built with PennyLane + PyTorch:

- **QAOA feature selection** - selects the most informative rPPG features using a
  cost Hamiltonian (mutual information weights + correlation redundancy penalty +
  cardinality constraint), optimized with COBYLA
- **Hybrid VQC** - angle-encoded features into a parameterized quantum circuit
  (`StronglyEntanglingLayers`) feeding a small classical head
- **Balanced focal loss** with label smoothing and confidence penalty
- **Evaluation** - accuracy / precision / recall / F1 / AUC-ROC / ECE, KYC-friendly
  decision bins (real / uncertain / fake), ROC and confusion-matrix plots
- **Classical baselines** - MLP, RandomForest, SVM on the same features

```bash
python quantum/run_quantum.py --all
```

The quantum component consumes a fixed 9-feature rPPG vector contract documented in
`quantum/docs/interface_contract.md` (label 1 = real, 0 = fake; values in [0, 1]).
Synthetic placeholder data is used for development until the rPPG pipeline is ready;
swap in real features as `quantum/output/data.npz` and rerun.

### 3. rPPG Pipeline (in progress)

Being developed separately; produces the feature vectors consumed by Component 2.

## Install

```bash
pip install -r Implementation/requirements.txt   # Component 1 (CUDA PyTorch)
pip install pennylane torch numpy scikit-learn scipy matplotlib   # Component 2
```

## Team

- **Jhashank** - rPPG-to-classifier pipeline, quantum model (Component 2)
- **Sumit** - input-to-ROI pipeline (data, frames, face detection)
- **Aswin** - preprocessing policy, evaluation, integration

See `Docs/implementation_work_division_checklist.txt` for the full work breakdown.
