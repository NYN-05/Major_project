# Quantum ML Component - Run Instructions

This component (Component 2 of the project) implements the final quantum decision
stage: QAOA feature selection on rPPG features + a hybrid Variational Quantum
Classifier (PennyLane + PyTorch), plus classical baselines for comparison.

## Prerequisites

- Python 3.10 interpreter: `C:\Users\JHASHANK\AppData\Local\Programs\Python\Python310\python.exe`
- Dependencies: `pennylane`, `torch`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`

Install if missing:

```powershell
$python = "C:\Users\JHASHANK\AppData\Local\Programs\Python\Python310\python.exe"
& $python -m pip install pennylane torch numpy scikit-learn scipy matplotlib
```

## Run Everything (default: synthetic placeholder data)

```powershell
$python = "C:\Users\JHASHANK\AppData\Local\Programs\Python\Python310\python.exe"
& $python "quantum\run_quantum.py" --all
```

Pipeline steps and their outputs:

| Step                                | Flag            | Outputs                                                                                      |
| ----------------------------------- | --------------- | -------------------------------------------------------------------------------------------- |
| 1. Generate synthetic rPPG dataset  | `--gen-data`  | `quantum/output/data.npz`, `quantum/output/features.csv`, `quantum/output/scaler.json` |
| 2. QAOA feature selection           | `--select`    | `quantum/output/qaoa_selection.json`                                                       |
| 3. Train hybrid VQC                 | `--train`     | `quantum/output/hybrid_vqc.pt`, `quantum/output/training_log.jsonl`                      |
| 4. Evaluate quantum model           | `--evaluate`  | `quantum/output/metrics_quantum.json`, `roc_curve.png`, `confusion_matrix.png`         |
| 5. Classical baselines (MLP/RF/SVM) | `--baselines` | `quantum/output/metrics_baselines.json`                                                    |
| 6. Results report                   | `--report`    | `quantum/docs/results_report.md`                                                           |

Steps can be run individually in order; each step needs the previous ones' outputs.

## Using Real rPPG Features

See `quantum/docs/interface_contract.md` for the exact feature schema (9 features,
[0, 1] range, label 1 = real / 0 = fake, train/val/test splits).

Deliver the file as `quantum/output/data.npz` (preferred) or `quantum/output/features.csv`,
then rerun `--all`.

## Hardware Acceleration (GPU)

- **Classical head on CUDA (automatic):** with CUDA-enabled PyTorch and an NVIDIA GPU,
  the hybrid VQC's classical head and data batches run on the GPU. The PennyLane quantum
  block runs on CPU (`lightning.qubit` cannot take CUDA tensors). Falls back to full CPU
  automatically when no GPU is found. No flags required.
- **Quantum simulator on GPU (Linux/WSL only):** `resolve_device()` in `quantum/config.py`
  automatically picks `lightning.gpu` if `pennylane-lightning-gpu` is installed, else
  `lightning.qubit`. On Linux/WSL:
  `python -m pip install pennylane-lightning-gpu`.
  Note: `pennylane-lightning-gpu` has no Windows wheels, so it cannot run on native
  Windows.

## Tuning

Hyperparameters live in `quantum/config.py`:

- `QAOASelectionConfig`: QAOA layers `p_layers` (default 3), `max_iter`, penalties,
  `target_features` (default 6)
- `VQCConfig`: circuit layers `qml_layers`, epochs, batch size, learning rate,
  focal-loss terms (alpha/gamma/label smoothing/confidence penalty)
- `DecisionConfig`: uncertainty thresholds `fake_max_prob` / `real_min_prob`

## Validation

- The QAOA Hamiltonian is verified against the classical cost function
  (`verify_hamiltonian`), max error ~1e-15.
- Training logs per epoch: train/val loss and accuracy.
- Test metrics: accuracy, precision, recall, F1, AUC-ROC, ECE, confusion matrix,
  real/uncertain/fake decision counts.
