# Quantum ML Results Report

## Models Compared

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Hybrid VQC (QAOA-selected features) | 0.8938 | 0.8621 | 0.9375 | 0.8982 | 0.9716 |
| MLP | 0.9375 | 0.9167 | 0.9625 | 0.9390 | 0.9908 |
| RandomForest | 0.9563 | 0.9620 | 0.9500 | 0.9560 | 0.9945 |
| SVM | 0.9500 | 0.9500 | 0.9500 | 0.9500 | 0.9905 |

## QAOA Feature Selection

Selected features: temporal_consistency, signal_stability, amplitude_reliability, rhythm_quality, frame_quality_score, temporal_coverage_ratio

Expected-cost energy after optimization: -0.9727064568807694

## Confusion Matrix (Quantum Classifier)

| | Predicted Fake | Predicted Real |
|---|---:|---:|
| Actual Fake | 68 | 12 |
| Actual Real | 5 | 75 |

## Calibration and Decisions

- Expected Calibration Error (ECE): 0.1720
- Decision thresholds: fake if p <= 0.3, real if p >= 0.7, otherwise uncertain
- Test decisions -> real: 76, uncertain: 84, fake: 0

## Plots

- ROC curve: `C:\Users\JHASHANK\Desktop\Maj_Proj\quantum\output\roc_curve.png`
- Confusion matrix: `C:\Users\JHASHANK\Desktop\Maj_Proj\quantum\output\confusion_matrix.png`

## Notes

- The quantum model is a hybrid Variational Quantum Classifier (VQC) trained on QAOA-selected rPPG features.
- Results are on synthetic placeholder features; swap in the real rPPG feature file and rerun `run_quantum.py --all`.