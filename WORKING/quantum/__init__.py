"""Quantum decision layer (stage 3 of the deepfake-verification pipeline).

QAOA feature selection + a hybrid quantum-classical VQC (PennyLane + PyTorch)
over the real rPPG feature table, producing the final KYC verdict
(REAL / FAKE / UNCERTAIN).

Modules
-------
config      feature contract, label conventions, dataclass configs, artifact paths
data        build/load data.npz (label flip, plausibility filter, grouped split)
scaling     train-only z-score FeatureScaler (JSON artifact)
qaoa        QAOA feature selection + classical MI-greedy reference
vqc         HybridModel, focal loss, CUDA-aware train / predict / load (cached)
evaluation  metrics, decision bins, cross-validation, baselines
plots       ROC / confusion / calibration figure helpers
pipeline    CLI entry (``python -m quantum.pipeline``) + ``predict_features()``

Run the full flow from ``WORKING/`` with ``python -m quantum.pipeline --all``.
"""