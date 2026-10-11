"""Quantum decision layer (stage 3 of the deepfake-verification pipeline).

QAOA feature selection + a hybrid quantum-classical VQC (PennyLane + PyTorch)
over the real rPPG feature table, producing the final KYC verdict
(REAL / FAKE).

Modules
-------
config      feature contract, label conventions, PipelineStatus, artifact
            integrity helpers (sha256_file / verify_sha256 / write_sha256 /
            schema_digest), dataclass configs, output paths
data        build/load data_fused.npz (label flip, plausibility filter, grouped split)
scaling     train-only z-score FeatureScaler (JSON artifact) +
            validate_feature_schema / transform_features
qaoa        QAOA feature selection + classical MI-greedy reference
qaoa_sim    exact torch-native statevector QAOA simulator
vqc         HybridModel, focal loss, CUDA-aware train / predict / load (cached)
evaluation  metrics, decision bins, cross-validation, baselines,
            threshold selection, ROC / confusion / calibration figures
pipeline    CLI entry (``python -m quantum.pipeline``) + ``predict_features()``
explain     explainable output generation (Phase 11)
operations  device resolution, hashing, experiment records (shared with visual)
tests       standalone contract/regression suite (``python -m quantum.tests``)

Run the full flow from ``WORKING/`` with ``python -m quantum.pipeline --all``.
"""
