import os
from dataclasses import dataclass, field
from pathlib import Path

LABEL_REAL = 1
LABEL_FAKE = 0


def get_dfdc_dataset_path() -> Path:
    """Get DFDC dataset path from DFDC_DATASET_PATH environment variable.

    Raises:
        RuntimeError: If DFDC_DATASET_PATH is not set or the path does not exist.
    """
    env_path = os.environ.get("DFDC_DATASET_PATH")
    if not env_path:
        raise RuntimeError(
            "DFDC_DATASET_PATH environment variable is not set. "
            "Set it in .env or export it before running. "
            "Example: DFDC_DATASET_PATH=C:\\\\path\\\\to\\\\DFDC_Dataset"
        )
    path = Path(env_path)
    if not path.exists():
        raise RuntimeError(
            f"DFDC dataset path does not exist: {path}. "
            f"Check DFDC_DATASET_PATH in .env"
        )
    return path

# Feature contract: the raw 20-feature vector produced by the rPPG layer
# (same names/order as RPPGFeatures.feature_names() in RPPG/rppg/features.py).
RPPG_FEATURE_NAMES = [
    "heart_rate_bpm",
    "snr_db",
    "prv_std_ms",
    "spectral_entropy",
    "mad",
    "signal_quality_index",
    "cheek_forehead_correlation",
    "left_right_cheek_correlation",
    "hr_half_diff",
    "peak_prominence",
    "systolic_peak_width",
    "diastolic_notch_ratio",
    "forehead_cheek_phase_lag",
    "signal_to_motion_ratio",
    "peak_amplitude_variability",
    "pulse_transit_time_proxy",
    # Probe features (Phase 4 upstream improvement) - 4 selected
    "spectral_flatness",
    "spectral_centroid",
    "kurtosis",
    "phase_coherence_lr",
]

# Visual feature names (from visual/features.py)
VISUAL_FEATURE_NAMES = [
    # Deep features (16 dims)
    "deep_feat_0", "deep_feat_1", "deep_feat_2", "deep_feat_3",
    "deep_feat_4", "deep_feat_5", "deep_feat_6", "deep_feat_7",
    "deep_feat_8", "deep_feat_9", "deep_feat_10", "deep_feat_11",
    "deep_feat_12", "deep_feat_13", "deep_feat_14", "deep_feat_15",
    # LBP histogram (10 dims)
    "lbp_uniform_hist_0", "lbp_uniform_hist_1", "lbp_uniform_hist_2",
    "lbp_uniform_hist_3", "lbp_uniform_hist_4", "lbp_uniform_hist_5",
    "lbp_uniform_hist_6", "lbp_uniform_hist_7", "lbp_uniform_hist_8",
    "lbp_uniform_hist_9",
    # Texture (4 dims)
    "texture_contrast", "texture_energy", "texture_homogeneity",
    "texture_correlation",
    # Color (6 dims)
    "color_mean_r", "color_mean_g", "color_mean_b",
    "color_std_r", "color_std_g", "color_std_b",
    # Frequency (3 dims)
    "freq_low_energy", "freq_mid_energy", "freq_high_energy",
]

# Backward compatibility alias
FEATURE_NAMES = RPPG_FEATURE_NAMES

# Fused feature names (rPPG + Visual) - ONLY SUPPORTED MODE
FUSED_FEATURE_NAMES = RPPG_FEATURE_NAMES + VISUAL_FEATURE_NAMES

# Only fused mode is supported
FEATURE_SET_CONFIGS = {
    "fused": FUSED_FEATURE_NAMES,
}

# Legacy aliases for backward compatibility (deprecated)
RPPG_BASE_FEATURE_NAMES = RPPG_FEATURE_NAMES  # all 20 features
CROSS_ROI_FEATURE_NAMES = []  # cross-ROI features removed
PHASE6_FEATURE_SETS = {}
PHASE7_ABLATION_SETS = {}

FEATURE_MEANINGS = {
    "heart_rate_bpm": "Dominant pulse frequency (BPM) of the recovered rPPG signal",
    "snr_db": "Signal-to-noise ratio of the pulse spectrum (dB)",
    "prv_std_ms": "Pulse rate variability: std of inter-beat intervals (ms)",
    "spectral_entropy": "Shannon entropy of the normalized in-band power spectrum",
    "mad": "Mean absolute deviation of the pulse waveform",
    "signal_quality_index": "Beat-regularity and spectral-concentration quality in [0, 1]",
    "cheek_forehead_correlation": "Pearson correlation between cheek and forehead pulse signals",
    "left_right_cheek_correlation": "Pearson correlation between left and right cheek pulse signals",
    "hr_half_diff": "Absolute difference between first-half and second-half heart rates (BPM)",
    "peak_prominence": "Spectral peak-to-mean ratio of the in-band pulse spectrum",
    "systolic_peak_width": "Median half-height width (ms) of systolic peaks in the pulse waveform",
    "diastolic_notch_ratio": "Mean dicrotic-notch depth relative to systolic peak height",
    "forehead_cheek_phase_lag": "Time lag (ms) between forehead and cheek pulse signals",
    "signal_to_motion_ratio": "Log ratio of physiological-band to motion-band spectral power (dB)",
    "peak_amplitude_variability": "Coefficient of variation of systolic peak amplitudes",
    "pulse_transit_time_proxy": "Inter-ROI propagation delay (ms) as a pulse transit time proxy",
    # Probe features (Phase 4)
    "spectral_flatness": "Geometric/arithmetic mean ratio of in-band PSD (flat=1, tonal<1)",
    "spectral_centroid": "Center of mass of in-band power spectrum (Hz)",
    "kurtosis": "Fisher kurtosis of time-domain pulse waveform (heavy-tailed=artifacts)",
    "phase_coherence_lr": "Phase coherence std between left/right cheek signals (lower=more coherent)",
    "phase_coherence_cf": "Phase coherence std between forehead/cheek signals (lower=more coherent)",
    "pulse_cv_interval": "Coefficient of variation of inter-beat intervals (regular=low CV)",
    "zero_crossing_rate": "Zero-crossing rate of pulse waveform (high=noise/artifacts)",
    # Visual features
    "deep_feat_0": "ResNet50 GAP feature 0 (PCA-reduced)",
    "deep_feat_1": "ResNet50 GAP feature 1 (PCA-reduced)",
    "deep_feat_2": "ResNet50 GAP feature 2 (PCA-reduced)",
    "deep_feat_3": "ResNet50 GAP feature 3 (PCA-reduced)",
    "deep_feat_4": "ResNet50 GAP feature 4 (PCA-reduced)",
    "deep_feat_5": "ResNet50 GAP feature 5 (PCA-reduced)",
    "deep_feat_6": "ResNet50 GAP feature 6 (PCA-reduced)",
    "deep_feat_7": "ResNet50 GAP feature 7 (PCA-reduced)",
    "deep_feat_8": "ResNet50 GAP feature 8 (PCA-reduced)",
    "deep_feat_9": "ResNet50 GAP feature 9 (PCA-reduced)",
    "deep_feat_10": "ResNet50 GAP feature 10 (PCA-reduced)",
    "deep_feat_11": "ResNet50 GAP feature 11 (PCA-reduced)",
    "deep_feat_12": "ResNet50 GAP feature 12 (PCA-reduced)",
    "deep_feat_13": "ResNet50 GAP feature 13 (PCA-reduced)",
    "deep_feat_14": "ResNet50 GAP feature 14 (PCA-reduced)",
    "deep_feat_15": "ResNet50 GAP feature 15 (PCA-reduced)",
    "lbp_uniform_hist_0": "LBP uniform pattern histogram bin 0",
    "lbp_uniform_hist_1": "LBP uniform pattern histogram bin 1",
    "lbp_uniform_hist_2": "LBP uniform pattern histogram bin 2",
    "lbp_uniform_hist_3": "LBP uniform pattern histogram bin 3",
    "lbp_uniform_hist_4": "LBP uniform pattern histogram bin 4",
    "lbp_uniform_hist_5": "LBP uniform pattern histogram bin 5",
    "lbp_uniform_hist_6": "LBP uniform pattern histogram bin 6",
    "lbp_uniform_hist_7": "LBP uniform pattern histogram bin 7",
    "lbp_uniform_hist_8": "LBP uniform pattern histogram bin 8",
    "lbp_uniform_hist_9": "LBP uniform pattern histogram bin 9",
    "texture_contrast": "GLCM contrast",
    "texture_energy": "GLCM energy",
    "texture_homogeneity": "GLCM homogeneity",
    "texture_correlation": "GLCM correlation",
    "color_mean_r": "Mean red channel intensity",
    "color_mean_g": "Mean green channel intensity",
    "color_mean_b": "Mean blue channel intensity",
    "color_std_r": "Std red channel intensity",
    "color_std_g": "Std green channel intensity",
    "color_std_b": "Std blue channel intensity",
    "freq_low_energy": "Low-frequency DCT energy ratio",
    "freq_mid_energy": "Mid-frequency DCT energy ratio",
    "freq_high_energy": "High-frequency DCT energy ratio",
}

QUANTUM_ROOT = Path(__file__).resolve().parent
WORKING_ROOT = QUANTUM_ROOT.parent
OUTPUT_DIR = WORKING_ROOT / "output" / "quantum"


@dataclass(frozen=True)
class DataConfig:
    seed: int = 42
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    filter_implausible: bool = True
    hr_min: float = 30.0
    hr_max: float = 220.0
    csv_file: Path = field(
        default_factory=lambda: WORKING_ROOT / "output" / "rppg" / "dataset_features.csv"
    )
    data_file: Path = field(default_factory=lambda: OUTPUT_DIR / "data.npz")


@dataclass(frozen=True)
class QAOASelectionConfig:
    p_layers: int = 3
    max_iter: int = 500
    redundancy_penalty: float = 0.3
    cardinality_penalty: float = 0.5
    target_features: int = 3
    seed: int = 42
    restarts: int = 8
    n_jobs: int = 0
    # Simulator backend: "auto" uses the torch-native statevector sim
    # (fast, float64 complex128; see qaoa_sim.QAOASimulator); "torch"
    # forces the torch sim; "pennylane" uses the legacy PennyLane QNode
    # path (lightning.qubit on CPU). "lightning"/"default" are accepted
    # as aliases for "pennylane" for backward compat.
    device: str = "auto"
    selection_file: Path = field(default_factory=lambda: OUTPUT_DIR / "qaoa_selection.json")


@dataclass(frozen=True)
class VQCConfig:
    qml_layers: int = 3
    hidden_units: int = 8
    dropout: float = 0.2
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 5e-2
    weight_decay: float = 1e-2
    alpha: float = 0.45
    gamma: float = 1.0
    label_smoothing: float = 0.03
    confidence_penalty: float = 0.0
    lr_schedule: str = "cosine_warmup"
    warmup_epochs: int = 3
    patience: int = 12
    min_delta: float = 1e-4
    clip_grad: float = 1.0
    broadcast_qnode: bool = True
    # Quantum circuit backend: "auto" uses the torch-native layer
    # (fast, complex128, differentiable; see vqc.QuantumLayerTorch);
    # "pennylane" uses the legacy PennyLane QNode path (default.qubit
    # on CPU with backprop). The torch head (weights, loss, optimizer)
    # always runs on resolve_device() regardless of this flag.
    qnode_impl: str = "auto"
    # Legacy field kept for backward compat with saved metadata.
    qnode_backend: str = "auto"
    save_checkpoint: bool = True
    seed: int = 42
    checkpoint_file: Path = field(default_factory=lambda: OUTPUT_DIR / "hybrid_vqc.pt")
    log_file: Path = field(default_factory=lambda: OUTPUT_DIR / "training_log.jsonl")
    metrics_file: Path = field(default_factory=lambda: OUTPUT_DIR / "metrics_quantum.json")


@dataclass(frozen=True)
class DecisionConfig:
    # Three-state decision thresholds
    # prob_real >= real_min_prob -> REAL
    # prob_real <= fake_max_prob -> FAKE
    # fake_max_prob < prob_real < real_min_prob -> INSUFFICIENT EVIDENCE / REVIEW REQUIRED
    fake_max_prob: float = 0.3
    real_min_prob: float = 0.7
    # Quality threshold: minimum PQS for sufficient evidence
    # Below this -> INSUFFICIENT EVIDENCE / REVIEW REQUIRED
    quality_threshold: float = 0.5
    # Legacy single threshold (kept for backward compatibility)
    decision_threshold: float = 0.5
    metrics_baseline_file: Path = field(default_factory=lambda: OUTPUT_DIR / "metrics_baselines.json")
    roc_plot: Path = field(default_factory=lambda: OUTPUT_DIR / "roc_curve.png")
    confusion_plot: Path = field(default_factory=lambda: OUTPUT_DIR / "confusion_matrix.png")
    calibration_plot: Path = field(default_factory=lambda: OUTPUT_DIR / "calibration_curve.png")