from dataclasses import dataclass, field
from pathlib import Path

LABEL_REAL = 1
LABEL_FAKE = 0

# Feature contract: the raw 20-feature vector produced by the rPPG layer
# (same names/order as RPPGFeatures.feature_names() in RPPG/rppg/features.py).
FEATURE_NAMES = [
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
    "hr_window_std",
    "sqi_window_std",
    "entropy_window_std",
    "max_hr_deviation_bpm",
]

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
    "hr_window_std": "Std of per-window heart rate estimates (BPM) - pulse stability across the clip",
    "sqi_window_std": "Std of per-window signal quality index - pulse-quality stability",
    "entropy_window_std": "Std of per-window spectral entropy - spectral stability",
    "max_hr_deviation_bpm": "Max per-window HR deviation from the median HR (BPM)",
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
    max_iter: int = 200
    redundancy_penalty: float = 0.3
    cardinality_penalty: float = 0.5
    target_features: int = 3
    seed: int = 42
    restarts: int = 4
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
    # Batch 32 with <=32 train rows = a single batch per epoch (benign at this scale).
    batch_size: int = 32
    learning_rate: float = 1e-2
    weight_decay: float = 1e-2
    alpha: float = 0.75
    gamma: float = 2.0
    label_smoothing: float = 0.03
    confidence_penalty: float = 0.02
    lr_schedule: str = "cosine"
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
    fake_max_prob: float = 0.3
    real_min_prob: float = 0.7
    metrics_baseline_file: Path = field(default_factory=lambda: OUTPUT_DIR / "metrics_baselines.json")
    roc_plot: Path = field(default_factory=lambda: OUTPUT_DIR / "roc_curve.png")
    confusion_plot: Path = field(default_factory=lambda: OUTPUT_DIR / "confusion_matrix.png")
    calibration_plot: Path = field(default_factory=lambda: OUTPUT_DIR / "calibration_curve.png")