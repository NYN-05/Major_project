from dataclasses import dataclass, field
from pathlib import Path

LABEL_REAL = 1
LABEL_FAKE = 0

# Feature contract: the raw 8-feature vector produced by the rPPG layer
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
}

QUANTUM_ROOT = Path(__file__).resolve().parent
WORKING_ROOT = QUANTUM_ROOT.parent
OUTPUT_DIR = WORKING_ROOT / "output" / "quantum"
DOCS_DIR = OUTPUT_DIR


@dataclass(frozen=True)
class DataConfig:
    seed: int = 42
    train_ratio: float = 0.6
    val_ratio: float = 0.2
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
    target_features: int = 6
    seed: int = 42
    selection_file: Path = field(default_factory=lambda: OUTPUT_DIR / "qaoa_selection.json")


@dataclass(frozen=True)
class VQCConfig:
    qml_layers: int = 3
    hidden_units: int = 8
    dropout: float = 0.2
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-2
    weight_decay: float = 1e-2
    alpha: float = 0.75
    gamma: float = 2.0
    label_smoothing: float = 0.1
    confidence_penalty: float = 0.3
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
    report_file: Path = field(default_factory=lambda: DOCS_DIR / "results_report.md")