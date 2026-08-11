from dataclasses import dataclass, field
from pathlib import Path

LABEL_REAL = 1
LABEL_FAKE = 0

FEATURE_NAMES = [
    "temporal_consistency",
    "inter_region_agreement",
    "signal_stability",
    "amplitude_reliability",
    "rhythm_quality",
    "sync_behavior",
    "frame_quality_score",
    "roi_stability",
    "temporal_coverage_ratio",
]

FEATURE_MEANINGS = {
    "temporal_consistency": "Coherence of the recovered pulse signal across the sampled sequence",
    "inter_region_agreement": "Agreement between left cheek, right cheek, and forehead signals",
    "signal_stability": "Waveform stability of the rPPG signal over time",
    "amplitude_reliability": "Strength and reliability of the pulse amplitude",
    "rhythm_quality": "Periodicity and biological plausibility of the rhythm",
    "sync_behavior": "Synchronization behavior across facial regions",
    "frame_quality_score": "Mean frame quality score from the extraction pipeline",
    "roi_stability": "ROI consistency across the frame sequence",
    "temporal_coverage_ratio": "Fraction of sampled frames accepted after quality filtering",
}

QUANTUM_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = QUANTUM_ROOT / "output"
DOCS_DIR = QUANTUM_ROOT / "docs"


@dataclass(frozen=True)
class DataConfig:
    seed: int = 42
    train_per_class: int = 400
    val_per_class: int = 80
    test_per_class: int = 80
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