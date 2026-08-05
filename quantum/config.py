from dataclasses import dataclass, field
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUANTUM_ROOT = Path(__file__).resolve().parents[0]
OUTPUT_DIR = QUANTUM_ROOT / "output"
DOCS_DIR = QUANTUM_ROOT / "docs"


def resolve_device(preferred: str = "lightning.gpu", fallback: str = "lightning.qubit") -> str:
    try:
        import pennylane as qml

        qml.device(preferred, wires=2)
        return preferred
    except Exception:
        return fallback


def default_num_workers() -> int:
    return max(1, min(12, int((os.cpu_count() or 4) * 0.9)))


@dataclass(frozen=True)
class DataConfig:
    seed: int = 42
    train_per_class: int = 400
    val_per_class: int = 80
    test_per_class: int = 80
    data_file: Path = field(default_factory=lambda: OUTPUT_DIR / "data.npz")
    csv_file: Path = field(default_factory=lambda: OUTPUT_DIR / "features.csv")
    scaler_file: Path = field(default_factory=lambda: OUTPUT_DIR / "scaler.json")


@dataclass(frozen=True)
class QAOASelectionConfig:
    p_layers: int = 3
    max_iter: int = 200
    redundancy_penalty: float = 0.3
    cardinality_penalty: float = 0.5
    target_features: int = 6
    device: str = field(default_factory=resolve_device)
    selection_file: Path = field(default_factory=lambda: OUTPUT_DIR / "qaoa_selection.json")


@dataclass(frozen=True)
class VQCConfig:
    qml_layers: int = 3
    hidden_units: int = 8
    dropout: float = 0.2
    device: str = field(default_factory=resolve_device)
    num_workers: int = field(default_factory=default_num_workers)
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-2
    weight_decay: float = 1e-2
    alpha: float = 0.75
    gamma: float = 2.0
    label_smoothing: float = 0.1
    confidence_penalty: float = 0.3
    checkpoint_file: Path = field(default_factory=lambda: OUTPUT_DIR / "hybrid_vqc.pt")
    log_file: Path = field(default_factory=lambda: OUTPUT_DIR / "training_log.jsonl")
    metrics_file: Path = field(default_factory=lambda: OUTPUT_DIR / "metrics_quantum.json")
    roc_plot: Path = field(default_factory=lambda: OUTPUT_DIR / "roc_curve.png")
    confusion_plot: Path = field(default_factory=lambda: OUTPUT_DIR / "confusion_matrix.png")


@dataclass(frozen=True)
class DecisionConfig:
    fake_max_prob: float = 0.3
    real_min_prob: float = 0.7
    metrics_baseline_file: Path = field(default_factory=lambda: OUTPUT_DIR / "metrics_baselines.json")
    report_file: Path = field(default_factory=lambda: DOCS_DIR / "results_report.md")
