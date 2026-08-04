import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

from quantum.config import DecisionConfig, VQCConfig
from quantum.dummy_data import load_dataset
from quantum.qaoa_selection import load_selection
from quantum.train import predict_proba
from quantum.vqc import HybridVQC

DECISION_LABELS = {"fake": 0, "uncertain": 1, "real": 2}


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probs, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = y_true[mask].mean()
        total += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(total)


def decision_bins(probs: np.ndarray, config: DecisionConfig) -> np.ndarray:
    labels = np.full(len(probs), DECISION_LABELS["uncertain"])
    labels[probs >= config.real_min_prob] = DECISION_LABELS["real"]
    labels[probs <= config.fake_max_prob] = DECISION_LABELS["fake"]
    return labels


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, config: DecisionConfig) -> dict:
    y_pred = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_true, y_pred)
    decisions = decision_bins(probs, config)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(roc_auc),
        "ece": expected_calibration_error(y_true, probs),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "decisions": {
            "real_count": int((decisions == DECISION_LABELS["real"]).sum()),
            "uncertain_count": int((decisions == DECISION_LABELS["uncertain"]).sum()),
            "fake_count": int((decisions == DECISION_LABELS["fake"]).sum()),
        },
        "thresholds": {"fake_max_prob": config.fake_max_prob, "real_min_prob": config.real_min_prob},
    }


def plot_roc(y_true: np.ndarray, probs: np.ndarray, path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Hybrid VQC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Quantum Classifier")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, probs: np.ndarray, path) -> None:
    y_pred = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fake", "Real"])
    ax.set_yticklabels(["Fake", "Real"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Quantum Classifier")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_quantum_model(
    data: dict,
    selection: dict,
    config: VQCConfig,
    decision_config: DecisionConfig,
) -> dict:
    checkpoint = load_checkpoint(config.checkpoint_file)
    model = HybridVQC(n_qubits=len(selection["selected_indices"]), config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected = selection["selected_indices"]

    probs_test = predict_proba(model, data["X_test"], selected)
    metrics = compute_metrics(data["y_test"], probs_test, decision_config)
    metrics["model"] = "Hybrid VQC (QAOA-selected features)"
    metrics["selected_features"] = selection["selected_features"]
    metrics["expected_cost_energy"] = selection.get("expected_cost_energy")
    metrics["checkpoint"] = str(config.checkpoint_file)
    metrics["n_epochs_trained"] = checkpoint["config"]["epochs"]

    plot_roc(data["y_test"], probs_test, config.roc_plot)
    plot_confusion(data["y_test"], probs_test, config.confusion_plot)
    metrics["roc_plot"] = str(config.roc_plot)
    metrics["confusion_plot"] = str(config.confusion_plot)

    config.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_checkpoint(path):
    import torch

    return torch.load(path, map_location="cpu")


def load_quantum_metrics(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
