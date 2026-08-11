import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier

from quantum.config import DecisionConfig, VQCConfig
from quantum.vqc import load_vqc_model, predict_vqc


def expected_calibration_error(y_true, prob_real, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (prob_real > lo) & (prob_real <= hi)
        total = int(in_bin.sum())
        if total == 0:
            continue
        bin_confidence = float(prob_real[in_bin].mean())
        bin_accuracy = float(y_true[in_bin].mean())
        ece += (total / len(prob_real)) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def classification_metrics(y_true, prob_real):
    predictions = (prob_real >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(roc_auc_score(y_true, prob_real)),
        "ece": expected_calibration_error(y_true, prob_real),
    }


def decision_bins(y_true, prob_real, cfg=None):
    cfg = cfg or DecisionConfig()
    fake = prob_real <= cfg.fake_max_prob
    real = prob_real >= cfg.real_min_prob
    uncertain = ~(fake | real)
    confirmed_accuracy = None
    if (fake | real).any():
        predictions = (prob_real >= 0.5).astype(int)
        confirmed_accuracy = float(accuracy_score(y_true[fake | real], predictions[fake | real]))
    return {
        "real": int(real.sum()),
        "uncertain": int(uncertain.sum()),
        "fake": int(fake.sum()),
        "uncertain_rate": float(uncertain.mean()),
        "confirmed_accuracy": confirmed_accuracy,
    }


def _plot_roc(y_true, prob_real, path):
    fpr, tpr, _ = roc_curve(y_true, prob_real)
    auc = roc_auc_score(y_true, prob_real)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC-ROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Hybrid VQC ROC Curve")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion(y_true, predictions, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (Hybrid VQC)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_quantum_model(X_test, y_test, vqc_cfg=None, decision_cfg=None):
    vqc_cfg = vqc_cfg or VQCConfig()
    decision_cfg = decision_cfg or DecisionConfig()
    model = load_vqc_model(X_test.shape[1], vqc_cfg)
    prob_real = predict_vqc(model, X_test)
    payload = {
        "metrics": classification_metrics(y_test, prob_real),
        "decision_bins": decision_bins(y_test, prob_real, decision_cfg),
    }
    _plot_roc(y_test, prob_real, decision_cfg.roc_plot)
    _plot_confusion(y_test, (prob_real >= 0.5).astype(int), decision_cfg.confusion_plot)
    with open(vqc_cfg.metrics_file, "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def run_baselines(X_train, y_train, X_test, y_test, decision_cfg=None, seed=42):
    """Classical research baselines (RandomForest, MLP) on the same
    selected features; kept separate from the QAOA -> VQC decision path."""
    decision_cfg = decision_cfg or DecisionConfig()
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=500, random_state=seed
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(np.asarray(X_train), np.asarray(y_train))
        prob_real = model.predict_proba(np.asarray(X_test))[:, 1]
        results[name] = classification_metrics(y_test, prob_real)
    with open(decision_cfg.metrics_baseline_file, "w") as fh:
        json.dump(results, fh, indent=2)
    return results