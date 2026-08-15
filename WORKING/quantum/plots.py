"""Matplotlib figure helpers for quantum-layer evaluation artifacts.

Each helper writes a single PNG (Agg backend, no GUI) and closes its
figure so callers can loop over many plots without leaking memory.
"""

import numpy as np


def _plt():
    """Deferred matplotlib import (~0.76 s on this host).

    Keep it (and the Agg backend selection) inside the plot functions
    so `import quantum.pipeline` stays fast for QAOA spawn workers and
    server restarts; only evaluation that actually renders figures pays
    the import cost.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def _sklearn():
    """Deferred sklearn import (~2.4 s on this host).

    Keep it inside the plot functions so `import quantum.pipeline`
    stays fast for QAOA spawn workers and server restarts.
    """
    from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve  # noqa: PLC0415

    return ConfusionMatrixDisplay, roc_auc_score, roc_curve


def plot_roc_curve(y_true, prob_real, path):
    if len(set(int(v) for v in y_true)) < 2:
        return
    plt = _plt()
    ConfusionMatrixDisplay, roc_auc_score, roc_curve = _sklearn()
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


def plot_confusion_matrix(y_true, predictions, path):
    plt = _plt()
    ConfusionMatrixDisplay, _, _ = _sklearn()
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (Hybrid VQC)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration_curve(y_true, prob_real, path, n_bins=10):
    """Reliability diagram: observed vs predicted probability per bin."""
    plt = _plt()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, observed, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (prob_real > lo) & (prob_real <= hi)
        if int(in_bin.sum()) == 0:
            continue
        centers.append((lo + hi) / 2.0)
        observed.append(float(y_true[in_bin].mean()))
        counts.append(int(in_bin.sum()))
    fig, ax = plt.subplots(figsize=(5, 5))
    if centers:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(centers, observed, "o-", label="Hybrid VQC")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title("Calibration Curve (Hybrid VQC)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)