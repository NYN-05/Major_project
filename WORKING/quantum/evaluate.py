import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from quantum.config import DecisionConfig, VQCConfig
from quantum.vqc import load_vqc_model, predict_vqc, train_vqc


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


def balanced_accuracy(y_true, prob_real):
    """Balanced accuracy (mean of per-class recall) from probabilities."""
    predictions = (prob_real >= 0.5).astype(int)
    _, recall, _, _ = precision_recall_fscore_support(
        y_true, predictions, average=None, zero_division=0
    )
    return float(np.mean(recall)) if len(recall) else 0.0


def _aggregate_cv(rows):
    """Mean/std summary across cross-validation folds."""
    keys = sorted(rows[0].keys())
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    std = {k: float(np.std([r[k] for r in rows])) for k in keys}
    return {"mean": mean, "std": std, "folds": rows}


def run_cv(fit_predict, X, y, n_splits=5, seed=42):
    """Stratified K-fold cross-validation for a fit_predict callable.

    `fit_predict(Xtr, ytr, Xte, yte) -> P(class=1)` is invoked per fold;
    returns _aggregate_cv payload with per-fold metrics plus balanced
    accuracy.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = np.asarray(X)
    y = np.asarray(y)
    rows = []
    for tr, te in skf.split(X, y):
        probs = fit_predict(X[tr], y[tr], X[te], y[te])
        m = classification_metrics(y[te], probs)
        m["balanced_accuracy"] = balanced_accuracy(y[te], probs)
        rows.append(m)
    return _aggregate_cv(rows)


def _enough_for_cv(y, n_splits=5):
    counts = np.bincount(np.asarray(y, dtype=int))
    return min(counts) >= 2 * n_splits


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


def _plot_calibration(y_true, prob_real, path, n_bins=10):
    """Reliability diagram: observed vs predicted probability per bin."""
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


def evaluate_quantum_model(X_test, y_test, vqc_cfg=None, decision_cfg=None, X_train=None, y_train=None):
    vqc_cfg = vqc_cfg or VQCConfig()
    decision_cfg = decision_cfg or DecisionConfig()
    model = load_vqc_model(X_test.shape[1], vqc_cfg)
    prob_real = predict_vqc(model, X_test)
    payload = {
        "metrics": classification_metrics(y_test, prob_real),
        "balanced_accuracy": balanced_accuracy(y_test, prob_real),
        "decision_bins": decision_bins(y_test, prob_real, decision_cfg),
    }

    # Cross-validation of the training procedure on the train split (the
    # hold-out test above is untouched). Reported as mean +/- std.
    if _enough_for_cv(y_train, n_splits=5):
        cv_cfg = VQCConfig(**{**vqc_cfg.__dict__, "save_checkpoint": False})

        def _fit_vqc(Xtr, ytr, Xte, yte):
            model = train_vqc(Xtr, ytr, cv_cfg, X_val=Xte, y_val=yte)
            return predict_vqc(model, Xte)

        payload["cv"] = run_cv(_fit_vqc, X_train, y_train, seed=vqc_cfg.seed)

    _plot_roc(y_test, prob_real, decision_cfg.roc_plot)
    _plot_confusion(y_test, (prob_real >= 0.5).astype(int), decision_cfg.confusion_plot)
    _plot_calibration(y_test, prob_real, decision_cfg.calibration_plot)
    with open(vqc_cfg.metrics_file, "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def run_baselines(X_train, y_train, X_test, y_test, decision_cfg=None, seed=42, n_splits=5):
    """Classical research baselines on the same selected features.

    Compared against the QAOA -> VQC decision path on the hold-out test,
    plus StratifiedKFold cross-validation of each baseline on the train
    split (mean +/- std, balanced accuracy).
    """
    decision_cfg = decision_cfg or DecisionConfig()
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    models = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=500, random_state=seed
        ),
        "logistic_regression": lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "linear_svc": lambda: CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed),
            method="sigmoid",
            cv=3,
        ),
        "gaussian_nb": GaussianNB,
        "xgboost": lambda: XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=3,
            eval_metric="logloss",
            early_stopping_rounds=20,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    results = {}
    for name, make_model in models.items():
        model = make_model()
        if isinstance(model, XGBClassifier):
            model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
        else:
            model.fit(X_train, y_train)
        prob_real = model.predict_proba(X_test)[:, 1]
        entry = classification_metrics(y_test, prob_real)
        entry["balanced_accuracy"] = balanced_accuracy(y_test, prob_real)
        if _enough_for_cv(y_train, n_splits):

            def _fit_fold(Xtr, ytr, Xte, yte, make=make_model):
                m = make()
                if isinstance(m, XGBClassifier):
                    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr)], verbose=False)
                else:
                    m.fit(Xtr, ytr)
                return m.predict_proba(Xte)[:, 1]

            entry["cv"] = run_cv(_fit_fold, X_train, y_train, n_splits=n_splits, seed=seed)
        results[name] = entry

    with open(decision_cfg.metrics_baseline_file, "w") as fh:
        json.dump(results, fh, indent=2)
    return results
