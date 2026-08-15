"""Evaluation of the trained quantum model: metrics, decision bins,
cross-validation, classical baselines, and the artifact figures.

The quantum model is scored on the held-out test split (accuracy, F1,
AUC-ROC, ECE, decision bins). The train split is additionally
cross-validated to report procedure stability (mean +/- std). Classical
baselines (RandomForest, MLP, LogisticRegression, LinearSVC, GaussianNB,
XGBoost) run on the same selected features for comparison. Plots are
delegated to quantum.plots.
"""

import functools
import json
import os

import numpy as np

from quantum.config import DecisionConfig, VQCConfig
from quantum.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_roc_curve,
)
from quantum.vqc import load_vqc_model, predict_vqc, train_vqc


def _sklearn():
    """Lazy sklearn import: costs ~12s on this host, so it is only paid by
    processes that actually run evaluation/baselines (not by QAOA workers)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC

    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError:  # older sklearn
        StratifiedGroupKFold = None

    return {
        "CalibratedClassifierCV": CalibratedClassifierCV,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "average_precision_score": average_precision_score,
        "confusion_matrix": confusion_matrix,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "roc_auc_score": roc_auc_score,
        "GroupKFold": GroupKFold,
        "StratifiedGroupKFold": StratifiedGroupKFold,
        "StratifiedKFold": StratifiedKFold,
        "GaussianNB": GaussianNB,
        "MLPClassifier": MLPClassifier,
        "LinearSVC": LinearSVC,
    }


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
    sk = _sklearn()
    predictions = (prob_real >= 0.5).astype(int)
    precision, recall, f1, _ = sk["precision_recall_fscore_support"](
        y_true, predictions, average="binary", zero_division=0
    )
    n_classes = len(set(int(v) for v in y_true))
    if n_classes < 2:
        auc_roc = None
        pr_auc = None
    else:
        auc_roc = float(sk["roc_auc_score"](y_true, prob_real))
        pr_auc = float(sk["average_precision_score"](y_true, prob_real))
    tn, fp, fn, tp = sk["confusion_matrix"](y_true, predictions).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": float(sk["accuracy_score"](y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": specificity,
        "f1": float(f1),
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "ece": expected_calibration_error(y_true, prob_real),
    }


def balanced_accuracy(y_true, prob_real):
    """Balanced accuracy (mean of per-class recall) from probabilities."""
    sk = _sklearn()
    predictions = (prob_real >= 0.5).astype(int)
    _, recall, _, _ = sk["precision_recall_fscore_support"](
        y_true, predictions, average=None, zero_division=0
    )
    return float(np.mean(recall)) if len(recall) else 0.0


def _aggregate_cv(rows):
    """Mean/std summary across cross-validation folds."""
    keys = sorted(rows[0].keys())
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    std = {k: float(np.std([r[k] for r in rows])) for k in keys}
    return {"mean": mean, "std": std, "folds": rows}


def run_cv(fit_predict, X, y, n_splits=5, seed=42, n_jobs=0, groups=None):
    """Group-aware K-fold cross-validation for a fit_predict callable.

    `fit_predict(Xtr, ytr, Xte, yte) -> P(class=1)` is invoked per fold;
    returns _aggregate_cv payload with per-fold metrics plus balanced
    accuracy. When ``groups`` is provided, folds never separate one
    subject's clips (GroupKFold / StratifiedGroupKFold), which is the
    honest CV under the subject-grouped split. Folds are executed in
    parallel subprocesses when ``n_jobs`` > 1 (or 0 = auto).
    """
    sk = _sklearn()
    X = np.asarray(X)
    y = np.asarray(y)
    if groups is not None:
        groups = np.asarray(groups)
        if sk["StratifiedGroupKFold"] is not None:
            splitter = sk["StratifiedGroupKFold"](
                n_splits=n_splits, shuffle=True, random_state=seed
            )
        else:
            splitter = sk["GroupKFold"](n_splits=n_splits)
        fold_idx = list(splitter.split(X, y, groups))
    else:
        skf = sk["StratifiedKFold"](n_splits=n_splits, shuffle=True, random_state=seed)
        fold_idx = list(skf.split(X, y))
    fold_args = [(X[tr], y[tr], X[te], y[te]) for tr, te in fold_idx]

    if n_splits > 1 and (n_jobs != 1):
        workers = n_jobs if n_jobs > 0 else min(n_splits, os.cpu_count() or 1)
        try:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as pool:
                probs_per_fold = list(pool.map(fit_predict, *zip(*fold_args)))
        except Exception:
            probs_per_fold = [fit_predict(*args) for args in fold_args]
    else:
        probs_per_fold = [fit_predict(*args) for args in fold_args]

    rows = []
    for (_, _, _, yte), probs in zip(fold_args, probs_per_fold):
        m = classification_metrics(yte, probs)
        m["balanced_accuracy"] = balanced_accuracy(yte, probs)
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
        confirmed_accuracy = float(
            _sklearn()["accuracy_score"](y_true[fake | real], predictions[fake | real])
        )
    return {
        "real": int(real.sum()),
        "uncertain": int(uncertain.sum()),
        "fake": int(fake.sum()),
        "uncertain_rate": float(uncertain.mean()),
        "confirmed_accuracy": confirmed_accuracy,
    }


def _fit_vqc_fold(Xtr, ytr, Xte, yte, cfg):
    """Module-level CV fold worker: train a VQC on the fold, return P(real).

    Defined at module level so multiprocessing can pickle it for parallel
    cross-validation.
    """
    model = train_vqc(Xtr, ytr, cfg, X_val=Xte, y_val=yte)
    return predict_vqc(model, Xte)


def evaluate_quantum_model(
    X_test, y_test, vqc_cfg=None, decision_cfg=None, X_train=None, y_train=None, groups_train=None
):
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
    # hold-out test above is untouched). Reported as mean +/- std. Uses
    # subject-grouped folds when groups are available.
    if _enough_for_cv(y_train, n_splits=5):
        cv_cfg = VQCConfig(**{**vqc_cfg.__dict__, "save_checkpoint": False})
        payload["cv"] = run_cv(
            functools.partial(_fit_vqc_fold, cfg=cv_cfg),
            X_train,
            y_train,
            seed=vqc_cfg.seed,
            n_jobs=os.cpu_count() or 1,
            groups=groups_train,
        )

    plot_roc_curve(y_test, prob_real, decision_cfg.roc_plot)
    plot_confusion_matrix(y_test, (prob_real >= 0.5).astype(int), decision_cfg.confusion_plot)
    plot_calibration_curve(y_test, prob_real, decision_cfg.calibration_plot)
    with open(vqc_cfg.metrics_file, "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def run_baselines(
    X_train, y_train, X_test, y_test, decision_cfg=None, seed=42, n_splits=5, groups_train=None
):
    """Classical research baselines on the same selected features.

    Compared against the QAOA -> VQC decision path on the hold-out test,
    plus subject-grouped cross-validation of each baseline on the train
    split (mean +/- std, balanced accuracy).
    """
    decision_cfg = decision_cfg or DecisionConfig()
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    # xgboost is imported lazily: importing it costs ~13s on this host and
    # would otherwise slow down every process that imports this module
    # (including the QAOA parallel workers). It is also optional: a host
    # without xgboost must still complete `--all` (the VQC is the deliverable,
    # baselines are comparison only).
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415
    except ImportError:
        XGBClassifier = None

    sk = _sklearn()
    models = {
        "random_forest": lambda: sk["RandomForestClassifier"](
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
        "mlp": lambda: sk["MLPClassifier"](
            hidden_layer_sizes=(32, 16), max_iter=500, random_state=seed
        ),
        "logistic_regression": lambda: sk["LogisticRegression"](
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "linear_svc": lambda: sk["CalibratedClassifierCV"](
            sk["LinearSVC"](class_weight="balanced", max_iter=5000, random_state=seed),
            method="sigmoid",
            cv=3,
        ),
        "gaussian_nb": sk["GaussianNB"],
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
    xgboost_types = () if XGBClassifier is None else (XGBClassifier,)
    if XGBClassifier is None:
        models.pop("xgboost", None)
    for name, make_model in models.items():
        model = make_model()
        if isinstance(model, xgboost_types):
            model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
        else:
            model.fit(X_train, y_train)
        prob_real = model.predict_proba(X_test)[:, 1]
        entry = classification_metrics(y_test, prob_real)
        entry["balanced_accuracy"] = balanced_accuracy(y_test, prob_real)
        if _enough_for_cv(y_train, n_splits):

            def _fit_fold(Xtr, ytr, Xte, yte, make=make_model):
                m = make()
                if isinstance(m, xgboost_types):
                    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr)], verbose=False)
                else:
                    m.fit(Xtr, ytr)
                return m.predict_proba(Xte)[:, 1]

            entry["cv"] = run_cv(
                _fit_fold, X_train, y_train, n_splits=n_splits, seed=seed, n_jobs=1, groups=groups_train
            )
        results[name] = entry

    if XGBClassifier is None:
        results["xgboost"] = {"skipped": "xgboost not installed"}

    with open(decision_cfg.metrics_baseline_file, "w") as fh:
        json.dump(results, fh, indent=2)
    return results
