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


def classification_metrics(y_true, prob_real, decision_threshold=0.5):
    """Binary classification metrics (REAL vs FAKE).
    
    Args:
        y_true: True labels (0=FAKE, 1=REAL)
        prob_real: Predicted probability of REAL
        decision_threshold: Threshold for REAL (>=) vs FAKE (<)
    """
    sk = _sklearn()
    
    # Binary predictions
    predictions = (prob_real >= decision_threshold).astype(int)  # 1=REAL, 0=FAKE
    
    precision, recall, f1, _ = sk["precision_recall_fscore_support"](
        y_true, predictions, average="binary", zero_division=0
    )
    if len(set(int(v) for v in y_true)) >= 2:
        auc_roc = float(sk["roc_auc_score"](y_true, prob_real))
        pr_auc = float(sk["average_precision_score"](y_true, prob_real))
    else:
        auc_roc = None
        pr_auc = None
    
    cm = sk["confusion_matrix"](y_true, predictions)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Single class case
        tn = fp = fn = tp = 0
        if cm.size == 1:
            if y_true[0] == 0:  # only FAKE
                tn = cm[0, 0]
            else:  # only REAL
                tp = cm[0, 0]
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    accuracy = float(sk["accuracy_score"](y_true, predictions))
    
    # Coverage (always 1.0 for binary)
    n_total = len(y_true)
    coverage = 1.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": specificity,
        "f1": float(f1),
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "confusion_matrix_binary": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "ece": expected_calibration_error(y_true, prob_real),
        "coverage": float(coverage),
        "n_classified": int(n_total),
        "n_total": int(n_total),
        "n_insufficient": 0,
    }


def balanced_accuracy(y_true, prob_real, decision_threshold=0.5):
    """Balanced accuracy (mean of per-class recall) from probabilities for binary."""
    sk = _sklearn()
    
    predictions = (prob_real >= decision_threshold).astype(int)
    
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


def decision_bins(y_true, prob_real, cfg=None, pqs=None):
    """Binary decision bins: REAL, FAKE.
    
    Decision logic:
    - prob_real >= decision_threshold -> REAL
    - prob_real < decision_threshold -> FAKE
    """
    cfg = cfg or DecisionConfig()
    threshold = cfg.decision_threshold
    
    # Binary classification
    real_mask = prob_real >= threshold
    fake_mask = prob_real < threshold
    
    n = len(prob_real)
    real_count = int(real_mask.sum())
    fake_count = int(fake_mask.sum())
    
    return {
        "real": real_count,
        "fake": fake_count,
        "coverage": 1.0,
        "decision_threshold": float(threshold),
    }


def optimal_threshold_youden(y_true, prob_real):
    """Compute optimal threshold using Youden's J statistic (max sensitivity + specificity - 1).
    
    Returns the threshold that maximizes J = sensitivity + specificity - 1.
    """
    thresholds = np.unique(prob_real)
    best_j = -1.0
    best_t = 0.5
    for t in thresholds:
        pred = (prob_real >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sensitivity + specificity - 1
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


def analyze_threshold_behavior(y_true, prob_real, cfg=None):
    """Analyze how thresholds affect predictions.

    Returns dict with threshold sweep analysis to diagnose whether
    the decision boundary is effective or if discrimination is weak.

    Case A: Scores separate classes but threshold is too conservative
    Case B: Scores do not separate classes (discrimination problem)
    """
    cfg = cfg or DecisionConfig()

    # Sort scores and compute metrics at each threshold
    indices = np.argsort(-prob_real)  # descending order
    sorted_probs = prob_real[indices]
    sorted_y = y_true[indices]

    # Compute running metrics at each threshold (descending prob_real)
    tps = np.cumsum(sorted_y)  # true positives (REAL) at each threshold
    fps = np.cumsum(1 - sorted_y)  # false positives (FAKE predicted as REAL)
    n_pos = int(sorted_y.sum())  # total REAL samples
    n_neg = int((1 - sorted_y).sum())  # total FAKE samples
    fn_s = n_pos - tps  # false negatives (REAL misclassified)
    tn_s = n_neg - fps  # true negatives (FAKE correctly identified)

    # REAL recall and precision at each threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where((tps + fn_s) > 0, tps / (tps + fn_s), 0.0)
        precision = np.where((tps + fps) > 0, tps / (tps + fps), 0.0)
        specificity_arr = np.where((tn_s + fps) > 0, tn_s / (tn_s + fps), 0.0)

    # Score ranges per class
    fake_thresholds = sorted_probs[sorted_y == 0]  # scores of FAKE samples
    real_thresholds = sorted_probs[sorted_y == 1]  # scores of REAL samples

    analysis = {
        "n_total": int(len(prob_real)),
        "n_real": n_pos,
        "n_fake": n_neg,
        "decision_threshold": cfg.decision_threshold,
        "score_range": [float(prob_real.min()), float(prob_real.max())],
        "fake_scores_min": float(fake_thresholds.min()) if len(fake_thresholds) > 0 else None,
        "real_scores_max": float(real_thresholds.max()) if len(real_thresholds) > 0 else None,
        "proportion_fake_below_threshold": float((prob_real < cfg.decision_threshold).sum()) / len(prob_real),
        "proportion_real_above_threshold": float((prob_real >= cfg.decision_threshold).sum()) / len(prob_real),
        "recall_at_last": float(recall[-1]) if len(recall) > 0 else 0.0,
        "specificity_at_last": float(specificity_arr[-1]) if len(specificity_arr) > 0 else 0.0,
        "precision_at_last": float(precision[-1]) if len(precision) > 0 else 0.0,
        "threshold_sweep": [
            {
                "threshold": float(sorted_probs[i]),
                "recall": float(recall[i]),
                "precision": float(precision[i]),
                "specificity": float(specificity_arr[i]),
            }
            for i in range(0, len(sorted_probs), max(1, len(sorted_probs) // 20))
        ],
        "diagnosis": "B" if (fake_thresholds.max() if len(fake_thresholds) > 0 else 0) > (real_thresholds.min() if len(real_thresholds) > 0 else 1) else "A",
        "diagnosis_text": (
            "Scores do not separate classes (discrimination problem)"
            if (fake_thresholds.max() if len(fake_thresholds) > 0 else 0) > (real_thresholds.min() if len(real_thresholds) > 0 else 1)
            else "Scores separate classes but threshold may be suboptimal"
        ),
    }
    return analysis


def _fit_vqc_fold(Xtr, ytr, Xte, yte, cfg):
    """Module-level CV fold worker: train a VQC on the fold, return P(real).

    Defined at module level so multiprocessing can pickle it for parallel
    cross-validation.
    """
    model = train_vqc(Xtr, ytr, cfg, X_val=Xte, y_val=yte)
    return predict_vqc(model, Xte)


def evaluate_quantum_model(
    X_test, y_test, vqc_cfg=None, decision_cfg=None, X_train=None, y_train=None, groups_train=None, pqs=None
):
    vqc_cfg = vqc_cfg or VQCConfig()
    decision_cfg = decision_cfg or DecisionConfig()
    model = load_vqc_model(X_test.shape[1], vqc_cfg)
    prob_real = predict_vqc(model, X_test)

    # Load optimal threshold from checkpoint metadata
    import torch
    ckpt = torch.load(vqc_cfg.checkpoint_file, map_location="cpu", weights_only=False)
    opt_threshold = ckpt.get("metadata", {}).get("decision_threshold", decision_cfg.decision_threshold)
    from dataclasses import replace
    decision_cfg_opt = replace(decision_cfg, decision_threshold=opt_threshold)

    payload = {
        "metrics": classification_metrics(y_test, prob_real),
        "balanced_accuracy": balanced_accuracy(y_test, prob_real),
        "decision_bins": decision_bins(y_test, prob_real, decision_cfg_opt, pqs=pqs),
    }

    # Cross-validation of the training procedure on the train split (the
    # hold-out test above is untouched). Reported as mean +/- std. Uses
    # subject-grouped folds when groups are available.
    if _enough_for_cv(y_train, n_splits=5):
        cv_cfg = VQCConfig(**{**vqc_cfg.__dict__, "save_checkpoint": False})
        # On GPU hosts the fold trainers use CUDA themselves; running folds
        # in parallel would open one CUDA context per worker on a shared
        # card (OOM risk on 6 GB). The GPU is the parallelism: folds run
        # sequentially on it. CPU hosts keep parallel folds.
        try:
            import torch  # noqa: PLC0415

            gpu_host = torch.cuda.is_available()
        except Exception:
            gpu_host = False
        n_jobs = 1 if gpu_host else (os.cpu_count() or 1)
        payload["cv"] = run_cv(
            functools.partial(_fit_vqc_fold, cfg=cv_cfg),
            X_train,
            y_train,
            seed=vqc_cfg.seed,
            n_jobs=n_jobs,
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
            **(lambda: ({"tree_method": "hist", "device": "cuda"}
                if XGBClassifier is not None
                else {}))(),
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
