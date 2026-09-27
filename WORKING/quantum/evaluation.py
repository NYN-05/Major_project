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


def classification_metrics(y_true, prob_real, fake_max_prob=0.3, real_min_prob=0.7):
    """Classification metrics for three-state decision (REAL, FAKE, INSUFFICIENT EVIDENCE).
    
    Args:
        y_true: True labels (0=FAKE, 1=REAL)
        prob_real: Predicted probability of REAL
        fake_max_prob: Threshold below which prediction is FAKE
        real_min_prob: Threshold above which prediction is REAL
    """
    sk = _sklearn()
    
    # Three-state predictions
    predictions = np.full_like(prob_real, -1, dtype=int)  # -1 = INSUFFICIENT EVIDENCE
    predictions[prob_real >= real_min_prob] = 1  # REAL
    predictions[prob_real <= fake_max_prob] = 0  # FAKE
    # -1 remains for INSUFFICIENT EVIDENCE
    
    # Binary metrics for REAL vs FAKE (ignoring INSUFFICIENT EVIDENCE)
    has_sufficient = predictions != -1
    if has_sufficient.any():
        y_sufficient = y_true[has_sufficient]
        pred_sufficient = predictions[has_sufficient]
        prob_sufficient = prob_real[has_sufficient]
        
        precision, recall, f1, _ = sk["precision_recall_fscore_support"](
            y_sufficient, pred_sufficient, average="binary", zero_division=0
        )
        if len(set(int(v) for v in y_sufficient)) >= 2:
            auc_roc = float(sk["roc_auc_score"](y_sufficient, prob_sufficient))
            pr_auc = float(sk["average_precision_score"](y_sufficient, prob_sufficient))
        else:
            auc_roc = None
            pr_auc = None
        
        tn, fp, fn, tp = sk["confusion_matrix"](y_sufficient, pred_sufficient).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        accuracy = float(sk["accuracy_score"](y_sufficient, pred_sufficient))
    else:
        precision = recall = f1 = 0.0
        auc_roc = pr_auc = None
        specificity = 0.0
        accuracy = 0.0
        tn = fp = fn = tp = 0
    
    # Coverage metrics
    n_total = len(y_true)
    n_classified = int(has_sufficient.sum()) if has_sufficient.any() else 0
    coverage = n_classified / n_total if n_total > 0 else 0.0
    
    # Three-class confusion matrix (REAL=1, FAKE=0, INSUFFICIENT=-1)
    y_true_3class = np.full_like(y_true, -1)
    y_true_3class[y_true == 1] = 1  # REAL
    y_true_3class[y_true == 0] = 0  # FAKE
    
    cm_3class = sk["confusion_matrix"](y_true_3class, predictions, labels=[-1, 0, 1])
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": specificity,
        "f1": float(f1),
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "confusion_matrix_binary": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "confusion_matrix_3class": cm_3class.tolist(),
        "ece": expected_calibration_error(
            y_true[has_sufficient] if has_sufficient.any() else y_true, 
            prob_real[has_sufficient] if has_sufficient.any() else prob_real
        ),
        "coverage": float(coverage),
        "n_classified": int(n_classified),
        "n_total": int(n_total),
        "n_insufficient": int((~has_sufficient).sum()) if has_sufficient.any() else n_total,
    }


def balanced_accuracy(y_true, prob_real, fake_max_prob=0.3, real_min_prob=0.7):
    """Balanced accuracy (mean of per-class recall) from probabilities for three-state."""
    sk = _sklearn()
    
    predictions = np.full_like(prob_real, -1, dtype=int)
    predictions[prob_real >= real_min_prob] = 1
    predictions[prob_real <= fake_max_prob] = 0
    
    has_sufficient = predictions != -1
    if not has_sufficient.any():
        return 0.0
    
    y_sufficient = y_true[has_sufficient]
    pred_sufficient = predictions[has_sufficient]
    
    _, recall, _, _ = sk["precision_recall_fscore_support"](
        y_sufficient, pred_sufficient, average=None, zero_division=0
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
    """Three-state decision bins: REAL, FAKE, INSUFFICIENT EVIDENCE / REVIEW REQUIRED.
    
    Decision logic:
    - prob_real >= real_min_prob -> REAL
    - prob_real <= fake_max_prob -> FAKE
    - fake_max_prob < prob_real < real_min_prob -> INSUFFICIENT EVIDENCE
    - If pqs is provided and pqs < quality_threshold -> INSUFFICIENT EVIDENCE (overrides prob_real)
    """
    cfg = cfg or DecisionConfig()
    real_min = cfg.real_min_prob
    fake_max = cfg.fake_max_prob
    quality_thresh = cfg.quality_threshold
    
    # Start with all samples as INSUFFICIENT EVIDENCE
    n = len(prob_real)
    decision = np.full(n, "INSUFFICIENT EVIDENCE", dtype=object)
    
    # Apply quality threshold if PQS provided
    if pqs is not None:
        sufficient_quality = pqs >= cfg.quality_threshold
    else:
        sufficient_quality = np.ones(len(prob_real), dtype=bool)
    
    # Classify as REAL
    real_mask = (prob_real >= real_min) & sufficient_quality
    decision[real_mask] = "REAL"
    
    # Classify as FAKE
    fake_mask = (prob_real <= fake_max) & sufficient_quality
    decision[fake_mask] = "FAKE"
    
    # Count results
    real_count = int(real_mask.sum())
    fake_count = int(fake_mask.sum())
    insufficient_count = int((~real_mask & ~fake_mask).sum())
    
    return {
        "real": real_count,
        "fake": fake_count,
        "insufficient_evidence": insufficient_count,
        "coverage": float((real_mask | fake_mask).sum()) / len(prob_real) if len(prob_real) > 0 else 0.0,
        "fake_max_prob": float(fake_max),
        "real_min_prob": float(real_min),
        "quality_threshold": float(quality_thresh),
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
    100% UNCERTAIN is due to weak discrimination or conservative thresholds.

    Case A: Scores separate classes but 0.3/0.7 is too conservative
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

    # FAKE recall = proportion of FAKE samples with prob_real <= fake_max_prob
    fake_recall_direct = (
        float((prob_real <= cfg.fake_max_prob).sum()) / n_neg if n_neg > 0 else 0.0
    )

    # Score ranges per class
    fake_thresholds = sorted_probs[sorted_y == 0]  # scores of FAKE samples
    real_thresholds = sorted_probs[sorted_y == 1]  # scores of REAL samples

    analysis = {
        "n_total": int(len(prob_real)),
        "n_real": n_pos,
        "n_fake": n_neg,
        "n_uncertain_at_03_07": int(
            ((prob_real > cfg.fake_max_prob) & (prob_real < cfg.real_min_prob)).sum()
        ),
        "fake_max_prob": cfg.fake_max_prob,
        "real_min_prob": cfg.real_min_prob,
        "score_range": [float(prob_real.min()), float(prob_real.max())],
        "fake_scores": float(fake_thresholds.min()) if len(fake_thresholds) > 0 else None,
        "real_scores": float(real_thresholds.max()) if len(real_thresholds) > 0 else None,
        "proportion_fake_below_03": float((prob_real < cfg.fake_max_prob).sum()) / len(prob_real),
        "proportion_real_above_07": float((prob_real >= cfg.real_min_prob).sum()) / len(prob_real),
        "fake_recall_direct": fake_recall_direct,
        "real_recall": float(recall[-1]) if len(recall) > 0 else 0.0,
        "specificity": float(specificity_arr[-1]) if len(specificity_arr) > 0 else 0.0,
        "precision_at_last": float(precision[-1]) if len(precision) > 0 else 0.0,
    }

    # Diagnosis: Case A or Case B
    # Case A: Some samples have prob_real >= 0.7 or <= 0.3, but 0.3/0.7 threshold misses them
    # Case B: All (or almost all) prob_real are between 0.3 and 0.7

    n_real_above_07 = analysis["proportion_real_above_07"] * analysis["n_total"]
    n_fake_below_03 = analysis["proportion_fake_below_03"] * analysis["n_total"]

    if n_real_above_07 == 0 and n_fake_below_03 == 0:
        analysis["diagnosis"] = "Case B: Scores do not separate classes - all probabilities concentrated in uncertain region"
        analysis["recommendation"] = "Proceed upstream to rPPG and feature improvement (Phase 4-6)"
    elif analysis["n_uncertain_at_03_07"] < analysis["n_total"]:
        analysis["diagnosis"] = "Case A: Scores partially separate classes - some confident predictions possible"
        analysis["recommendation"] = "Threshold adjustment could increase coverage, but discrimination is limited (AUC ~0.53)"
    else:
        analysis["diagnosis"] = "Case B: Scores do not separate classes"
        analysis["recommendation"] = "Proceed upstream to rPPG and feature improvement"

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
