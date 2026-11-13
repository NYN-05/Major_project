"""Evaluation of the trained quantum model: metrics, decision bins,
cross-validation, classical baselines, and the artifact figures.

The quantum model is scored on the held-out test split (accuracy, F1,
AUC-ROC, ECE, decision bins). The train split is additionally
cross-validated to report procedure stability (mean +/- std). Classical
baselines (RandomForest, MLP, LogisticRegression, LinearSVC, GaussianNB,
XGBoost) run on the same selected features for comparison. Figure
helpers for ROC/confusion/calibration PNGs live at the bottom of this
module (deferred matplotlib/sklearn imports keep module import fast).
"""

import functools
import json
import os

import numpy as np

from quantum.config import DecisionConfig, VQCConfig
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
    y_true = np.asarray(y_true, dtype=int)
    prob_real = np.asarray(prob_real, dtype=float)
    if len(y_true) != len(prob_real) or len(y_true) == 0:
        raise ValueError("Calibration requires non-empty paired labels and probabilities")
    if not np.isfinite(prob_real).all() or ((prob_real < 0) | (prob_real > 1)).any():
        raise ValueError("Calibration probabilities must be finite and in [0, 1]")
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
    y_true = np.asarray(y_true, dtype=int)
    prob_real = np.asarray(prob_real, dtype=float)
    if len(y_true) != len(prob_real) or len(y_true) == 0:
        raise ValueError("Metrics require non-empty paired labels and probabilities")
    if not np.isfinite(prob_real).all() or ((prob_real < 0) | (prob_real > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    
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
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fake_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    false_acceptance_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    false_rejection_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    brier = float(np.mean((prob_real - np.asarray(y_true, dtype=float)) ** 2))
    
    # Coverage (always 1.0 for binary)
    n_total = len(y_true)
    coverage = 1.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": specificity,
        "balanced_accuracy": float(balanced_acc),
        "fake_recall": fake_recall,
        "false_acceptance_rate": false_acceptance_rate,
        "false_rejection_rate": false_rejection_rate,
        "brier_score": brier,
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
    # The per-fold confusion matrix is a 2x2 list: averaging it collapses it
    # to a meaningless scalar, so it is summed element-wise (pooled) instead.
    matrix_key = "confusion_matrix_binary"
    keys = sorted(k for k in rows[0].keys() if k != matrix_key)
    # AUC/PR-AUC are None for a single-class fold; average only defined values.
    mean, std = {}, {}
    for k in keys:
        vals = [r[k] for r in rows if r[k] is not None]
        mean[k] = float(np.mean(vals)) if vals else None
        std[k] = float(np.std(vals)) if vals else None
    out = {"mean": mean, "std": std, "folds": rows}
    if matrix_key in rows[0]:
        out["confusion_matrix_pooled"] = np.sum(
            [np.asarray(r[matrix_key]) for r in rows], axis=0
        ).astype(int).tolist()
    return out


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


def choose_validation_threshold(y_true, prob_real, *, method="youden"):
    """Choose a threshold using validation data only."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob_real, dtype=float)
    arr = np.asarray(p, dtype=np.float64)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count + inf_count:
        raise ValueError(
            f"Invalid feature vector: nan_count={nan_count}, inf_count={inf_count}"
        )
    if len(y) != len(p) or len(np.unique(y)) < 2:
        raise ValueError("Threshold selection requires paired validation data with both classes")
    candidates = np.unique(np.r_[0.0, p, 1.0])
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        pred = p >= threshold
        tp = np.sum(pred & (y == 1))
        tn = np.sum(~pred & (y == 0))
        fp = np.sum(pred & (y == 0))
        fn = np.sum(~pred & (y == 1))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        score = sensitivity + specificity - 1 if method == "youden" else (sensitivity + specificity) / 2
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1]


def optimal_threshold_youden(y_true, prob_real):
    """Compute optimal threshold using Youden's J statistic (max sensitivity + specificity - 1).
    
    Returns the threshold that maximizes J = sensitivity + specificity - 1.
    """
    return choose_validation_threshold(y_true, prob_real, method="youden")


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
    ckpt = torch.load(vqc_cfg.checkpoint_file, map_location="cpu", weights_only=True)
    opt_threshold = ckpt.get("metadata", {}).get("decision_threshold", decision_cfg.decision_threshold)
    from dataclasses import replace
    decision_cfg_opt = replace(decision_cfg, decision_threshold=opt_threshold)

    payload = {
        "metrics": classification_metrics(y_test, prob_real),
        # Metrics at the deployed, validation-calibrated cut that inference
        # actually uses; "metrics" above stays at 0.5 for baseline parity.
        "metrics_at_deployed_threshold": classification_metrics(
            y_test, prob_real, decision_threshold=opt_threshold
        ),
        "deployed_threshold": float(opt_threshold),
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
    plot_confusion_matrix(
        y_test, (prob_real >= opt_threshold).astype(int), decision_cfg.confusion_plot
    )
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


# ---------------------------------------------------------------------------
# Group-aware uncertainty and nested cross-validation (Stage 3).
#
# Clips from one source group are not independent, so uncertainty is estimated
# by resampling whole groups, and every fitted step (scaling, feature selection,
# tuning, threshold) must live inside the training portion of each fold.
# ---------------------------------------------------------------------------


def _auc_rank(y, s):
    """ROC-AUC via average ranks; cheap enough for thousands of bootstrap resamples."""
    from scipy.stats import rankdata

    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _balanced_accuracy_from_pred(y, pred):
    pos = np.asarray(y) == 1
    if not pos.any() or pos.all():
        return float("nan")
    return float(0.5 * (pred[pos].mean() + (~pred[~pos]).mean()))


def _metrics_over_seeds(y, scores, thresholds):
    """Metrics of ``scores`` (n_seeds x n) against ``y``, averaged over the seeds."""
    names = ["auc", "accuracy@0.5", "balanced_accuracy@0.5"]
    if thresholds is not None:
        names += ["accuracy@fold_threshold", "balanced_accuracy@fold_threshold"]
    acc = {k: [] for k in names}
    real = np.asarray(y) == 1
    for k in range(scores.shape[0]):
        s = scores[k]
        p = s >= 0.5
        acc["auc"].append(_auc_rank(y, s))
        acc["accuracy@0.5"].append(float((p == real).mean()))
        acc["balanced_accuracy@0.5"].append(_balanced_accuracy_from_pred(y, p))
        if thresholds is not None:
            pt = s >= thresholds[k]
            acc["accuracy@fold_threshold"].append(float((pt == real).mean()))
            acc["balanced_accuracy@fold_threshold"].append(_balanced_accuracy_from_pred(y, pt))
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def _group_resamples(groups, n_boot, seed):
    """Yield index arrays that resample whole groups with replacement."""
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.where(inverse == g)[0] for g in range(len(uniq))]
    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), size=len(uniq))
        yield np.concatenate([members[g] for g in pick])


def group_bootstrap(y, scores, groups, thresholds=None, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CIs from resampling source groups (not clips).

    ``scores`` is P(real) with shape (n,) or (n_seeds, n); the reported metric is the mean over
    seeds, so the interval reflects group-sampling uncertainty of that seed-averaged metric.
    ``thresholds`` (same shape) are the per-clip fold thresholds chosen inside training data.
    The result states the number of independent groups; narrow intervals do not imply
    generalisation beyond the dataset or correctness of the grouping.
    """
    y = np.asarray(y)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    thr = None if thresholds is None else np.atleast_2d(np.asarray(thresholds, dtype=float))
    point = _metrics_over_seeds(y, scores, thr)
    boots = {k: [] for k in point}
    for idx in _group_resamples(groups, n_boot, seed):
        m = _metrics_over_seeds(y[idx], scores[:, idx], None if thr is None else thr[:, idx])
        for k, v in m.items():
            boots[k].append(v)
    out = {}
    for k, v in boots.items():
        lo, hi = np.nanpercentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = {"point": point[k], "ci95": [float(lo), float(hi)]}
    return {
        "n_clips": int(len(y)),
        "n_groups": int(len(np.unique(groups))),
        "n_seeds": int(scores.shape[0]),
        "n_boot": int(n_boot),
        "metrics": out,
    }


def paired_group_bootstrap_diff(y, scores_a, scores_b, groups, metric="auc", n_boot=2000, seed=0,
                                thresholds_a=None, thresholds_b=None, alpha=0.05):
    """Paired group bootstrap of metric(A) - metric(B) on identical resamples."""
    y = np.asarray(y)
    a = np.atleast_2d(np.asarray(scores_a, dtype=float))
    b = np.atleast_2d(np.asarray(scores_b, dtype=float))
    ta = None if thresholds_a is None else np.atleast_2d(np.asarray(thresholds_a, dtype=float))
    tb = None if thresholds_b is None else np.atleast_2d(np.asarray(thresholds_b, dtype=float))
    point = (_metrics_over_seeds(y, a, ta)[metric] - _metrics_over_seeds(y, b, tb)[metric])
    diffs = []
    for idx in _group_resamples(groups, n_boot, seed):
        da = _metrics_over_seeds(y[idx], a[:, idx], None if ta is None else ta[:, idx])[metric]
        db = _metrics_over_seeds(y[idx], b[:, idx], None if tb is None else tb[:, idx])[metric]
        diffs.append(da - db)
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "metric": metric, "difference_a_minus_b": float(point), "ci95": [float(lo), float(hi)],
        "fraction_of_resamples_a_not_better": float(np.mean(np.asarray(diffs) <= 0.0)),
        "n_groups": int(len(np.unique(groups))), "n_boot": int(n_boot),
    }


def nested_group_cv(fit_predict, X, y, groups, outer_fold, seed):
    """Outer loop over frozen development folds (``outer_fold`` >= 0; -1 rows are ignored).

    ``fit_predict(X_tr, y_tr, g_tr, X_te, seed) -> {"p_test": P(real), "threshold": float, ...}``
    must fit everything (scaling, selection, tuning, threshold) on its training arguments only.
    Raises AssertionError if an outer test group also appears in the training portion.
    Returns per-clip scores / thresholds (NaN outside the folds) and one record per fold.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    outer_fold = np.asarray(outer_fold)
    scores = np.full(len(y), np.nan)
    thresholds = np.full(len(y), np.nan)
    records = []
    for k in sorted(int(f) for f in np.unique(outer_fold) if f >= 0):
        te = np.where(outer_fold == k)[0]
        tr = np.where((outer_fold >= 0) & (outer_fold != k))[0]
        shared = set(groups[tr].tolist()) & set(groups[te].tolist())
        assert not shared, f"outer fold {k}: {len(shared)} groups appear in both train and test"
        out = fit_predict(X[tr], y[tr], groups[tr], X[te], seed)
        p = np.asarray(out["p_test"], dtype=float)
        scores[te] = p
        thresholds[te] = float(out["threshold"])
        rec = {"fold": k, "n_train": int(len(tr)), "n_test": int(len(te)), "threshold": float(out["threshold"]),
               "auc": _auc_rank(y[te], p), "accuracy@0.5": float(((p >= 0.5) == (y[te] == 1)).mean()),
               "accuracy@fold_threshold": float(((p >= out["threshold"]) == (y[te] == 1)).mean())}
        rec.update({kk: vv for kk, vv in out.items() if kk not in ("p_test", "threshold")})
        records.append(rec)
    return {"scores": scores, "thresholds": thresholds, "folds": records}


def make_grid_search_fit_predict(make_estimator, param_grid, inner_splits=3):
    """Build a ``fit_predict`` that tunes by inner grouped CV (AUC), then refits and predicts.

    ``make_estimator(params, seed)`` returns an unfitted sklearn-style estimator. The decision
    threshold is the Youden cut of the inner out-of-fold predictions of the chosen parameters,
    so it never sees the outer test fold.
    """
    sk = _sklearn()

    def fit_predict(X_tr, y_tr, g_tr, X_te, seed):
        splitter = sk["StratifiedGroupKFold"](n_splits=inner_splits, shuffle=True, random_state=seed)
        inner = list(splitter.split(X_tr, y_tr, g_tr))
        best = None
        for params in param_grid:
            oof = np.zeros(len(y_tr))
            for a, b in inner:
                oof[b] = make_estimator(params, seed).fit(X_tr[a], y_tr[a]).predict_proba(X_tr[b])[:, 1]
            auc = _auc_rank(y_tr, oof)
            if best is None or auc > best[0]:
                best = (auc, params, oof)
        auc, params, oof = best
        model = make_estimator(params, seed).fit(X_tr, y_tr)
        return {
            "p_test": model.predict_proba(X_te)[:, 1],
            "threshold": choose_validation_threshold(y_tr, oof, method="youden"),
            "chosen_params": params, "inner_auc": float(auc),
        }

    return fit_predict


def make_random_search_fit_predict(make_estimator, param_space, n_configs=10, space_seed=1234):
    """Equal-budget model selection: ``n_configs`` random configurations, one shared inner split.

    Every model compared in Stage 5 gets the same budget: ``n_configs`` configurations sampled
    from its own ``param_space`` (dict name -> list of values), each trained on the same inner
    grouped 75/25 split of the outer-training clips and scored by AUC on the held-out quarter.
    The best configuration is refit on all outer-training clips; the decision threshold is the
    Youden cut of its inner held-out predictions, so the outer test fold is never consulted.
    ``make_estimator(params, seed)`` must return an object with ``fit`` and ``predict_proba``.
    """
    sk = _sklearn()
    keys = sorted(param_space)

    def fit_predict(X_tr, y_tr, g_tr, X_te, seed):
        rng = np.random.RandomState(space_seed)
        configs, seen = [], set()
        space_size = int(np.prod([len(param_space[k]) for k in keys]))
        while len(configs) < min(n_configs, space_size):
            params = {k: param_space[k][rng.randint(len(param_space[k]))] for k in keys}
            tag = tuple(str(params[k]) for k in keys)
            if tag not in seen:
                seen.add(tag)
                configs.append(params)
        splitter = sk["StratifiedGroupKFold"](n_splits=4, shuffle=True, random_state=seed)
        a, b = next(iter(splitter.split(X_tr, y_tr, g_tr)))
        best = None
        for params in configs:
            p = make_estimator(params, seed).fit(X_tr[a], y_tr[a]).predict_proba(X_tr[b])[:, 1]
            auc = _auc_rank(y_tr[b], p)
            if best is None or auc > best[0]:
                best = (auc, params, p)
        auc, params, p_inner = best
        model = make_estimator(params, seed).fit(X_tr, y_tr)
        out = {
            "p_test": model.predict_proba(X_te)[:, 1],
            "threshold": choose_validation_threshold(y_tr[b], p_inner, method="youden"),
            "chosen_params": params, "inner_auc": float(auc), "n_configs": len(configs),
        }
        if hasattr(model, "n_params_"):
            out["n_params"] = int(model.n_params_)
        return out

    return fit_predict


# ---------------------------------------------------------------------------
# Figure helpers (formerly quantum/plots.py). Deferred matplotlib/sklearn
# imports keep `import quantum.evaluation` fast for QAOA spawn workers and
# server restarts; only evaluation that renders figures pays the import cost.
# ---------------------------------------------------------------------------


def _plt():
    """Deferred matplotlib import (~0.76 s on this host)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def _plot_sklearn():
    """Deferred sklearn import (~2.4 s on this host)."""
    from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve  # noqa: PLC0415

    return ConfusionMatrixDisplay, roc_auc_score, roc_curve


def plot_roc_curve(y_true, prob_real, path):
    if len(set(int(v) for v in y_true)) < 2:
        return
    plt = _plt()
    ConfusionMatrixDisplay, roc_auc_score, roc_curve = _plot_sklearn()
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
    ConfusionMatrixDisplay, _, _ = _plot_sklearn()
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
