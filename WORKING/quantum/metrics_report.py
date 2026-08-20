"""Print or dump quantum-layer performance metrics.

Default (fast, stdlib-only): reads saved artifacts from output/quantum/.

    python -m quantum.metrics_report

Fresh recomputation from checkpoint:

    python -m quantum.metrics_report --live

Also write output/quantum/metrics_report.json:

    python -m quantum.metrics_report --json

Both flags compose freely.
"""

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from quantum.config import OUTPUT_DIR, DecisionConfig, VQCConfig

SEPARATOR = "=" * 72
THIN_SEP  = "-" * 72

LOSS_FMT = ".6f"
METRIC_FMT = ".4f"


# ──────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────

def _load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(val, fmt=METRIC_FMT, fallback="n/a"):
    if val is None:
        return fallback
    try:
        return f"{float(val):{fmt}}"
    except (TypeError, ValueError):
        return str(val)


def _section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def _sub(title: str):
    print(f"\n{THIN_SEP}")
    print(f"  {title}")
    print(THIN_SEP)


def _row(label, value, width=34):
    print(f"  {label:<{width}}{value}")


def _matrix_str(mat):
    if not mat or not mat[0]:
        return "[[?,?],[?,?]]"
    tn, fp = mat[0]
    fn, tp = mat[1]
    return f"[[{tn}, {fp}], [{fn}, {tp}]]"


def _fold_summary(cv_mean, cv_std, keys=None):
    if keys is None:
        keys = ["accuracy", "auc_roc", "balanced_accuracy", "f1", "specificity",
                 "precision", "recall", "pr_auc", "ece"]
    for k in keys:
        m = cv_mean.get(k)
        s = cv_std.get(k)
        if m is None:
            continue
        s_str = _fmt(s)
        print(f"    {k:<26s} {_fmt(m)} +/- {s_str}")


def _class_counts(y):
    from collections import Counter
    c = Counter(int(v) for v in y)
    return c.get(1, 0), c.get(0, 0)


# ──────────────────────────────────────────────────────────────────
#  Report sections
# ──────────────────────────────────────────────────────────────────

def section_dataset_artifacts(report):
    data_file = OUTPUT_DIR / "data.npz"
    if not data_file.exists():
        print(f"\n  [!] data.npz not found at {data_file}")
        return report
    data = dict(np.load(data_file))
    report["dataset"] = {}
    _section("DATASET (data.npz)")
    for split in ("train", "val", "test"):
        Xk = f"X_{split}"
        if Xk not in data:
            continue
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        n_real, n_fake = _class_counts(y)
        n_total = len(y)
        n_groups = len(set(data.get(f"groups_{split}", []).tolist()))
        ratio = f"{n_real/n_total:.1%}" if n_total else "n/a"
        _row(f"{split.upper()}:", f"{n_total} samples  (real={n_real}  fake={n_fake}  ratio={ratio})  groups={n_groups}")
        report["dataset"][split] = {
            "n": int(n_total), "real": int(n_real), "fake": int(n_fake),
            "real_ratio": float(n_real/n_total) if n_total else None,
            "groups": int(n_groups),
        }
    fn = data.get("feature_names", [])
    n_feat = len(fn) if hasattr(fn, '__len__') else "?"
    _row("Features:", f"{n_feat} total (QAOA selects subset)")
    report["dataset"]["n_features"] = int(n_feat) if n_feat != "?" else None
    return report


def section_qaoa_selection(report):
    path = OUTPUT_DIR / "qaoa_selection.json"
    if not path.exists():
        print(f"\n  [!] qaoa_selection.json not found")
        return report
    sel = _load_json(path)
    report["qaoa_selection"] = sel
    _section("QAOA FEATURE SELECTION")
    _row("Selected features:", ", ".join(sel.get("selected_features", [])))
    _row("QAOA cost:", _fmt(sel.get("cost")))
    _row("Converged:", "yes" if sel.get("success", False) else "no")
    restarts = sel.get("restarts", {})
    _row("Restarts (best of):", f"{restarts.get('n_restarts','?')} restarts, seed {restarts.get('chosen_seed','?')}")
    costs = restarts.get("all_costs", [])
    if costs:
        _row("Restart costs:", ", ".join(_fmt(c) for c in costs))
    selected_idx = sel.get("selected_indices", [])
    selected_names = sel.get("selected_features", [])
    weights = sel.get("feature_weights", [])
    print("\n  Feature selection weights (all 20):")
    from quantum.config import FEATURE_NAMES
    for i, fname in enumerate(FEATURE_NAMES):
        w = weights[i] if i < len(weights) else None
        marker = " <--" if i in selected_idx else ""
        print(f"    [{i:2d}] {fname:<34s} {_fmt(w)}{marker}")
    return report


def section_quantum_test_metrics(report):
    path = OUTPUT_DIR / "metrics_quantum.json"
    if not path.exists():
        print(f"\n  [!] metrics_quantum.json not found")
        return report
    mq = _load_json(path)
    report["quantum_test"] = mq
    m = mq.get("metrics", {})
    _section("QUANTUM MODEL - TEST SET")
    _row("Accuracy:", _fmt(m.get("accuracy")))
    _row("Precision:", _fmt(m.get("precision")))
    _row("Recall:", _fmt(m.get("recall")))
    _row("Specificity:", _fmt(m.get("specificity")))
    _row("F1:", _fmt(m.get("f1")))
    _row("AUC-ROC:", _fmt(m.get("auc_roc")))
    _row("PR-AUC:", _fmt(m.get("pr_auc")))
    _row("ECE:", _fmt(m.get("ece")))
    _row("Balanced accuracy:", _fmt(mq.get("balanced_accuracy")))
    _sub("Confusion matrix")
    cm = m.get("confusion_matrix", [[0,0],[0,0]])
    tn, fp = cm[0]
    fn, tp = cm[1]
    print(f"    Predicted ->")
    print(f"    Actual     FAKE   REAL")
    print(f"      REAL       {fn:5d}  {tp:5d}")
    print(f"      FAKE       {tn:5d}  {fp:5d}")
    return report


def section_decision_bins(report):
    path = OUTPUT_DIR / "metrics_quantum.json"
    if not path.exists():
        return report
    mq = _load_json(path)
    report.setdefault("quantum_test", mq)
    db = mq.get("decision_bins", {})
    _section("DECISION BINS (thresholds: <=0.3 -> FAKE, >=0.7 -> REAL)")
    _row("Real:", db.get("real", "n/a"))
    _row("Uncertain:", db.get("uncertain", "n/a"))
    _row("Fake:", db.get("fake", "n/a"))
    _row("Uncertain rate:", f"{db.get('uncertain_rate',0):.1%}")
    acc = db.get("confirmed_accuracy")
    _row("Confirmed accuracy:", _fmt(acc) if acc is not None else "n/a (0 confident)")
    return report


def section_threshold_diagnosis(report):
    path = OUTPUT_DIR / "threshold_analysis.json"
    if not path.exists():
        print(f"\n  [!] threshold_analysis.json not found")
        return report
    ta = _load_json(path)
    report["threshold_analysis"] = ta
    _section("THRESHOLD ANALYSIS")
    _row("Score range:", f"[{_fmt(ta.get('score_range', [None,None])[0])}, {_fmt(ta.get('score_range', [None,None])[1])}]")
    _row("Fake scores (max):", _fmt(ta.get("fake_scores")))
    _row("Real scores (max):", _fmt(ta.get("real_scores")))
    _row("Proportion fake <=0.3:", f"{ta.get('proportion_fake_below_03',0):.1%}")
    _row("Proportion real >=0.7:", f"{ta.get('proportion_real_above_07',0):.1%}")
    _row("Diagnosis:", ta.get("diagnosis", "n/a"))
    _row("Recommendation:", ta.get("recommendation", "n/a"))
    return report


def section_cv(report):
    path = OUTPUT_DIR / "metrics_quantum.json"
    if not path.exists():
        return report
    mq = _load_json(path)
    cv = mq.get("cv", {})
    if not cv:
        print(f"\n  [!] No cross-validation data in metrics_quantum.json")
        return report
    report.setdefault("quantum_test", mq)["cv"] = cv
    mean = cv.get("mean", {})
    std  = cv.get("std", {})
    _section("GROUPED 5-FOLD CROSS-VALIDATION (train split)")
    _row("Fold count:", 5)
    folds = cv.get("folds", [])
    _row("Test AUCs per fold:", ", ".join(_fmt(f.get("auc_roc")) for f in folds))
    print()
    _fold_summary(mean, std)
    return report


def section_baselines(report):
    path = OUTPUT_DIR / "metrics_baselines.json"
    if not path.exists():
        print(f"\n  [!] metrics_baselines.json not found")
        return report
    baselines = _load_json(path)
    report["baselines"] = baselines
    _section("CLASSICAL BASELINES (same selected features, test set)")
    # Get quantum test metrics for the comparison row
    q_path = OUTPUT_DIR / "metrics_quantum.json"
    q_metrics = {}
    if q_path.exists():
        q_all = _load_json(q_path)
        q_metrics = q_all.get("metrics", {})
        q_metrics["balanced_accuracy"] = q_all.get("balanced_accuracy", None)

    display_keys = [
        ("accuracy", "Acc"),
        ("precision", "Prec"),
        ("recall", "Rec"),
        ("specificity", "Spec"),
        ("f1", "F1"),
        ("auc_roc", "AUC"),
        ("pr_auc", "PR-AUC"),
        ("ece", "ECE"),
        ("balanced_accuracy", "Bal Acc"),
    ]

    # Header
    name_w = 20
    col_w = 8
    header = f"  {'Model':<{name_w}}" + "".join(f"{lbl:>{col_w}}" for _, lbl in display_keys)
    print(header)
    print("  " + "-" * (name_w + col_w * len(display_keys)))

    def _print_row(label, metrics_dict):
        vals = []
        for key, _ in display_keys:
            v = metrics_dict.get(key)
            if v is None:
                vals.append("n/a".rjust(col_w))
            else:
                vals.append(f"{float(v):.3f}".rjust(col_w))
        print(f"  {label:<{name_w}}" + "".join(vals))

    # Quantum row
    if q_metrics:
        _print_row("quantum_vqc", q_metrics)

    # Baseline rows
    for bname in ("random_forest", "logistic_regression", "gaussian_nb", "mlp", "linear_svc", "xgboost"):
        b = baselines.get(bname, {})
        if "skipped" in b:
            _print_row(f"{bname} [skipped]", {})
            continue
        _print_row(bname, b)

    # Baseline CV summary
    _sub("BASELINE CROSS-VALIDATION (5-fold, grouped)")
    header2 = f"  {'Model':<{name_w}}" + "".join(f"{lbl:>{col_w}}" for _, lbl in display_keys)
    print(header2)
    print("  " + "-" * (name_w + col_w * len(display_keys)))

    if q_metrics and "cv" in report.get("quantum_test", {}):
        cv_mean = report["quantum_test"]["cv"]["mean"]
        _print_row("quantum_vqc", cv_mean)

    for bname in ("random_forest", "logistic_regression", "gaussian_nb", "mlp", "linear_svc", "xgboost"):
        b = baselines.get(bname, {})
        cv = b.get("cv", {})
        if not cv or "mean" not in cv:
            continue
        _print_row(bname, cv["mean"])

    return report


def section_selection_comparison(report):
    path = OUTPUT_DIR / "selection_comparison.json"
    if not path.exists():
        return report
    sc = _load_json(path)
    report["selection_comparison"] = sc
    _section("SELECTION COMPARISON (QAOA vs classical)")
    _row("QAOA features:", ", ".join(sc.get("qaoa_features", [])))
    _row("Classical features:", ", ".join(sc.get("classical_features", [])))
    _row("Overlap:", f"{sc.get('overlap_count',0)} features")
    qaoa_only = sc.get("qaoa_only", [])
    class_only = sc.get("classical_only", [])
    if qaoa_only:
        _row("QAOA-only:", ", ".join(qaoa_only))
    if class_only:
        _row("Classical-only:", ", ".join(class_only))
    return report


# ──────────────────────────────────────────────────────────────────
#  --live recompute
# ──────────────────────────────────────────────────────────────────

def section_live_recompute(report):
    import numpy as np
    _section("LIVE RECOMPUTATION (data.npz + hybrid_vqc.pt)")

    from quantum.config import VQCConfig as _VC, DecisionConfig as _DC
    from quantum.data import load_dataset
    from quantum.evaluation import (
        classification_metrics,
        decision_bins,
        analyze_threshold_behavior,
        balanced_accuracy,
        expected_calibration_error,
        run_cv,
        _fit_vqc_fold,
    )
    from quantum.vqc import load_vqc_model, predict_vqc
    from quantum.scaling import FeatureScaler
    from quantum.qaoa import load_selection
    import functools

    vqc_cfg = _VC()
    dec_cfg = _DC()
    data = load_dataset()
    X_test, y_test = data["X_test"], data["y_test"]
    X_train, y_train = data["X_train"], data["y_train"]
    groups_train = data.get("groups_train", None)

    scaler = FeatureScaler().load()
    X_train_s = scaler.transform(data["X_train"])
    X_test_s  = scaler.transform(data["X_test"])

    selection = load_selection()
    indices = [int(i) for i in selection["selected_indices"]]
    n_feat = len(indices)

    X_train_sel = X_train_s[:, indices]
    X_test_sel  = X_test_s[:, indices]

    model = load_vqc_model(n_feat, vqc_cfg)
    prob_real = predict_vqc(model, X_test_sel)
    m = classification_metrics(y_test, prob_real)
    db = decision_bins(y_test, prob_real, dec_cfg)
    ta = analyze_threshold_behavior(y_test, prob_real, dec_cfg)
    ba = balanced_accuracy(y_test, prob_real)
    ece = expected_calibration_error(y_test, prob_real)

    # CV (on the scaled+selected train split)
    import torch
    gpu_host = torch.cuda.is_available()
    n_jobs = 1 if gpu_host else (os.cpu_count() or 1)
    cv_cfg = _VC(**{**vqc_cfg.__dict__, "save_checkpoint": False})
    cv = run_cv(
        functools.partial(_fit_vqc_fold, cfg=cv_cfg),
        X_train_sel, y_train,
        seed=vqc_cfg.seed,
        n_jobs=n_jobs,
        groups=groups_train,
    )

    live = {
        "metrics": m,
        "balanced_accuracy": ba,
        "decision_bins": db,
        "ece": ece,
        "threshold_analysis": ta,
        "cv": cv,
        "score_histogram": {
            "min": float(prob_real.min()),
            "max": float(prob_real.max()),
            "mean": float(prob_real.mean()),
            "std": float(prob_real.std()),
            "median": float(np.median(prob_real)),
        },
    }
    report["live_recomputed"] = live

    _row("Accuracy:", _fmt(m["accuracy"]))
    _row("Precision:", _fmt(m["precision"]))
    _row("Recall:", _fmt(m["recall"]))
    _row("Specificity:", _fmt(m["specificity"]))
    _row("F1:", _fmt(m["f1"]))
    _row("AUC-ROC:", _fmt(m["auc_roc"]))
    _row("PR-AUC:", _fmt(m["pr_auc"]))
    _row("ECE:", _fmt(m["ece"]))
    _row("Balanced accuracy:", _fmt(ba))

    _sub("Confusion matrix (live)")
    cm = m["confusion_matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]
    print(f"    Predicted ->")
    print(f"    Actual     FAKE   REAL")
    print(f"      REAL       {fn:5d}  {tp:5d}")
    print(f"      FAKE       {tn:5d}  {fp:5d}")

    _sub("Decision bins (live)")
    _row("Real:", db["real"])
    _row("Uncertain:", db["uncertain"])
    _row("Fake:", db["fake"])
    _row("Uncertain rate:", f"{db['uncertain_rate']:.1%}")

    _sub("Score distribution (live)")
    sh = live["score_histogram"]
    _row("Min:", _fmt(sh["min"]))
    _row("Max:", _fmt(sh["max"]))
    _row("Mean:", _fmt(sh["mean"]))
    _row("Std:", _fmt(sh["std"]))
    _row("Median:", _fmt(sh["median"]))

    _sub("Threshold diagnosis (live)")
    _row("Diagnosis:", ta.get("diagnosis", "n/a"))
    _row("Recommendation:", ta.get("recommendation", "n/a"))

    _sub("Grouped 5-fold CV (live)")
    _fold_summary(cv["mean"], cv["std"])
    folds = cv.get("folds", [])
    _row("Fold AUCs:", ", ".join(_fmt(f.get("auc_roc")) for f in folds))

    return report


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main():
    global np  # needed for section_dataset_artifacts / section_live_recompute

    parser = argparse.ArgumentParser(
        description="Quantum layer performance metrics report"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Recompute metrics from data.npz + hybrid_vqc.pt (slow, needs torch)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Also write the full report to output/quantum/metrics_report.json"
    )
    args = parser.parse_args()

    import numpy as np  # noqa: E402 — deferred; stdlib-only default mode still fast

    print(f"\n  Quantum Performance Metrics Report")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  {'=' * 60}")

    if not OUTPUT_DIR.exists():
        print(f"\n  [!] Output directory not found: {OUTPUT_DIR}")
        print(f"  Run first: python -m quantum.pipeline --all")
        sys.exit(1)

    report = {}

    report = section_dataset_artifacts(report)
    report = section_qaoa_selection(report)
    report = section_quantum_test_metrics(report)
    report = section_decision_bins(report)
    report = section_threshold_diagnosis(report)
    report = section_cv(report)
    report = section_baselines(report)
    report = section_selection_comparison(report)

    if args.live:
        try:
            report = section_live_recompute(report)
        except Exception as exc:
            print(f"\n  [!] Live recomputation failed: {exc}")
            print(f"  Ensure torch is installed and data.npz / hybrid_vqc.pt exist.")

    # --json
    if args.json:
        out_path = OUTPUT_DIR / "metrics_report.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\n  Report written to: {out_path}")

    print(f"\n{SEPARATOR}")
    print(f"  Done.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
