import argparse
import json
import torch
from dataclasses import asdict

import numpy as np

from quantum.config import (
    DataConfig,
    DecisionConfig,
    FEATURE_NAMES,
    OUTPUT_DIR,
    QAOASelectionConfig,
    VQCConfig,
)

_vqc_cfg = VQCConfig()
from quantum.data import build_dataset, load_dataset
from quantum.evaluation import evaluate_quantum_model, run_baselines
from quantum.scaling import FeatureScaler, SCALER_FILE
from quantum.qaoa import (
    QAOASelector,
    compare_selections,
    load_selection,
    save_selection,
    select_classical,
    verify_hamiltonian,
)
from quantum.vqc import load_vqc_model, predict_vqc, train_vqc


def build_parser():
    parser = argparse.ArgumentParser(
        description="Quantum decision layer: QAOA feature selection + hybrid VQC classification."
    )
    parser.add_argument("--build-data", action="store_true", help="Build data.npz from the real rPPG feature table")
    parser.add_argument("--select", action="store_true", help="Run QAOA feature selection")
    parser.add_argument("--train", action="store_true", help="Train the hybrid VQC")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained VQC")
    parser.add_argument("--baselines", action="store_true", help="Train classical baselines")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline in order")
    parser.add_argument(
        "--dev-only",
        action="store_true",
        help="Development mode: evaluate on the validation split, never the final test split. "
        "Use a normal run (no flag) only to produce the one frozen test-set evaluation.",
    )
    return parser


def predict_features(features):
    """Inference entry point for the quantum decision layer.

    Reuses the saved training-time artifacts so inference sees exactly
    the training-time transformation:
        train-fitted FeatureScaler -> QAOA-selected indices ->
        trained hybrid VQC -> P(real) -> KYC verdict (REAL/FAKE/UNCERTAIN).

    `features` must be a dict keyed by the 10 rPPG feature names
    (FEATURE_NAMES order).
    """
    x = np.asarray([[float(features[name]) for name in FEATURE_NAMES]], dtype=np.float64)
    scaler = FeatureScaler(FEATURE_NAMES).load()
    if scaler.mean_.shape[0] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"feature_scaler.json is out of sync: expected {len(FEATURE_NAMES)} features, "
            f"found {scaler.mean_.shape[0]}. Rerun `python -m quantum.pipeline --all` from WORKING/."
        )
    x_scaled = scaler.transform(x)
    if not np.isfinite(x_scaled).all():
        return {
            "prob_real": None,
            "verdict": "INCONCLUSIVE",
            "confidence": None,
            "reason": "rPPG feature vector contains non-finite values after scaling",
            "selected_features": None,
            "selected_indices": None,
            "scaler_file": str(SCALER_FILE),
        }
    selection = load_selection()
    indices = [int(i) for i in selection["selected_indices"]]
    if not indices or any(i < 0 or i >= len(FEATURE_NAMES) for i in indices):
        raise RuntimeError(
            f"qaoa_selection.json indices out of range for {len(FEATURE_NAMES)} features: "
            f"{indices}. Rerun `python -m quantum.pipeline --all` from WORKING/."
        )
    try:
        model = load_vqc_model(len(indices))
    except Exception as exc:
        raise RuntimeError(
            f"hybrid_vqc.pt incompatible with the QAOA selection "
            f"({len(indices)} features): {type(exc).__name__}: {exc}. "
            "Rerun `python -m quantum.pipeline --all` from WORKING/."
        ) from exc

    # Load optimal threshold from checkpoint metadata
    ckpt = torch.load(_vqc_cfg.checkpoint_file, map_location="cpu", weights_only=False)
    opt_threshold = ckpt.get("metadata", {}).get("decision_threshold", DecisionConfig().decision_threshold)
    prob_real = float(predict_vqc(model, x_scaled[:, indices])[0])

    if prob_real >= opt_threshold:
        verdict = "REAL"
    else:
        verdict = "FAKE"
    return {
        "prob_real": prob_real,
        "verdict": verdict,
        "confidence": round(prob_real, 6),
        "selected_features": selection["selected_features"],
        "selected_indices": indices,
        "scaler_file": str(SCALER_FILE),
        "decision_threshold": opt_threshold,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    data_cfg = DataConfig()
    qaoa_cfg = QAOASelectionConfig()
    vqc_cfg = VQCConfig()
    decision_cfg = DecisionConfig()

    if not any(
        [
            args.build_data,
            args.select,
            args.train,
            args.evaluate,
            args.baselines,
            args.all,
        ]
    ):
        parser.print_help()
        return 2

    dev_only = bool(getattr(args, "dev_only", False))
    if dev_only:
        print("=" * 72)
        print("  DEV-ONLY MODE: all evaluation below runs on the VALIDATION split.")
        print("  The final TEST split is isolated and touched only by a normal")
        print("  run (no --dev-only) once a configuration is frozen.")
        print("=" * 72)

    if args.build_data or args.all:
        print("[1/6] Building dataset from rPPG layer output...")
        build_dataset(data_cfg)
    data = load_dataset(data_cfg.data_file)
    for key in ("X_train", "X_val", "X_test"):
        print(f"  {key}: {data[key].shape}")

    scaler = FeatureScaler(FEATURE_NAMES).fit(data["X_train"])
    scaler.save()
    X_train = scaler.transform(data["X_train"])
    X_val = scaler.transform(data["X_val"])
    X_test = scaler.transform(data["X_test"])
    print(f"  Scaler fitted on train only (z-score), saved: {SCALER_FILE}")

    selection = None
    if args.select or args.all:
        print("[2/6] Running QAOA feature selection (train split only)...")
        from quantum.qaoa import simulator_device

        # Pre-select features classically if >20 (state vector sim limit)
        pre_select_k = 18
        if len(FEATURE_NAMES) > pre_select_k:
            print(f"  Pre-selecting {pre_select_k} features via classical AUC (from {len(FEATURE_NAMES)})")
            classical_pre = select_classical(X_train, data["y_train"], qaoa_cfg)
            classical_pre = classical_pre["selected_indices"]
            pre_select_k = min(pre_select_k, len(classical_pre))
            classical_pre = classical_pre[:pre_select_k]
            X_train_qaoa = X_train[:, classical_pre]
            feat_names_qaoa = [FEATURE_NAMES[i] for i in classical_pre]
            print(f"  Pre-selected: {feat_names_qaoa}")
        else:
            X_train_qaoa = X_train
            feat_names_qaoa = FEATURE_NAMES
            classical_pre = None

        qaoa_dev, qaoa_backend = simulator_device(len(feat_names_qaoa), qaoa_cfg)
        print(
            f"  QAOA simulator: {qaoa_backend}"
            f"{' (' + str(getattr(qaoa_dev, 'short_name', '')) + ')' if qaoa_dev else ''}"
        )
        error = verify_hamiltonian(X_train_qaoa, data["y_train"], qaoa_cfg)
        assert error < 1e-6, (
            f"Hamiltonian verification FAILED (max error {error:.2e}): "
            "_cost_terms does not reproduce _classical_cost"
        )
        print(f"  Hamiltonian verification OK (max error {error:.2e})")
        selection = QAOASelector(qaoa_cfg).select(X_train_qaoa, data["y_train"])
        save_selection(selection, qaoa_cfg.selection_file)
        # Map QAOA indices back to original feature space
        if classical_pre is not None:
            selection["selected_indices"] = [classical_pre[i] for i in selection["selected_indices"]]
            selection["selected_features"] = [FEATURE_NAMES[i] for i in selection["selected_indices"]]
        print(
            f"  Selected {len(selection['selected_features'])} features: {selection['selected_features']}"
        )
        restarts = selection.get("restarts", {})
        if restarts:
            print(
                f"  QAOA: {restarts['n_restarts']} parallel restarts, "
                f"chosen seed {restarts['chosen_seed']} (cost {selection['cost']:.4f})"
            )

        classical = select_classical(X_train, data["y_train"], qaoa_cfg)
        comparison = compare_selections(selection, classical)
        comparison_file = OUTPUT_DIR / "selection_comparison.json"
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        with open(comparison_file, "w") as fh:
            json.dump(comparison, fh, indent=2)
        print(f"  Classical AUC-greedy reference: {classical['selected_features']}")
        print(f"  Overlap with QAOA: {comparison['overlap_count']} features (see {comparison_file})")
    else:
        selection = load_selection(qaoa_cfg.selection_file)

    indices = [int(i) for i in selection["selected_indices"]]
    if args.train or args.all:
        print(f"[3/6] Training hybrid VQC on {len(indices)} QAOA-selected features...")
        from quantum.vqc import qnode_backend_name, resolve_device

        print(
            f"  Torch device: {resolve_device()} | QNode simulator: {qnode_backend_name(vqc_cfg)}"
        )
        train_vqc(
            X_train[:, indices],
            data["y_train"].astype(float),
            vqc_cfg,
            X_val=X_val[:, indices],
            y_val=data["y_val"].astype(float),
            metadata={
                "selected_indices": indices,
                "selected_features": selection["selected_features"],
                "feature_names": FEATURE_NAMES,
                "n_features": len(indices),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "qaoa_config": asdict(qaoa_cfg),
                "vqc_config": asdict(vqc_cfg),
                "decision_config": asdict(decision_cfg),
                "qaoa_selection": selection,
            },
        )
        print(f"  Checkpoint (with metadata): {vqc_cfg.checkpoint_file}")

    # Compute and save optimal threshold using Youden's J on validation set
    if args.train or args.all:
        print("  Computing optimal threshold (Youden's J) on validation set...")
        from quantum.evaluation import optimal_threshold_youden
        val_probs = predict_vqc(load_vqc_model(len(indices)), X_val[:, indices])
        opt_threshold = optimal_threshold_youden(data["y_val"].astype(int), val_probs)
        print(f"    Optimal threshold: {opt_threshold:.6f}")
        # Update checkpoint metadata with optimal threshold
        ckpt = torch.load(vqc_cfg.checkpoint_file, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "metadata" in ckpt:
            ckpt["metadata"]["decision_threshold"] = float(opt_threshold)
            torch.save(ckpt, vqc_cfg.checkpoint_file)
            print(f"    Saved optimal threshold to checkpoint")

    if args.evaluate or args.all:
        eval_X = X_val[:, indices] if dev_only else X_test[:, indices]
        eval_y = data["y_val"] if dev_only else data["y_test"]
        print(
            f"[4/6] Evaluating quantum model on {'VAL' if dev_only else 'TEST'} split..."
        )
        result = evaluate_quantum_model(
            eval_X,
            eval_y,
            vqc_cfg,
            decision_cfg,
            X_train=X_train[:, indices],
            y_train=data["y_train"],
            groups_train=data["groups_train"],
        )
        metrics = result["metrics"]
        print(
            f"  accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"auc={metrics['auc_roc']:.4f} ece={metrics['ece']:.4f}"
        )
        if "cv" in result:
            cv = result["cv"]["mean"]
            print(
                f"  CV(5-fold): accuracy={cv['accuracy']:.4f}+-{result['cv']['std']['accuracy']:.4f} "
                f"balanced_acc={cv['balanced_accuracy']:.4f} auc={cv['auc_roc']:.4f}"
            )
        print(f"  decision bins: {result['decision_bins']}")

    if args.baselines or args.all:
        eval_X = X_val[:, indices] if dev_only else X_test[:, indices]
        eval_y = data["y_val"] if dev_only else data["y_test"]
        print(
            f"[5/6] Training classical baselines ({'VAL' if dev_only else 'TEST'} evaluation)..."
        )
        results = run_baselines(
            X_train[:, indices],
            data["y_train"],
            eval_X,
            eval_y,
            decision_cfg,
            groups_train=data["groups_train"],
        )
        for name, metrics in results.items():
            print(
                f"  {name}: accuracy={metrics['accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} auc={metrics['auc_roc']:.4f}"
            )
            if "cv" in metrics:
                cv = metrics["cv"]["mean"]
                print(
                    f"    CV(5-fold): balanced_acc={cv['balanced_accuracy']:.4f}+-{metrics['cv']['std']['balanced_accuracy']:.4f}"
                )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())