import argparse
import json
import torch
from dataclasses import asdict
from pathlib import Path

import numpy as np

from quantum.config import (
    DataConfig,
    DecisionConfig,
    FEATURE_NAMES,  # RPPG_FEATURE_NAMES alias
    RPPG_FEATURE_NAMES,
    VISUAL_FEATURE_NAMES,
    FUSED_FEATURE_NAMES,
    OUTPUT_DIR,
    QAOASelectionConfig,
    VQCConfig,
)

_vqc_cfg = VQCConfig()
from quantum.data import build_dataset, load_dataset, FEATURE_SETS
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
from quantum.explain import build_explanation, DecisionExplanation, Verdict, PhysiologicalEvidence, ReliabilityLevel


def _fmt4(v):
    return "n/a" if v is None else f"{v:.4f}"


# Only fused mode is supported
FEATURE_SET_CONFIGS = {
    "fused": {
        "feature_names": FUSED_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_fused.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_fused.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_fused.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_fused.pt",
    },
}


def get_feature_set_config(feature_set: str):
    """Get configuration for a feature set."""
    if feature_set not in FEATURE_SET_CONFIGS:
        raise ValueError(f"Unknown feature set: {feature_set}. Choose from: {list(FEATURE_SET_CONFIGS.keys())}")
    return FEATURE_SET_CONFIGS[feature_set]


def predict_features(features, feature_set: str = "rppg_only"):
    """Inference entry point for the quantum decision layer.

    Reuses the saved training-time artifacts so inference sees exactly
    the training-time transformation:
        train-fitted FeatureScaler -> QAOA-selected indices ->
        trained hybrid VQC -> P(real) -> KYC verdict (REAL/FAKE/INSUFFICIENT EVIDENCE / REVIEW REQUIRED).

    `features` must be a dict keyed by the feature names for the specified feature_set.

    Returns a DecisionExplanation with full explainable output.
    """
    cfg = get_feature_set_config(feature_set)
    feature_names = cfg["feature_names"]
    scaler_file = cfg["scaler_file"]
    selection_file = cfg["selection_file"]
    checkpoint_file = cfg["checkpoint_file"]

    x = np.asarray([[float(features[name]) for name in feature_names]], dtype=np.float64)
    scaler = FeatureScaler(feature_names).load(scaler_file)
    if scaler.mean_.shape[0] != len(feature_names):
        raise RuntimeError(
            f"feature_scaler.json is out of sync: expected {len(feature_names)} features, "
            f"found {scaler.mean_.shape[0]}. Rerun training for feature_set={feature_set}."
        )
    x_scaled = scaler.transform(x)
    if not np.isfinite(x_scaled).all():
        # Default to FAKE when features are invalid (binary decision)
        return DecisionExplanation(
            final_verdict=Verdict.FAKE,
            physiological_evidence=PhysiologicalEvidence(
                verdict=Verdict.FAKE,
                probability_real=0.0,
                confidence=0.0,
                reliability_level=ReliabilityLevel.UNKNOWN,
            ),
            decision_logic="Feature vector contains non-finite values after scaling",
            requires_verification=True,
            coverage=0.0,
            overall_confidence=0.0,
        ).to_dict()
    selection = load_selection(selection_file)
    indices = [int(i) for i in selection["selected_indices"]]
    if not indices or any(i < 0 or i >= len(feature_names) for i in indices):
        raise RuntimeError(
            f"qaoa_selection.json indices out of range for {len(feature_names)} features: "
            f"{indices}. Rerun training for feature_set={feature_set}."
        )
    try:
        model = load_vqc_model(len(indices), VQCConfig(checkpoint_file=checkpoint_file))
    except Exception as exc:
        raise RuntimeError(
            f"{checkpoint_file.name} incompatible with the QAOA selection "
            f"({len(indices)} features): {type(exc).__name__}: {exc}. "
            f"Rerun training for feature_set={feature_set}."
        ) from exc

    # Load optimal threshold from checkpoint metadata
    ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    opt_threshold = ckpt.get("metadata", {}).get("decision_threshold", DecisionConfig().decision_threshold)
    prob_real = float(predict_vqc(model, x_scaled[:, indices])[0])
    
    # Build explainable output using Phase 11 explain module
    from quantum.explain import build_explanation
    decision_cfg = DecisionConfig()
    return build_explanation(
        prob_real=prob_real,
        decision_cfg=decision_cfg,
    )


def run_pipeline_for_feature_set(
    feature_set: str,
    data_cfg: DataConfig,
    qaoa_cfg: QAOASelectionConfig,
    vqc_cfg: VQCConfig,
    decision_cfg: DecisionConfig,
    dev_only: bool = False,
    csv_file=None,
    run_qaoa: bool = True,
    run_train: bool = True,
    run_eval: bool = True,
    run_baselines_flag: bool = True,
):
    """Run the full quantum pipeline for a specific feature set."""
    print(f"\n{'='*72}")
    print(f"  RUNNING PIPELINE FOR FEATURE SET: {feature_set.upper()}")
    print(f"{'='*72}")

    fc = get_feature_set_config(feature_set)
    feature_names = fc["feature_names"]

    # Override config paths for this feature set
    qaoa_cfg = QAOASelectionConfig(
        **{**asdict(qaoa_cfg), "selection_file": fc["selection_file"]}
    )
    vqc_cfg = VQCConfig(
        **{**asdict(vqc_cfg), "checkpoint_file": fc["checkpoint_file"]}
    )

    # Build/load dataset
    print(f"[1/6] Building dataset for {feature_set}...")
    data = build_dataset(cfg=data_cfg, feature_set=feature_set, csv_file=csv_file)
    for key in ("X_train", "X_val", "X_test"):
        print(f"  {key}: {data[key].shape}")

    # Fit scaler
    scaler = FeatureScaler(feature_names).fit(data["X_train"])
    scaler.save(fc["scaler_file"])
    X_train = scaler.transform(data["X_train"])
    X_val = scaler.transform(data["X_val"])
    X_test = scaler.transform(data["X_test"])
    print(f"  Scaler fitted on train only (z-score), saved: {fc['scaler_file']}")

    # QAOA feature selection
    selection = None
    if run_qaoa:
        print(f"[2/6] Running QAOA feature selection (train split only)...")
        from quantum.qaoa import simulator_device

        # Pre-select features classically if >20 (state vector sim limit)
        pre_select_k = 18
        if len(feature_names) > pre_select_k:
            print(f"  Pre-selecting {pre_select_k} features via classical AUC (from {len(feature_names)})")
            classical_pre = select_classical(X_train, data["y_train"], qaoa_cfg, feature_names=feature_names)
            classical_pre = classical_pre["selected_indices"]
            pre_select_k = min(pre_select_k, len(classical_pre))
            classical_pre = classical_pre[:pre_select_k]
            X_train_qaoa = X_train[:, classical_pre]
            feat_names_qaoa = [feature_names[i] for i in classical_pre]
            print(f"  Pre-selected: {feat_names_qaoa}")
        else:
            X_train_qaoa = X_train
            feat_names_qaoa = feature_names
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
            selection["selected_features"] = [feature_names[i] for i in selection["selected_indices"]]
        print(
            f"  Selected {len(selection['selected_features'])} features: {selection['selected_features']}"
        )
        restarts = selection.get("restarts", {})
        if restarts:
            print(
                f"  QAOA: {restarts['n_restarts']} parallel restarts, "
                f"chosen seed {restarts['chosen_seed']} (cost {selection['cost']:.4f})"
            )

        classical = select_classical(X_train, data["y_train"], qaoa_cfg, feature_names=feature_names)
        comparison = compare_selections(selection, classical, feature_names=feature_names)
        comparison_file = OUTPUT_DIR / f"selection_comparison_{feature_set}.json"
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        with open(comparison_file, "w") as fh:
            json.dump(comparison, fh, indent=2)
        print(f"  Classical AUC-greedy reference: {classical['selected_features']}")
        print(f"  Overlap with QAOA: {comparison['overlap_count']} features (see {comparison_file})")
    else:
        selection = load_selection(qaoa_cfg.selection_file)

    indices = [int(i) for i in selection["selected_indices"]]

    # Train VQC
    if run_train:
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
                "feature_names": feature_names,
                "n_features": len(indices),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "qaoa_config": asdict(qaoa_cfg),
                "vqc_config": asdict(vqc_cfg),
                "decision_config": asdict(decision_cfg),
                "qaoa_selection": selection,
                "feature_set": feature_set,
            },
        )
        print(f"  Checkpoint (with metadata): {vqc_cfg.checkpoint_file}")

    # Compute and save optimal threshold using Youden's J on validation set
    if run_train:
        print("  Computing optimal threshold (Youden's J) on validation set...")
        from quantum.evaluation import optimal_threshold_youden
        val_probs = predict_vqc(load_vqc_model(len(indices), vqc_cfg), X_val[:, indices])
        opt_threshold = optimal_threshold_youden(data["y_val"].astype(int), val_probs)
        print(f"    Optimal threshold: {opt_threshold:.6f}")
        # Update checkpoint metadata with optimal threshold
        ckpt = torch.load(vqc_cfg.checkpoint_file, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "metadata" in ckpt:
            ckpt["metadata"]["decision_threshold"] = float(opt_threshold)
            torch.save(ckpt, vqc_cfg.checkpoint_file)
            print(f"    Saved optimal threshold to checkpoint")

    # Evaluate
    eval_results = None
    if run_eval:
        eval_X = X_val[:, indices] if dev_only else X_test[:, indices]
        eval_y = data["y_val"] if dev_only else data["y_test"]
        print(
            f"[4/6] Evaluating quantum model on {'VAL' if dev_only else 'TEST'} split..."
        )
        
        # Compute PQS for evaluation data if quality_threshold is enabled
        eval_pqs = None
        if decision_cfg.quality_threshold > 0:
            # In a full implementation, we'd compute PQS for each eval sample
            # For now, we'll rely on the prob_real thresholds
            eval_pqs = None
        
        eval_results = evaluate_quantum_model(
            eval_X,
            eval_y,
            vqc_cfg,
            decision_cfg,
            X_train=X_train[:, indices],
            y_train=data["y_train"],
            groups_train=data["groups_train"],
            pqs=eval_pqs,
        )
        metrics = eval_results["metrics"]
        print(
            f"  accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"auc={_fmt4(metrics['auc_roc'])} ece={metrics['ece']:.4f}"
        )
        if "cv" in eval_results:
            cv = eval_results["cv"]["mean"]
            print(
                f"  CV(5-fold): accuracy={cv['accuracy']:.4f}+-{eval_results['cv']['std']['accuracy']:.4f} "
                f"balanced_acc={cv['balanced_accuracy']:.4f} auc={_fmt4(cv['auc_roc'])}"
            )
        print(f"  decision bins: {eval_results['decision_bins']}")

    # Classical baselines
    baseline_results = None
    if run_baselines_flag:
        eval_X = X_val[:, indices] if dev_only else X_test[:, indices]
        eval_y = data["y_val"] if dev_only else data["y_test"]
        print(
            f"[5/6] Training classical baselines ({'VAL' if dev_only else 'TEST'} evaluation)..."
        )
        baseline_results = run_baselines(
            X_train[:, indices],
            data["y_train"],
            eval_X,
            eval_y,
            decision_cfg,
            groups_train=data["groups_train"],
        )
        for name, metrics in baseline_results.items():
            print(
                f"  {name}: accuracy={metrics['accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} auc={_fmt4(metrics['auc_roc'])}"
            )
            if "cv" in metrics:
                cv = metrics["cv"]["mean"]
                print(
                    f"    CV(5-fold): balanced_acc={cv['balanced_accuracy']:.4f}+-{metrics['cv']['std']['balanced_accuracy']:.4f}"
                )

    return {
        "feature_set": feature_set,
        "selection": selection,
        "eval_results": eval_results,
        "baseline_results": baseline_results,
        "indices": indices,
        "scaler_file": str(fc["scaler_file"]),
        "checkpoint_file": str(fc["checkpoint_file"]),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Quantum decision layer: QAOA feature selection + hybrid VQC classification (fused mode only)."
    )
    parser.add_argument("--build-data", action="store_true", help="Build data.npz from the feature table")
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
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Path to visual features CSV file (for fused feature set)",
    )
    return parser


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
        print("  run (no flag) only to produce the one frozen test-set evaluation.")
        print("=" * 72)

    csv_file = Path(args.csv_file) if args.csv_file else None

    # Only fused feature set is supported
    feature_set = "fused"
    run_qaoa = args.select or args.all
    run_train = args.train or args.all
    run_eval = args.evaluate or args.all
    run_baselines_flag = args.baselines or args.all

    run_pipeline_for_feature_set(
        feature_set=feature_set,
        data_cfg=data_cfg,
        qaoa_cfg=qaoa_cfg,
        vqc_cfg=vqc_cfg,
        decision_cfg=decision_cfg,
        dev_only=dev_only,
        csv_file=csv_file,
        run_qaoa=run_qaoa,
        run_train=run_train,
        run_eval=run_eval,
        run_baselines_flag=run_baselines_flag,
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())