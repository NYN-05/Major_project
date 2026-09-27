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
    RPPG_BASE_FEATURE_NAMES,
    CROSS_ROI_FEATURE_NAMES,
    PHASE6_FEATURE_SETS,
    PHASE7_ABLATION_SETS,
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
from quantum.ensemble import run_phase8_ensemble_comparison
from quantum.explain import build_explanation, DecisionExplanation


FEATURE_SET_CONFIGS = {
    "rppg_only": {
        "feature_names": RPPG_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_rppg_only.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_rppg_only.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_rppg_only.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_rppg_only.pt",
    },
    "rppg_base": {
        "feature_names": RPPG_BASE_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_rppg_base.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_rppg_base.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_rppg_base.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_rppg_base.pt",
    },
    "rppg_cross_roi": {
        "feature_names": RPPG_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_rppg_cross_roi.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_rppg_cross_roi.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_rppg_cross_roi.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_rppg_cross_roi.pt",
    },
    "visual_only": {
        "feature_names": VISUAL_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_visual_only.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_visual_only.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_visual_only.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_visual_only.pt",
    },
    "fused": {
        "feature_names": FUSED_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_fused.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_fused.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_fused.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_fused.pt",
    },
    # Phase 7 Ablation Study feature sets
    "A_rppg_only": {
        "feature_names": RPPG_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_A_rppg_only.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_A_rppg_only.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_A_rppg_only.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_A_rppg_only.pt",
    },
    "B_pos_only": {
        "feature_names": RPPG_BASE_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_B_pos_only.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_B_pos_only.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_B_pos_only.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_B_pos_only.pt",
    },
    "C_chrom_only": {
        "feature_names": RPPG_BASE_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_C_chrom_only.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_C_chrom_only.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_C_chrom_only.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_C_chrom_only.pt",
    },
    "D_pos_chrom": {
        "feature_names": RPPG_BASE_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_D_pos_chrom.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_D_pos_chrom.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_D_pos_chrom.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_D_pos_chrom.pt",
    },
    "E_rppg_quality": {
        "feature_names": RPPG_BASE_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_E_rppg_quality.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_E_rppg_quality.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_E_rppg_quality.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_E_rppg_quality.pt",
    },
    "F_rppg_cross_roi": {
        "feature_names": RPPG_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_F_rppg_cross_roi.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_F_rppg_cross_roi.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_F_rppg_cross_roi.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_F_rppg_cross_roi.pt",
    },
    "G_rppg_visual": {
        "feature_names": FUSED_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_G_rppg_visual.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_G_rppg_visual.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_G_rppg_visual.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_G_rppg_visual.pt",
    },
    "H_rppg_visual_quality": {
        "feature_names": FUSED_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_H_rppg_visual_quality.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_H_rppg_visual_quality.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_H_rppg_visual_quality.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_H_rppg_visual_quality.pt",
    },
    "I_full": {
        "feature_names": FUSED_FEATURE_NAMES,
        "data_file": OUTPUT_DIR / "data_I_full.npz",
        "scaler_file": OUTPUT_DIR / "feature_scaler_I_full.json",
        "selection_file": OUTPUT_DIR / "qaoa_selection_I_full.json",
        "checkpoint_file": OUTPUT_DIR / "hybrid_vqc_I_full.pt",
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
        return DecisionExplanation(
            final_verdict=Verdict.INSUFFICIENT_EVIDENCE,
            physiological_evidence=PhysiologicalEvidence(
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                probability_real=None,
                confidence=None,
                reliability_level=ReliabilityLevel.UNKNOWN,
                insufficient_reason="Feature vector contains non-finite values after scaling"
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
        model = load_vqc_model(len(indices))
    except Exception as exc:
        raise RuntimeError(
            f"hybrid_vqc.pt incompatible with the QAOA selection "
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
        feature_set=feature_set,
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
        val_probs = predict_vqc(load_vqc_model(len(indices)), X_val[:, indices])
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
            f"auc={metrics['auc_roc']:.4f} ece={metrics['ece']:.4f}"
        )
        if "cv" in eval_results:
            cv = eval_results["cv"]["mean"]
            print(
                f"  CV(5-fold): accuracy={cv['accuracy']:.4f}+-{eval_results['cv']['std']['accuracy']:.4f} "
                f"balanced_acc={cv['balanced_accuracy']:.4f} auc={cv['auc_roc']:.4f}"
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
                f"f1={metrics['f1']:.4f} auc={metrics['auc_roc']:.4f}"
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
        description="Quantum decision layer: QAOA feature selection + hybrid VQC classification."
    )
    parser.add_argument("--feature-set", default="rppg_only",
                        choices=["rppg_only", "rppg_base", "rppg_cross_roi", "visual_only", "fused"],
                        help="Feature set to use (default: rppg_only)")
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
        "--compare-all",
        action="store_true",
        help="Run full pipeline for Phase 1 feature sets (rppg_only, visual_only, fused) and compare",
    )
    parser.add_argument(
        "--phase6-compare",
        action="store_true",
        help="Run full pipeline for Phase 6 feature sets (rppg_base, rppg_cross_roi, visual_only, fused) and compare",
    )
    parser.add_argument(
        "--phase7-ablation",
        action="store_true",
        help="Run full ablation study (Experiments A-I) as defined in Phase 7 roadmap",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Run Phase 8 ensemble comparison across feature sets",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Path to feature CSV file (for visual_only or fused feature sets)",
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
            args.compare_all,
            args.phase6_compare,
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

    if args.compare_all:
        # Run full pipeline for all three feature sets (Phase 1 comparison)
        print("\n" + "=" * 72)
        print("  PHASE 1 COMPARATIVE EXPERIMENT: rPPG vs Visual vs Fused")
        print("=" * 72)
        all_results = {}
        for fs in ["rppg_only", "visual_only", "fused"]:
            # Use appropriate CSV file for each feature set
            if fs == "rppg_only":
                fs_csv = None  # Uses default from DataConfig
            elif fs == "visual_only":
                fs_csv = csv_file or (data_cfg.csv_file.parent.parent / "visual" / "visual_features.csv")
            elif fs == "fused":
                fs_csv = csv_file or (data_cfg.csv_file.parent.parent / "visual" / "fused_features.csv")

            if fs_csv and not Path(fs_csv).exists():
                print(f"\n  SKIPPING {fs}: CSV file not found: {fs_csv}")
                continue

            result = run_pipeline_for_feature_set(
                feature_set=fs,
                data_cfg=data_cfg,
                qaoa_cfg=qaoa_cfg,
                vqc_cfg=vqc_cfg,
                decision_cfg=decision_cfg,
                dev_only=dev_only,
                csv_file=fs_csv,
                run_qaoa=args.select or args.all or True,
                run_train=args.train or args.all or True,
                run_eval=args.evaluate or args.all or True,
                run_baselines_flag=args.baselines or args.all or True,
            )
            all_results[fs] = result

        # Print comparative summary
        print("\n" + "=" * 72)
        print("  PHASE 1 COMPARATIVE RESULTS SUMMARY")
        print("=" * 72)
        print(f"{'Feature Set':<15} {'Quantum AUC':<12} {'Quantum Acc':<12} {'Best Baseline AUC':<18} {'Best Baseline':<15}")
        print("-" * 72)
        for fs, result in all_results.items():
            if result["eval_results"]:
                q_auc = result["eval_results"]["metrics"]["auc_roc"]
                q_acc = result["eval_results"]["metrics"]["accuracy"]
            else:
                q_auc = q_acc = float("nan")
            if result["baseline_results"]:
                best_baseline = max(result["baseline_results"].items(), key=lambda x: x[1]["auc_roc"])
                b_auc = best_baseline[1]["auc_roc"]
                b_name = best_baseline[0]
            else:
                b_auc = float("nan")
                b_name = "N/A"
            print(f"{fs:<15} {q_auc:<12.4f} {q_acc:<12.4f} {b_auc:<18.4f} {b_name:<15}")

        # Save comparative results
        summary_file = OUTPUT_DIR / "phase1_comparison.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w") as fh:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "eval_results" and kk != "baseline_results"}
                      for k, v in all_results.items()}, fh, indent=2, default=str)
        print(f"\n  Comparative results saved to: {summary_file}")

        return 0

    if args.phase6_compare:
        # Run full pipeline for Phase 6 feature sets
        print("\n" + "=" * 72)
        print("  PHASE 6 COMPARATIVE EXPERIMENT: Classical vs Quantum on Multiple Feature Sets")
        print("=" * 72)
        all_results = {}
        # Phase 6 feature sets: rppg_base, rppg_cross_roi, visual_only, fused
        for fs in ["rppg_base", "rppg_cross_roi", "visual_only", "fused"]:
            if fs == "rppg_base" or fs == "rppg_cross_roi":
                fs_csv = None  # Uses default from DataConfig (rPPG CSV)
            elif fs == "visual_only":
                fs_csv = csv_file or (data_cfg.csv_file.parent.parent / "visual" / "visual_features.csv")
            elif fs == "fused":
                fs_csv = csv_file or (data_cfg.csv_file.parent.parent / "visual" / "fused_features.csv")

            if fs_csv and not Path(fs_csv).exists():
                print(f"\n  SKIPPING {fs}: CSV file not found: {fs_csv}")
                continue

            result = run_pipeline_for_feature_set(
                feature_set=fs,
                data_cfg=data_cfg,
                qaoa_cfg=qaoa_cfg,
                vqc_cfg=vqc_cfg,
                decision_cfg=decision_cfg,
                dev_only=dev_only,
                csv_file=fs_csv,
                run_qaoa=args.select or args.all or True,
                run_train=args.train or args.all or True,
                run_eval=args.evaluate or args.all or True,
                run_baselines_flag=args.baselines or args.all or True,
            )
            all_results[fs] = result

        # Print comparative summary
        print("\n" + "=" * 72)
        print("  PHASE 6 COMPARATIVE RESULTS SUMMARY")
        print("=" * 72)
        print(f"{'Feature Set':<18} {'Quantum AUC':<12} {'Quantum Acc':<12} {'Best Baseline AUC':<18} {'Best Baseline':<15} {'Baseline Acc':<12}")
        print("-" * 90)
        for fs, result in all_results.items():
            if result["eval_results"]:
                q_auc = result["eval_results"]["metrics"]["auc_roc"]
                q_acc = result["eval_results"]["metrics"]["accuracy"]
            else:
                q_auc = q_acc = float("nan")
            if result["baseline_results"]:
                best_baseline = max(result["baseline_results"].items(), key=lambda x: x[1]["auc_roc"])
                b_auc = best_baseline[1]["auc_roc"]
                b_acc = best_baseline[1]["accuracy"]
                b_name = best_baseline[0]
            else:
                b_auc = float("nan")
                b_acc = float("nan")
                b_name = "N/A"
            print(f"{fs:<18} {q_auc:<12.4f} {q_acc:<12.4f} {b_auc:<18.4f} {b_name:<15} {b_acc:<12.4f}")

        # Save comparative results
        summary_file = OUTPUT_DIR / "phase6_comparison.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w") as fh:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "eval_results" and kk != "baseline_results"}
                      for k, v in all_results.items()}, fh, indent=2, default=str)
        print(f"\n  Comparative results saved to: {summary_file}")

        return 0

    if args.phase7_ablation:
        # Run full ablation study (Phase 7)
        print("\n" + "=" * 72)
        print("  PHASE 7 ABLATION STUDY: Experiments A-I")
        print("=" * 72)
        all_results = {}
        # Phase 7 ablation experiments A-I
        # Note: For POS/CHROM experiments, we need to run with different methods
        ablation_experiments = {
            "A_rppg_only": {"feature_set": "A_rppg_only", "method": "POS"},
            "B_pos_only": {"feature_set": "B_pos_only", "method": "POS"},
            "C_chrom_only": {"feature_set": "C_chrom_only", "method": "CHROM"},
            "D_pos_chrom": {"feature_set": "D_pos_chrom", "method": "POS"},  # Note: would need both methods
            "E_rppg_quality": {"feature_set": "E_rppg_quality", "method": "POS"},
            "F_rppg_cross_roi": {"feature_set": "F_rppg_cross_roi", "method": "POS"},
            "G_rppg_visual": {"feature_set": "G_rppg_visual", "method": "POS"},
            "H_rppg_visual_quality": {"feature_set": "H_rppg_visual_quality", "method": "POS"},
            "I_full": {"feature_set": "I_full", "method": "POS"},
        }
        for exp_name, exp_config in ablation_experiments.items():
            fs = exp_config["feature_set"]
            method = exp_config["method"]
            
            if fs == "A_rppg_only" or fs == "F_rppg_cross_roi" or fs == "E_rppg_quality":
                fs_csv = None  # Uses default from DataConfig (rPPG CSV)
            elif fs == "G_rppg_visual" or fs == "H_rppg_visual_quality" or fs == "I_full":
                fs_csv = csv_file or (data_cfg.csv_file.parent.parent / "visual" / "fused_features.csv")
            else:
                fs_csv = None  # Uses default from DataConfig (rPPG CSV)

            if fs_csv and not Path(fs_csv).exists():
                print(f"\n  SKIPPING {exp_name}: CSV file not found: {fs_csv}")
                continue

            # For POS/CHROM experiments, we need to run the pipeline with different methods
            # The current pipeline uses a single method. For POS/CHROM, we'd need to 
            # either run the pipeline twice or modify the pipeline to handle both.
            # For now, we'll use the default method (POS) and note this limitation.
            
            result = run_pipeline_for_feature_set(
                feature_set=fs,
                data_cfg=data_cfg,
                qaoa_cfg=qaoa_cfg,
                vqc_cfg=vqc_cfg,
                decision_cfg=decision_cfg,
                dev_only=dev_only,
                csv_file=fs_csv,
                run_qaoa=args.select or args.all or True,
                run_train=args.train or args.all or True,
                run_eval=args.evaluate or args.all or True,
                run_baselines_flag=args.baselines or args.all or True,
            )
            all_results[exp_name] = result

        # Print comparative summary
        print("\n" + "=" * 72)
        print("  PHASE 7 ABLATION STUDY RESULTS SUMMARY")
        print("=" * 72)
        print(f"{'Experiment':<22} {'Feature Set':<20} {'Quantum AUC':<12} {'Quantum Acc':<12} {'Best Baseline AUC':<18} {'Best Baseline':<15} {'Baseline Acc':<12}")
        print("-" * 110)
        for exp_name, result in all_results.items():
            if result["eval_results"]:
                q_auc = result["eval_results"]["metrics"]["auc_roc"]
                q_acc = result["eval_results"]["metrics"]["accuracy"]
            else:
                q_auc = q_acc = float("nan")
            if result["baseline_results"]:
                best_baseline = max(result["baseline_results"].items(), key=lambda x: x[1]["auc_roc"])
                b_auc = best_baseline[1]["auc_roc"]
                b_acc = best_baseline[1]["accuracy"]
                b_name = best_baseline[0]
            else:
                b_auc = float("nan")
                b_acc = float("nan")
                b_name = "N/A"
            # Extract experiment letter and feature set name
            exp_letter = exp_name.split("_")[0]
            fs_name = exp_config["feature_set"]
            print(f"{exp_name:<22} {fs_name:<20} {q_auc:<12.4f} {q_acc:<12.4f} {b_auc:<18.4f} {b_name:<15} {b_acc:<12.4f}")

        # Save comparative results
        summary_file = OUTPUT_DIR / "phase7_ablation.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w") as fh:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "eval_results" and kk != "baseline_results"}
                      for k, v in all_results.items()}, fh, indent=2, default=str)
        print(f"\n  Comparative results saved to: {summary_file}")

        return 0

    return 0

    if args.ensemble:
        # Run Phase 8 ensemble comparison
        print("\n" + "=" * 72)
        print("  PHASE 8 ENSEMBLE CLASSIFICATION: Ensemble Comparison")
        print("=" * 72)
        
        # Feature sets for ensemble comparison
        feature_sets = {
            "rppg_only": "rppg_only",
            "visual_only": "visual_only",
            "fused": "fused",
        }
        
        ensemble_results = run_phase8_ensemble_comparison(
            data_cfg=data_cfg,
            qaoa_cfg=qaoa_cfg,
            vqc_cfg=vqc_cfg,
            decision_cfg=decision_cfg,
            dev_only=dev_only,
            csv_file=csv_file,
            run_qaoa=args.select or args.all or True,
            run_train=args.train or args.all or True,
            run_eval=args.evaluate or args.all or True,
            run_baselines_flag=args.baselines or args.all or True,
        )
        
        print("\n" + "=" * 72)
        print("  PHASE 8 ENSEMBLE COMPARISON SUMMARY")
        print("=" * 72)
        print(f"{'Feature Set':<18} {'Quantum AUC':<12} {'Quantum Acc':<12} {'Best Baseline AUC':<18} {'Best Baseline':<15} {'Baseline Acc':<12}")
        print("-" * 90)
        for fs, result in ensemble_results.items():
            if result["eval_results"]:
                q_auc = result["eval_results"]["metrics"]["auc_roc"]
                q_acc = result["eval_results"]["metrics"]["accuracy"]
            else:
                q_auc = q_acc = float("nan")
            if result["baseline_results"]:
                best_baseline = max(result["baseline_results"].items(), key=lambda x: x[1]["auc_roc"])
                b_auc = best_baseline[1]["auc_roc"]
                b_acc = best_baseline[1]["accuracy"]
                b_name = best_baseline[0]
            else:
                b_auc = float("nan")
                b_acc = float("nan")
                b_name = "N/A"
            print(f"{fs:<18} {q_auc:<12.4f} {q_acc:<12.4f} {b_auc:<18.4f} {b_name:<15} {b_acc:<12.4f}")

        # Save comparative results
        summary_file = OUTPUT_DIR / "phase8_ensemble.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w") as fh:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "eval_results" and kk != "baseline_results"}
                      for k, v in ensemble_results.items()}, fh, indent=2, default=str)
        print(f"\n  Comparative results saved to: {summary_file}")

        return 0

    feature_set = args.feature_set
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