import argparse

from quantum.baselines import run_baselines
from quantum.config import DataConfig, DecisionConfig, QAOASelectionConfig, VQCConfig
from quantum.data import generate_dataset, load_dataset
from quantum.evaluate import evaluate_quantum_model
from quantum.selection import QAOASelector, load_selection, save_selection, verify_hamiltonian
from quantum.vqc import train_vqc


def build_parser():
    parser = argparse.ArgumentParser(
        description="Quantum decision layer: QAOA feature selection + hybrid VQC classification."
    )
    parser.add_argument("--gen-data", action="store_true", help="Generate the synthetic rPPG dataset")
    parser.add_argument("--select", action="store_true", help="Run QAOA feature selection")
    parser.add_argument("--train", action="store_true", help="Train the hybrid VQC")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained VQC")
    parser.add_argument("--baselines", action="store_true", help="Train classical baselines")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline in order")
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
            args.gen_data,
            args.select,
            args.train,
            args.evaluate,
            args.baselines,
            args.all,
        ]
    ):
        parser.print_help()
        return 2

    if args.gen_data or args.all:
        print("[1/6] Generating synthetic rPPG feature dataset...")
        generate_dataset(data_cfg)
    data = load_dataset(data_cfg.data_file)
    for key in ("X_train", "X_val", "X_test"):
        print(f"  {key}: {data[key].shape}")

    selection = None
    if args.select or args.all:
        print("[2/6] Running QAOA feature selection...")
        error = verify_hamiltonian(data["X_train"], data["y_train"], qaoa_cfg)
        print(f"  Hamiltonian verification max error: {error:.2e}")
        selection = QAOASelector(qaoa_cfg).select(data["X_train"], data["y_train"])
        save_selection(selection, qaoa_cfg.selection_file)
        print(f"  Selected features: {selection['selected_features']}")
    else:
        selection = load_selection(qaoa_cfg.selection_file)

    indices = [int(i) for i in selection["selected_indices"]]
    if args.train or args.all:
        print("[3/6] Training hybrid VQC...")
        train_vqc(
            data["X_train"][:, indices],
            data["y_train"].astype(float),
            vqc_cfg,
        )
        print(f"  Checkpoint: {vqc_cfg.checkpoint_file}")

    if args.evaluate or args.all:
        print("[4/6] Evaluating quantum model...")
        result = evaluate_quantum_model(
            data["X_test"][:, indices],
            data["y_test"],
            vqc_cfg,
            decision_cfg,
        )
        metrics = result["metrics"]
        print(
            f"  accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"auc={metrics['auc_roc']:.4f} ece={metrics['ece']:.4f}"
        )
        print(f"  decision bins: {result['decision_bins']}")

    if args.baselines or args.all:
        print("[5/6] Training classical baselines...")
        results = run_baselines(
            data["X_train"][:, indices],
            data["y_train"],
            data["X_test"][:, indices],
            data["y_test"],
            decision_cfg,
        )
        for name, metrics in results.items():
            print(
                f"  {name}: accuracy={metrics['accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} auc={metrics['auc_roc']:.4f}"
            )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())