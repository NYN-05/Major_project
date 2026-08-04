import argparse
import sys
from pathlib import Path

QUANTUM_ROOT = Path(__file__).resolve().parents[0]
if str(QUANTUM_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(QUANTUM_ROOT.parent))

from quantum.classical_baseline import run_baselines
from quantum.config import DataConfig, DecisionConfig, QAOASelectionConfig, VQCConfig
from quantum.dummy_data import generate_dataset, load_dataset
from quantum.evaluate import evaluate_quantum_model
from quantum.qaoa_selection import QAOASelector, save_selection, verify_hamiltonian
from quantum.results_report import build_report
from quantum.train import train_vqc


def parse_args():
    parser = argparse.ArgumentParser(description="Quantum ML component runner.")
    parser.add_argument("--gen-data", action="store_true", help="Generate synthetic rPPG feature dataset")
    parser.add_argument("--select", action="store_true", help="Run QAOA feature selection")
    parser.add_argument("--train", action="store_true", help="Train hybrid VQC")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate quantum model on test split")
    parser.add_argument("--baselines", action="store_true", help="Train and evaluate classical baselines")
    parser.add_argument("--report", action="store_true", help="Build markdown results report")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline in order")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_cfg = DataConfig()
    qaoa_cfg = QAOASelectionConfig()
    vqc_cfg = VQCConfig()
    decision_cfg = DecisionConfig()

    if not (args.all or args.gen_data or args.select or args.train or args.evaluate or args.baselines or args.report):
        parser_help = "No step selected. Use --all or one of: --gen-data --select --train --evaluate --baselines --report"
        print(parser_help)
        return 2

    if args.gen_data or args.all:
        print("[1/6] Generating synthetic rPPG feature dataset...")
        generate_dataset(data_cfg)
        data = load_dataset(data_cfg.data_file)
        for key in ("X_train", "X_val", "X_test"):
            print(f"  {key}: {data[key].shape}")
    else:
        data = load_dataset(data_cfg.data_file)

    if args.select or args.all:
        print("[2/6] Running QAOA feature selection...")
        err = verify_hamiltonian(data["X_train"], data["y_train"])
        print(f"  Hamiltonian verification max error: {err:.2e}")
        selector = QAOASelector(qaoa_cfg)
        selection = selector.select(data["X_train"], data["y_train"])
        save_selection(selection, qaoa_cfg.selection_file)
        print(f"  Selected features: {selection['selected_features']}")
        print(f"  Marginals: {[round(v, 3) for v in selection['marginal_probabilities']]}")
    else:
        selection = None

    if args.train or args.all:
        print("[3/6] Training hybrid VQC...")
        if selection is None:
            from quantum.qaoa_selection import load_selection

            selection = load_selection(qaoa_cfg.selection_file)
        model = train_vqc(data, selection["selected_indices"], vqc_cfg)
        print(f"  Checkpoint saved to {vqc_cfg.checkpoint_file}")
        print(f"  Training log: {vqc_cfg.log_file}")

    if args.evaluate or args.all:
        print("[4/6] Evaluating quantum model...")
        if selection is None:
            from quantum.qaoa_selection import load_selection

            selection = load_selection(qaoa_cfg.selection_file)
        metrics = evaluate_quantum_model(data, selection, vqc_cfg, decision_cfg)
        print(f"  Test accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc_roc']:.4f}")

    if args.baselines or args.all:
        print("[5/6] Training classical baselines...")
        results = run_baselines(data, decision_cfg)
        for name, m in results.items():
            print(f"  {name}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc_roc']:.4f}")

    if args.report or args.all:
        print("[6/6] Building results report...")
        report_path = build_report(decision_cfg, vqc_cfg)
        print(f"  Report: {report_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
