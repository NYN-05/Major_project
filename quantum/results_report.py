import json
from pathlib import Path

from quantum.config import DecisionConfig, VQCConfig
from quantum.evaluate import load_quantum_metrics
from quantum.classical_baseline import load_baseline_metrics


def build_report(decision_config: DecisionConfig, vqc_config: VQCConfig) -> Path:
    quantum = load_quantum_metrics(vqc_config.metrics_file)
    baselines = load_baseline_metrics(decision_config.metrics_baseline_file)

    lines = [
        "# Quantum ML Results Report",
        "",
        "## Models Compared",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    rows = []
    rows.append(
        (
            quantum["model"],
            quantum["accuracy"],
            quantum["precision"],
            quantum["recall"],
            quantum["f1"],
            quantum["auc_roc"],
        )
    )
    for name, metrics in baselines.items():
        rows.append(
            (metrics["model"], metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["auc_roc"])
        )

    for name, acc, prec, rec, f1, auc in rows:
        auc_text = "n/a" if auc is None else f"{auc:.4f}"
        lines.append(f"| {name} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {auc_text} |")

    lines.extend(
        [
            "",
            "## QAOA Feature Selection",
            "",
            "Selected features: " + ", ".join(quantum["selected_features"]),
            "",
            f"Expected-cost energy after optimization: {quantum.get('expected_cost_energy', 'n/a')}",
            "",
            "## Confusion Matrix (Quantum Classifier)",
            "",
            "| | Predicted Fake | Predicted Real |",
            "|---|---:|---:|",
            f"| Actual Fake | {quantum['confusion_matrix']['tn']} | {quantum['confusion_matrix']['fp']} |",
            f"| Actual Real | {quantum['confusion_matrix']['fn']} | {quantum['confusion_matrix']['tp']} |",
            "",
            "## Calibration and Decisions",
            "",
            f"- Expected Calibration Error (ECE): {quantum['ece']:.4f}",
            f"- Decision thresholds: fake if p <= {quantum['thresholds']['fake_max_prob']}, "
            f"real if p >= {quantum['thresholds']['real_min_prob']}, otherwise uncertain",
            f"- Test decisions -> real: {quantum['decisions']['real_count']}, "
            f"uncertain: {quantum['decisions']['uncertain_count']}, fake: {quantum['decisions']['fake_count']}",
            "",
            "## Plots",
            "",
            f"- ROC curve: `{quantum['roc_plot']}`",
            f"- Confusion matrix: `{quantum['confusion_plot']}`",
            "",
            "## Notes",
            "",
            "- The quantum model is a hybrid Variational Quantum Classifier (VQC) trained on "
            "QAOA-selected rPPG features.",
            "- Results are on synthetic placeholder features; swap in the real rPPG feature file "
            "and rerun `run_quantum.py --all`.",
        ]
    )

    decision_config.report_file.parent.mkdir(parents=True, exist_ok=True)
    decision_config.report_file.write_text("\n".join(lines), encoding="utf-8")
    return decision_config.report_file


if __name__ == "__main__":
    path = build_report(DecisionConfig(), VQCConfig())
    print(f"Report written to {path}")
