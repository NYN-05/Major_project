"""Flask API server for the Quantum ML component demo frontend."""

import json
import sys
from pathlib import Path

QUANTUM_ROOT = Path(__file__).resolve().parents[0]
if str(QUANTUM_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(QUANTUM_ROOT.parent))

from flask import Flask, jsonify, send_file

from quantum.config import DecisionConfig, QAOASelectionConfig, VQCConfig

OUTPUT_DIR = QUANTUM_ROOT / "output"
DOCS_DIR = QUANTUM_ROOT / "docs"
FRONTEND_DIR = QUANTUM_ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"

app = Flask(__name__)


def _read_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_lines(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@app.get("/api/metrics")
def api_metrics():
    vqc_cfg, dec_cfg = VQCConfig(), DecisionConfig()
    quantum = _read_json(vqc_cfg.metrics_file)
    baselines = _read_json(dec_cfg.metrics_baseline_file)
    selection = _read_json(QAOASelectionConfig().selection_file)
    return jsonify(
        {
            "quantum": quantum if quantum else {"model": "not trained yet"},
            "baselines": baselines if baselines else {},
            "selection": selection if selection else {},
        }
    )


@app.get("/api/training_log")
def api_training_log():
    return jsonify(_read_lines(VQCConfig().log_file))


@app.get("/api/selection")
def api_selection():
    return jsonify(_read_json(QAOASelectionConfig().selection_file))


@app.get("/api/report")
def api_report():
    path = DecisionConfig().report_file
    return jsonify({"exists": path.exists(), "content": path.read_text(encoding="utf-8") if path.exists() else None})


@app.get("/api/plots/roc")
def api_roc():
    return send_file(OUTPUT_DIR / "roc_curve.png", mimetype="image/png")


@app.get("/api/plots/confusion")
def api_confusion():
    return send_file(OUTPUT_DIR / "confusion_matrix.png", mimetype="image/png")


@app.get("/")
@app.get("/<path:path>")
def index(path="index.html"):
    if path.startswith("api/"):
        return jsonify({"error": "not found"}), 404
    if (DIST_DIR / path).exists():
        return send_file(DIST_DIR / path)
    return f"Quantum demo API running. Frontend build not found at {DIST_DIR}. Run `npm run build` in quantum/frontend."


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
