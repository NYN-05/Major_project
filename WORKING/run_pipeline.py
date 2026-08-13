"""
run_pipeline.py
================
Single endpoint for the full deepfake-verification flow:

    frames  ->  rPPG  ->  quantum  ->  final verdict (REAL / FAKE / UNCERTAIN)

Stages
------
1. FRAMES  : frame sampling + quality assessment via the `frame/` module
             (YOLO face detection, blur/dark/bright/face checks).
2. RPPG    : physiological feature extraction via the `RPPG/` module
             (POS/CHROM pulse reconstruction -> 8-feature vector).
3. QUANTUM : train-fitted feature scaling + QAOA-selected subset of the
              rPPG features -> trained Hybrid VQC checkpoint -> P(real) ->
              KYC decision bins (real >= 0.7, fake <= 0.3).

The quantum layer consumes the rPPG features directly (same names/order as
RPPGFeatures.feature_names()); no synthetic or transformed data is used.

Usage
-----
    python run_pipeline.py --source path/to/video.mp4 [--method POS|CHROM] [--out result.json]

Requires pre-trained quantum artifacts (output/qaoa_selection.json,
output/hybrid_vqc.pt). If missing, run once from this folder:
    python -m quantum.pipeline --all
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

WORKING = Path(__file__).resolve().parent
FRAME_ROOT = WORKING / "frame"
RPPG_ROOT = WORKING / "RPPG"
OUTPUT_ROOT = WORKING / "output" / "pipeline"
FRAMES_OUTPUT = WORKING / "output" / "frames"
RPPG_OUTPUT = WORKING / "output" / "rppg"

for _root in (FRAME_ROOT, RPPG_ROOT, WORKING):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from app.pipeline import run_frame_sampling_quality_layer  # stage 1
from rppg import RPPGPipeline  # stage 2

from quantum.pipeline import predict_features  # stage 3

FRAME_WEIGHTS = FRAME_ROOT / "weights" / "yolov8n-face-lindevs.pt"
RPPG_CLASSIFIER = RPPG_OUTPUT / "rppg_classifier.pkl"

VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Stage 1: frames
# ---------------------------------------------------------------------------

def run_frames_stage(video_path: Path) -> tuple[dict, dict, dict]:
    docs_dir = FRAMES_OUTPUT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    layer_result = run_frame_sampling_quality_layer(
        source=str(video_path),
        weights_path=str(FRAME_WEIGHTS),
        confidence_threshold=0.35,
        image_size=320,
        device="auto",
        use_half=False,
        sample_fps=10.0,
        output_root=str(FRAMES_OUTPUT / "frame_sequences"),
        extraction_log=str(FRAMES_OUTPUT / "frame_extraction_log.jsonl"),
        summary_file=str(FRAMES_OUTPUT / "frame_extraction_summary.json"),
        compare_rates="5,10,15",
        sampling_note_file=str(docs_dir / "frame_sampling_rate_comparison.md"),
        quality_checklist_file=str(docs_dir / "frame_quality_checklist.md"),
        quality_report_file=str(docs_dir / "frame_quality_examples_report.md"),
        save_quality_examples=False,
    )
    summaries = []
    summary_file = Path(layer_result.get("summary_file", ""))
    if summary_file.exists():
        summaries = json.loads(summary_file.read_text(encoding="utf-8"))
    target = next(
        (s for s in summaries if s.get("video") == video_path.name and s.get("status") == "success"),
        {},
    )
    layer_status = layer_result.get("status", "unknown")
    if not target:
        reason = layer_result.get("reason", "no per-video summary produced")
        return {"status": "skipped" if layer_status == "skipped" else "failure", "reason": reason}, frame_stats_from_summary({}), {}
    handoff = {
        "frames_dir": target.get("frames_dir", ""),
        "metadata_file": target.get("metadata_file", ""),
        "fps": float(target.get("sample_fps", 10.0)),
    }
    return {"status": "success", "layer_status": layer_status}, frame_stats_from_summary(target), handoff


def frame_stats_from_summary(summary: dict) -> dict:
    stats = {
        "sampled_frames": 0,
        "accepted_frames": 0,
        "rejections": {},
        "mean_quality_score": 0.0,
        "mean_face_confidence": 0.0,
        "temporal_coverage_ratio": 0.0,
    }
    if not summary:
        return stats
    stats["sampled_frames"] = int(summary.get("sampled_frames", 0))
    stats["accepted_frames"] = int(summary.get("accepted_frames", 0))
    stats["rejections"] = summary.get("rejections", {})

    scores: list[float] = []
    confidences: list[float] = []
    metadata_file = Path(summary.get("metadata_file", ""))
    if metadata_file.exists():
        for line in metadata_file.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            quality = record.get("quality", {})
            scores.append(float(quality.get("score", 0.0)))
            if quality.get("accepted"):
                confidences.append(float(quality.get("face_confidence", 0.0)))

    stats["mean_quality_score"] = float(np.mean(scores)) if scores else 0.0
    stats["mean_face_confidence"] = float(np.mean(confidences)) if confidences else 0.0
    stats["temporal_coverage_ratio"] = (
        stats["accepted_frames"] / stats["sampled_frames"] if stats["sampled_frames"] else 0.0
    )
    return stats


# ---------------------------------------------------------------------------
# Stage 2: rPPG
# ---------------------------------------------------------------------------

def run_rppg_stage(video_path: Path, method: str = "POS", handoff: dict | None = None) -> tuple[dict, np.ndarray | None]:
    """rPPG feature extraction. Uses the stage-1 accepted frames when the
    frame layer produced them (input_mode=stage1_frames); otherwise falls
    back to reading the video directly (input_mode=video_direct)."""
    pipeline = RPPGPipeline(method=method)
    input_mode = "video_direct"
    if handoff:
        frames_dir = handoff.get("frames_dir", "")
        if frames_dir and Path(frames_dir).is_dir():
            try:
                result = pipeline.process_frames(
                    frames_dir,
                    metadata_path=handoff.get("metadata_file"),
                    fps=float(handoff.get("fps", 10.0)),
                )
                input_mode = "stage1_frames"
            except IOError as exc:
                print(f"      [warn] stage-1 frame handoff failed ({exc}); falling back to video read")
                result = pipeline.process_video(str(video_path))
        else:
            result = pipeline.process_video(str(video_path))
    else:
        result = pipeline.process_video(str(video_path))
    payload = {
        "method": method,
        "input_mode": input_mode,
        "fps_used": float(result.fps),
        "n_frames_total": int(result.n_frames_total),
        "n_frames_usable": int(result.n_frames_usable),
        "warnings": list(result.warnings),
    }
    if result.features is None:
        return payload, None
    payload["features"] = result.features.to_dict()
    return payload, result.to_feature_vector()


def rppg_classifier_crosscheck(vector: np.ndarray) -> dict:
    """Cross-check via the trained rPPG RandomForest (label 1 = DEEPFAKE,
    opposite of the quantum stage where LABEL_REAL=1)."""
    if not RPPG_CLASSIFIER.exists():
        return {"skipped": "rppg_classifier.pkl not found"}
    try:
        with open(RPPG_CLASSIFIER, "rb") as fh:
            clf = pickle.load(fh)
    except (pickle.UnpicklingError, AttributeError, ImportError, ModuleNotFoundError) as exc:
        return {"skipped": f"rppg_classifier.pkl could not be loaded: {exc}"}
    x = vector.reshape(1, -1)
    n_features = getattr(clf, "n_features_in_", 0)
    classes = getattr(clf, "classes_", None)
    n_classes = 0 if classes is None else len(classes)
    if n_features != 10 or n_classes != 2:
        return {
            "skipped": "rppg_classifier.pkl incompatible "
            f"(n_features={n_features}, classes={n_classes}); expected 10 features, 2 classes",
        }
    try:
        proba = clf.predict_proba(x)[0]
        pred = int(clf.predict(x)[0])
    except Exception as exc:
        return {"skipped": f"rppg_classifier.pkl prediction failed: {type(exc).__name__}: {exc}"}
    if pred == 1:
        return {"verdict": "DEEPFAKE", "probability": float(proba[1])}
    return {"verdict": "REAL", "probability": float(proba[0])}


# ---------------------------------------------------------------------------
# Stage 4: quantum
# ---------------------------------------------------------------------------

def quantum_inference(features: dict) -> dict:
    """QAOA-selected subset of the actual rPPG features -> trained hybrid VQC.

    `predict_features` is the quantum layer's own inference entry point: it
    applies the saved training-time QAOA indices to the 10-feature rPPG output
    and returns P(real) plus the KYC verdict (REAL / FAKE / UNCERTAIN).
    """
    return predict_features(features)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="End-to-end deepfake verdict: frames -> rPPG -> quantum"
    )
    parser.add_argument("--source", required=True, help="Path to the input video (mp4/avi/...)")
    parser.add_argument("--method", default="POS", choices=["POS", "CHROM"], help="rPPG reconstruction method")
    parser.add_argument("--out", default=None, help="Output JSON path (default: output/pipeline/pipeline_result.json)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    video_path = Path(args.source).resolve()
    if not video_path.exists():
        print(f"[error] Video not found: {video_path}")
        return 2

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result = {"video": str(video_path), "timestamp": datetime.now().isoformat(), "stages": {}}

    print(f"[1/3] FRAMES stage  : {video_path.name}")
    frames_stage, frame_stats, frame_handoff = run_frames_stage(video_path)
    result["stages"]["frames"] = {**frames_stage, "stats": frame_stats}
    print(
        f"      accepted/sampled = {frame_stats['accepted_frames']}/{frame_stats['sampled_frames']} "
        f"mean_quality = {frame_stats['mean_quality_score']:.3f}"
    )

    print(f"[2/3] RPPG stage    : method={args.method}")
    rppg_stage, vector = run_rppg_stage(video_path, args.method, frame_handoff)
    result["stages"]["rppg"] = rppg_stage
    if vector is None:
        result["verdict"] = {
            "label": VERDICT_INCONCLUSIVE,
            "confidence": None,
            "reason": "rPPG produced no feature vector (insufficient usable frames)",
        }
        _finish(result, args.out, exit_code=3)
        return 3
    print(
        f"      input={rppg_stage['input_mode']} "
        f"usable = {rppg_stage['n_frames_usable']}/{rppg_stage['n_frames_total']} "
        f"HR = {rppg_stage['features']['heart_rate_bpm']:.1f} BPM"
    )
    result["stages"]["rppg_crosscheck"] = rppg_classifier_crosscheck(vector)

    print("[3/3] QUANTUM stage : rPPG features -> QAOA subset -> hybrid VQC")
    quantum = quantum_inference(rppg_stage["features"])
    result["stages"]["quantum"] = quantum
    print(
        f"      prob_real = {quantum['prob_real']:.4f} -> {quantum['verdict']} "
        f"(confidence {quantum['confidence']:.4f})"
    )

    result["verdict"] = {"label": quantum["verdict"], "confidence": quantum["confidence"]}
    _finish(result, args.out, exit_code=0)
    return 0


def _finish(result: dict, out_path: str | None, exit_code: int) -> None:
    out = Path(out_path) if out_path else OUTPUT_ROOT / "pipeline_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    verdict = result["verdict"]
    print("\n" + "=" * 60)
    print(f"FINAL VERDICT: {verdict['label']}")
    if verdict.get("confidence") is not None:
        print(f"CONFIDENCE   : {verdict['confidence']:.4f}")
    if verdict.get("reason"):
        print(f"REASON       : {verdict['reason']}")
    print(f"RESULT JSON  : {out}")
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())