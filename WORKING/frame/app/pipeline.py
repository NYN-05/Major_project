import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

WORKING_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_OUTPUT = WORKING_ROOT / "output" / "frames"

from app.config import MODEL_WEIGHTS, build_config
from app.detector import FaceDetector, annotate_frame, crop_faces
from app.processing import FrameIngestor, FrameQualityAssessor, StorageManager

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_EXTRACT_ONLY_FLAGS = frozenset(
    {
        "--input-dir",
        "--sample-fps",
        "--min-seq-len",
        "--blur-threshold",
        "--dark-threshold",
        "--bright-threshold",
        "--min-face-area-ratio",
        "--compare-rates",
        "--sampling-note-file",
        "--quality-checklist-file",
        "--quality-report-file",
        "--extraction-log",
        "--summary-file",
        "--save-quality-examples",
    }
)
_FACE_ONLY_FLAGS = frozenset({"--display", "--save-metadata", "--max-io-workers"})


def setup_logger(name: str = "face_pipeline", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Face pipeline (was app/main.py)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Production-grade face pipeline for image, video, and webcam sources.")
    parser.add_argument("--source", default="1", help="Image path, video path, stream URL, or webcam index")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_WEIGHTS.keys()),
        default="yolov8n",
        help="Predefined YOLO face model preset",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Custom YOLO face model weights path (overrides --model)",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Minimum face confidence threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="Model inference image size")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, 0, 1, gpu, cuda")
    parser.add_argument("--half", action="store_true", help="Use half precision on CUDA")
    parser.add_argument("--output-root", default=str(GLOBAL_OUTPUT / "annotated"), help="Root output folder")
    parser.add_argument("--display", action="store_true", help="Display annotated frames in a window")
    parser.add_argument("--save-metadata", action="store_true", help="Write JSONL metadata per frame")
    parser.add_argument("--max-io-workers", type=int, default=4, help="Background workers for storage I/O")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger()
    source = args.source
    selected_weights = args.weights if args.weights else MODEL_WEIGHTS[args.model]

    config = build_config(
        source=source,
        weights_path=selected_weights,
        confidence_threshold=args.conf,
        image_size=args.imgsz,
        device=args.device,
        use_half=args.half,
        display=args.display,
        output_root=args.output_root,
        save_metadata=args.save_metadata,
        max_io_workers=args.max_io_workers,
    )

    detector = FaceDetector(
        weights_path=config.weights_path,
        confidence_threshold=config.confidence_threshold,
        image_size=config.image_size,
        device=config.device,
        use_half=config.use_half,
    )
    ingestor = FrameIngestor(config.source)
    storage = StorageManager(
        full_frames_dir=config.full_frames_dir,
        cropped_faces_dir=config.cropped_faces_dir,
        metadata_file=config.metadata_file,
        save_metadata=config.save_metadata,
        max_workers=config.max_io_workers,
    )

    logger.info("Pipeline started")
    start = time.time()
    processed_frames = 0

    try:
        for packet in ingestor.frames():
            detections = detector.detect(packet.frame)
            annotated = annotate_frame(packet.frame, detections)

            timestamp_id = str(time.time_ns())
            storage.save_annotated_frame(timestamp_id=timestamp_id, annotated_frame=annotated)

            for face_id, crop in crop_faces(packet.frame, detections):
                storage.save_face_crop(frame_id=packet.frame_id, face_id=face_id, crop=crop)

            storage.save_metadata_record(
                frame_id=packet.frame_id,
                timestamp_id=timestamp_id,
                detections=detections,
            )

            processed_frames += 1

            if config.display:
                cv2.imshow("Face Pipeline", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        ingestor.close()
        storage.close()
        cv2.destroyAllWindows()

    elapsed = max(time.time() - start, 1e-6)
    fps = processed_frames / elapsed
    logger.info("Pipeline finished | frames=%d | fps=%.2f", processed_frames, fps)
    logger.info("Annotated frames: %s", config.full_frames_dir)
    logger.info("Cropped faces: %s", config.cropped_faces_dir)
    if config.save_metadata:
        logger.info("Metadata file: %s", config.metadata_file)

    layer_result = run_frame_sampling_quality_layer(
        source=config.source,
        weights_path=str(config.weights_path),
        confidence_threshold=config.confidence_threshold,
        image_size=config.image_size,
        device=config.device,
        use_half=config.use_half,
    )

    if layer_result.get("status") == "skipped":
        logger.info("Sampling/quality layer skipped: %s", layer_result.get("reason", "n/a"))
    else:
        logger.info("Sampling/quality layer status: %s", layer_result.get("status", "unknown"))
        logger.info("Sampling/quality summary: %s", layer_result.get("summary_file", "n/a"))

    return 0


# ---------------------------------------------------------------------------
# Frame sampling + quality layer (was app/extract_frames.py)
# ---------------------------------------------------------------------------

def parse_extract_args():
    parser = argparse.ArgumentParser(description="Extract ordered video frames with sampling and quality assessment.")
    parser.add_argument("--source", default=None, help="Single video path")
    parser.add_argument("--input-dir", default=None, help="Directory containing videos")
    parser.add_argument(
        "--output-root",
        default=str(GLOBAL_OUTPUT / "frame_sequences"),
        help="Root folder for extracted sequences",
    )
    parser.add_argument("--sample-fps", type=float, default=10.0, help="Project-standard frame sampling FPS")
    parser.add_argument("--min-seq-len", type=int, default=64, help="Minimum accepted frames for temporal modeling")

    parser.add_argument(
        "--model",
        choices=sorted(MODEL_WEIGHTS.keys()),
        default="yolov8n",
        help="Predefined YOLO face model preset",
    )
    parser.add_argument("--weights", default=None, help="Custom YOLO weights path (overrides --model)")
    parser.add_argument("--conf", type=float, default=0.35, help="Minimum face confidence threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="Model inference image size")
    parser.add_argument("--device", default="auto", help="Inference device")
    parser.add_argument("--half", action="store_true", help="Use half precision on CUDA")

    parser.add_argument("--blur-threshold", type=float, default=8.0, help="Blur rejection threshold")
    parser.add_argument("--dark-threshold", type=float, default=45.0, help="Dark-frame rejection threshold")
    parser.add_argument("--bright-threshold", type=float, default=220.0, help="Overexposed-frame rejection threshold")
    parser.add_argument(
        "--min-face-area-ratio",
        type=float,
        default=0.005,
        help="Reject if largest detected face covers less than this image-area ratio",
    )

    parser.add_argument(
        "--compare-rates",
        default="5,10,15",
        help="Comma-separated sampling FPS values for comparison note",
    )
    parser.add_argument(
        "--sampling-note-file",
        default=str(GLOBAL_OUTPUT / "docs" / "frame_sampling_rate_comparison.md"),
        help="Path for sampling-rate comparison note",
    )
    parser.add_argument(
        "--quality-checklist-file",
        default=str(GLOBAL_OUTPUT / "docs" / "frame_quality_checklist.md"),
        help="Path for frame-quality checklist sheet",
    )
    parser.add_argument(
        "--quality-report-file",
        default=str(GLOBAL_OUTPUT / "docs" / "frame_quality_examples_report.md"),
        help="Path for quality examples report",
    )

    parser.add_argument(
        "--extraction-log",
        default=str(GLOBAL_OUTPUT / "frame_extraction_log.jsonl"),
        help="Extraction success/failure log file",
    )
    parser.add_argument(
        "--summary-file",
        default=str(GLOBAL_OUTPUT / "frame_extraction_summary.json"),
        help="Aggregate extraction summary file",
    )
    parser.add_argument(
        "--save-quality-examples",
        action="store_true",
        help="Save one accepted and one rejected example frame per video",
    )
    return parser.parse_args()


def find_videos(source: str | None, input_dir: str | None) -> list[Path]:
    videos: list[Path] = []
    if source:
        source_path = Path(source)
        if source_path.exists() and source_path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(source_path)
    if input_dir:
        dir_path = Path(input_dir)
        if dir_path.exists():
            videos.extend(sorted(p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS))
    unique = sorted({p.resolve() for p in videos})
    return [Path(p) for p in unique]


def estimate_sample_counts(video_path: Path, rates: list[float]) -> list[tuple[float, int, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [(rate, 0, 0.0) for rate in rates]

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    estimates: list[tuple[float, int, float]] = []
    for rate in rates:
        if source_fps <= 0 or total_frames <= 0:
            estimates.append((rate, 0, 0.0))
            continue

        duration_sec = total_frames / source_fps
        estimated = max(1, int(duration_sec * rate))
        interval_ms = 1000.0 / rate
        estimates.append((rate, estimated, interval_ms))
    return estimates


def write_sampling_note(note_path: Path, comparison_rows: list[dict], standard_rate: float, min_seq_len: int):
    note_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_by_video: dict[str, int] = {}
    for row in comparison_rows:
        if abs(row["sampling_fps"] - standard_rate) < 1e-9:
            baseline_by_video[row["video_name"]] = max(1, row["estimated_frames"])

    lines = [
        "# Frame Sampling Rate Comparison",
        "",
        f"Project-standard baseline sampling rate: {standard_rate} FPS",
        f"Minimum sequence length target for temporal modeling: {min_seq_len} frames",
        "",
        "| Video | Sampling FPS | Est. Frames | Interval (ms) | Compute Cost vs Baseline | Signal Fidelity Proxy | Sufficient (>= min length) |",
        "|---|---:|---:|---:|---:|---|---|",
    ]

    for row in comparison_rows:
        enough = "Yes" if row["estimated_frames"] >= min_seq_len else "No"
        base_est = baseline_by_video.get(row["video_name"], max(1, row["estimated_frames"]))
        compute_cost_ratio = row["estimated_frames"] / base_est
        if row["sampling_fps"] < standard_rate:
            fidelity = "Lower temporal fidelity"
        elif row["sampling_fps"] > standard_rate:
            fidelity = "Higher temporal fidelity"
        else:
            fidelity = "Baseline temporal fidelity"

        lines.append(
            f"| {row['video_name']} | {row['sampling_fps']} | {row['estimated_frames']} | {row['interval_ms']:.2f} | {compute_cost_ratio:.2f}x | {fidelity} | {enough} |"
        )

    lines.extend(
        [
            "",
            "Baseline decision:",
            f"- Selected {standard_rate} FPS as the default project setting for baseline experiments.",
            "- Lower rates reduce compute but can weaken temporal signal fidelity.",
            "- Higher rates preserve temporal detail but increase storage and processing cost.",
            "- Accuracy impact must be validated in downstream classifier experiments.",
        ]
    )

    note_path.write_text("\n".join(lines), encoding="utf-8")


def write_quality_checklist(path: Path, blur_threshold: float, dark_threshold: float, bright_threshold: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Frame Quality Checklist",
        "",
        "## Rule Definitions",
        f"- Blur rule: reject when Laplacian variance < {blur_threshold}.",
        f"- Brightness dark rule: reject when gray mean < {dark_threshold}.",
        f"- Brightness overexposed rule: reject when gray mean > {bright_threshold}.",
        "- Face visibility rule: reject when no detectable face is present.",
        "- Face size rule: reject when detected face is too small for reliable physiological signal.",
        "- Extreme pose rule: reject when detected face touches frame edge margin or has unusual face-box aspect ratio.",
        "- Composite quality score: w1*Blur + w2*Brightness + w3*FaceConfidence.",
        "",
        "## Rejection Categories",
        "- blur",
        "- dark_frame",
        "- overexposed_frame",
        "- no_face",
        "- face_too_small",
        "- extreme_pose",
        "",
        "## Reporting Checklist",
        "- Mark example accepted frames.",
        "- Mark example rejected frames for each category.",
        "- Verify quality flags are exported into metadata.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_quality_examples_report(path: Path, summaries: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Frame Quality Examples Report",
        "",
        "This report marks example good and bad frames saved during extraction.",
        "",
        "| Video | Accepted Example | Rejected Example |",
        "|---|---|---|",
    ]

    for row in summaries:
        accepted = row.get("accepted_example", "not saved")
        rejected = row.get("rejected_example", "not saved")
        lines.append(f"| {row.get('video', 'unknown')} | {accepted} | {rejected} |")

    path.write_text("\n".join(lines), encoding="utf-8")


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record) + "\n")


def run_extraction_for_video(
    video_path: Path,
    output_root: Path,
    sample_fps: float,
    min_seq_len: int,
    detector: FaceDetector,
    assessor: FrameQualityAssessor,
    log_path: Path,
    save_quality_examples: bool,
) -> dict:
    video_name = video_path.stem
    video_output_dir = output_root / video_name
    frames_dir = video_output_dir / "frames"
    examples_dir = video_output_dir / "quality_examples"
    metadata_path = video_output_dir / "frame_metadata.jsonl"

    frames_dir.mkdir(parents=True, exist_ok=True)
    if save_quality_examples:
        examples_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists():
        metadata_path.unlink()
    for frame_file in frames_dir.glob("*.jpg"):
        frame_file.unlink()
    if save_quality_examples:
        for example_file in examples_dir.glob("*.jpg"):
            example_file.unlink()

    ingestor = FrameIngestor(str(video_path))

    sampled_count = 0
    accepted_count = 0
    rejected_count = 0
    rejections = Counter()
    accepted_example_saved = False
    rejected_example_saved = False
    accepted_shapes = set()
    accepted_timestamps: list[int] = []
    sampled_source_ids: list[int] = []
    sampled_timestamps: list[int] = []
    accepted_face_heights: list[int] = []
    accepted_face_widths: list[int] = []
    accepted_green_means: list[float] = []

    try:
        for packet in ingestor.frames(sample_fps=sample_fps):
            sampled_count += 1
            sampled_source_ids.append(packet.source_frame_id)
            sampled_timestamps.append(packet.timestamp_ms)
            detections = detector.detect(packet.frame)
            quality = assessor.evaluate(packet.frame, detections)

            record = {
                "video": video_path.name,
                "frame_id": packet.frame_id,
                "source_frame_id": packet.source_frame_id,
                "timestamp_ms": packet.timestamp_ms,
                "quality": {
                    "accepted": quality.accepted,
                    "quality_flag": quality.quality_flag,
                    "score": round(quality.quality_score, 4),
                    "rejection_reasons": quality.rejection_reasons,
                    "blur_score": round(quality.blur_score, 4),
                    "brightness_score": round(quality.brightness_score, 4),
                    "face_detected": quality.face_visible,
                    "face_confidence": round(quality.face_confidence, 4),
                    "max_face_area_ratio": round(quality.max_face_area_ratio, 6),
                    "face_too_small": quality.face_too_small,
                    "extreme_pose": quality.extreme_pose,
                },
            }

            if quality.accepted:
                accepted_count += 1
                accepted_shapes.add(packet.frame.shape[:2])
                accepted_timestamps.append(packet.timestamp_ms)
                accepted_green_means.append(float(packet.frame[:, :, 1].mean()))
                if detections:
                    best = max(detections, key=lambda d: d.confidence)
                    accepted_face_widths.append(best.x2 - best.x1)
                    accepted_face_heights.append(best.y2 - best.y1)
                frame_name = f"frame_{packet.frame_id:06d}_t{packet.timestamp_ms:010d}.jpg"
                cv2.imwrite(str(frames_dir / frame_name), packet.frame)
                record["saved_frame"] = frame_name

                if save_quality_examples and not accepted_example_saved:
                    cv2.imwrite(str(examples_dir / "accepted_example.jpg"), packet.frame)
                    accepted_example_saved = True
            else:
                rejected_count += 1
                for reason in quality.rejection_reasons:
                    rejections[reason] += 1
                if save_quality_examples and not rejected_example_saved:
                    cv2.imwrite(str(examples_dir / "rejected_example.jpg"), packet.frame)
                    rejected_example_saved = True

            append_jsonl(metadata_path, record)

        sufficient_for_temporal = accepted_count >= min_seq_len
        consistent_resolution = len(accepted_shapes) <= 1
        if len(accepted_timestamps) >= 2:
            effective_duration_sec = (max(accepted_timestamps) - min(accepted_timestamps)) / 1000.0
        else:
            effective_duration_sec = 0.0

        expected_interval_ms = (1000.0 / sample_fps) if sample_fps > 0 else 0.0
        tolerance_ms = expected_interval_ms * 0.35
        if len(sampled_timestamps) >= 2 and expected_interval_ms > 0:
            deltas_ms = [b - a for a, b in zip(sampled_timestamps[:-1], sampled_timestamps[1:])]
            continuity_breaks = sum(1 for d in deltas_ms if abs(d - expected_interval_ms) > tolerance_ms)
        else:
            continuity_breaks = 0

        avg_face_height = (sum(accepted_face_heights) / len(accepted_face_heights)) if accepted_face_heights else 0.0
        avg_face_width = (sum(accepted_face_widths) / len(accepted_face_widths)) if accepted_face_widths else 0.0
        if len(accepted_green_means) >= 2:
            mean_g = sum(accepted_green_means) / len(accepted_green_means)
            var_g = sum((g - mean_g) ** 2 for g in accepted_green_means) / len(accepted_green_means)
            color_variation_std = var_g ** 0.5
        else:
            color_variation_std = 0.0

        rppg_face_resolution_ok = avg_face_height >= 48 and avg_face_width >= 48
        rppg_color_variation_ok = color_variation_std > 0.5
        rppg_ready = sufficient_for_temporal and consistent_resolution and rppg_face_resolution_ok and rppg_color_variation_ok

        summary = {
            "video": video_path.name,
            "status": "success",
            "sample_fps": sample_fps,
            "sampled_frames": sampled_count,
            "total_frames_extracted": accepted_count,
            "accepted_frames": accepted_count,
            "rejected_frames": rejected_count,
            "rejections": dict(rejections),
            "effective_duration_sec": round(effective_duration_sec, 3),
            "sampled_sequence_continuity": {
                "expected_interval_ms": round(expected_interval_ms, 3),
                "tolerance_ms": round(tolerance_ms, 3),
                "continuity_breaks": continuity_breaks,
                "no_frame_drops_detected": continuity_breaks == 0,
            },
            "sequence_sufficient_for_temporal": sufficient_for_temporal,
            "consistent_frame_resolution": consistent_resolution,
            "rppg_checks": {
                "avg_face_width_px": round(avg_face_width, 2),
                "avg_face_height_px": round(avg_face_height, 2),
                "color_variation_std": round(color_variation_std, 4),
                "face_resolution_ok": rppg_face_resolution_ok,
                "color_variation_ok": rppg_color_variation_ok,
            },
            "rppg_input_ready": rppg_ready,
            "metadata_file": str(metadata_path),
            "frames_dir": str(frames_dir),
            "accepted_example": str(examples_dir / "accepted_example.jpg") if accepted_example_saved else "not saved",
            "rejected_example": str(examples_dir / "rejected_example.jpg") if rejected_example_saved else "not saved",
        }
        append_jsonl(log_path, summary)
        return summary

    except Exception as exc:
        failure = {
            "video": video_path.name,
            "status": "failure",
            "error": str(exc),
        }
        append_jsonl(log_path, failure)
        return failure
    finally:
        ingestor.close()


def run_frame_sampling_quality_layer(
    source: str,
    weights_path: str,
    confidence_threshold: float,
    image_size: int,
    device: str,
    use_half: bool,
    sample_fps: float = 10.0,
    min_seq_len: int = 64,
    output_root: str = None,
    extraction_log: str = None,
    summary_file: str = None,
    compare_rates: str = "5,10,15",
    sampling_note_file: str = None,
    quality_checklist_file: str = None,
    quality_report_file: str = None,
    blur_threshold: float = 8.0,
    dark_threshold: float = 45.0,
    bright_threshold: float = 220.0,
    min_face_area_ratio: float = 0.005,
    save_quality_examples: bool = True,
) -> dict:
    logger = setup_logger("frame_extractor")
    videos = find_videos(source=source, input_dir=None)

    if not videos:
        logger.info("Frame sampling + quality layer skipped (source is not a local video file): %s", source)
        return {
            "status": "skipped",
            "reason": "source is not a local video file",
            "source": source,
        }

    detector = FaceDetector(
        weights_path=Path(weights_path),
        confidence_threshold=confidence_threshold,
        image_size=image_size,
        device=device,
        use_half=use_half,
    )
    assessor = FrameQualityAssessor(
        blur_threshold=blur_threshold,
        dark_threshold=dark_threshold,
        bright_threshold=bright_threshold,
        min_face_area_ratio=min_face_area_ratio,
    )

    output_root_path = Path(output_root) if output_root else GLOBAL_OUTPUT / "frame_sequences"
    log_path = Path(extraction_log) if extraction_log else GLOBAL_OUTPUT / "frame_extraction_log.jsonl"
    summary_path = Path(summary_file) if summary_file else GLOBAL_OUTPUT / "frame_extraction_summary.json"
    sampling_note_path = Path(sampling_note_file) if sampling_note_file else GLOBAL_OUTPUT / "docs" / "frame_sampling_rate_comparison.md"
    quality_checklist_path = Path(quality_checklist_file) if quality_checklist_file else GLOBAL_OUTPUT / "docs" / "frame_quality_checklist.md"
    quality_report_path = Path(quality_report_file) if quality_report_file else GLOBAL_OUTPUT / "docs" / "frame_quality_examples_report.md"

    comparison_rates = [float(v.strip()) for v in compare_rates.split(",") if v.strip()]
    comparison_rows: list[dict] = []
    all_summaries: list[dict] = []

    for video_path in videos:
        for rate, est, interval_ms in estimate_sample_counts(video_path, comparison_rates):
            comparison_rows.append(
                {
                    "video_name": video_path.name,
                    "sampling_fps": rate,
                    "estimated_frames": est,
                    "interval_ms": interval_ms,
                }
            )

        summary = run_extraction_for_video(
            video_path=video_path,
            output_root=output_root_path,
            sample_fps=sample_fps,
            min_seq_len=min_seq_len,
            detector=detector,
            assessor=assessor,
            log_path=log_path,
            save_quality_examples=save_quality_examples,
        )
        all_summaries.append(summary)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    write_sampling_note(
        note_path=sampling_note_path,
        comparison_rows=comparison_rows,
        standard_rate=sample_fps,
        min_seq_len=min_seq_len,
    )
    write_quality_checklist(
        path=quality_checklist_path,
        blur_threshold=blur_threshold,
        dark_threshold=dark_threshold,
        bright_threshold=bright_threshold,
    )
    write_quality_examples_report(quality_report_path, all_summaries)

    success_count = sum(1 for row in all_summaries if row.get("status") == "success")
    return {
        "status": "success" if success_count == len(all_summaries) else "partial",
        "videos": len(all_summaries),
        "success": success_count,
        "summary_file": str(summary_path),
        "extraction_log": str(log_path),
        "sampling_note": sampling_note_file,
        "quality_checklist": quality_checklist_file,
        "quality_report": quality_report_file,
    }


def extract_main() -> int:
    args = parse_extract_args()
    logger = setup_logger("frame_extractor")

    videos = find_videos(args.source, args.input_dir)
    if not videos:
        logger.error("No videos found. Use --source or --input-dir.")
        return 1

    output_root = Path(args.output_root)
    log_path = Path(args.extraction_log)
    summary_file = Path(args.summary_file)

    selected_weights = args.weights if args.weights else MODEL_WEIGHTS[args.model]
    detector = FaceDetector(
        weights_path=Path(selected_weights),
        confidence_threshold=args.conf,
        image_size=args.imgsz,
        device=args.device,
        use_half=args.half,
    )
    assessor = FrameQualityAssessor(
        blur_threshold=args.blur_threshold,
        dark_threshold=args.dark_threshold,
        bright_threshold=args.bright_threshold,
        min_face_area_ratio=args.min_face_area_ratio,
    )

    comparison_rates = [float(v.strip()) for v in args.compare_rates.split(",") if v.strip()]
    comparison_rows: list[dict] = []
    all_summaries: list[dict] = []

    logger.info("Starting extraction for %d video(s)", len(videos))
    for video_path in videos:
        logger.info("Processing %s", video_path)
        for rate, est, interval_ms in estimate_sample_counts(video_path, comparison_rates):
            comparison_rows.append(
                {
                    "video_name": video_path.name,
                    "sampling_fps": rate,
                    "estimated_frames": est,
                    "interval_ms": interval_ms,
                }
            )

        summary = run_extraction_for_video(
            video_path=video_path,
            output_root=output_root,
            sample_fps=args.sample_fps,
            min_seq_len=args.min_seq_len,
            detector=detector,
            assessor=assessor,
            log_path=log_path,
            save_quality_examples=args.save_quality_examples,
        )
        all_summaries.append(summary)

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    write_sampling_note(
        note_path=Path(args.sampling_note_file),
        comparison_rows=comparison_rows,
        standard_rate=args.sample_fps,
        min_seq_len=args.min_seq_len,
    )
    write_quality_checklist(
        path=Path(args.quality_checklist_file),
        blur_threshold=args.blur_threshold,
        dark_threshold=args.dark_threshold,
        bright_threshold=args.bright_threshold,
    )
    write_quality_examples_report(Path(args.quality_report_file), all_summaries)

    success_count = sum(1 for row in all_summaries if row.get("status") == "success")
    logger.info("Completed extraction | success=%d | total=%d", success_count, len(all_summaries))
    logger.info("Summary file: %s", summary_file)
    logger.info("Extraction log: %s", log_path)
    logger.info("Sampling note: %s", args.sampling_note_file)
    logger.info("Quality checklist: %s", args.quality_checklist_file)
    logger.info("Quality report: %s", args.quality_report_file)

    return 0 if success_count == len(all_summaries) else 2


if __name__ == "__main__":
    argv_flags = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
    if argv_flags & _EXTRACT_ONLY_FLAGS and not (argv_flags & _FACE_ONLY_FLAGS):
        raise SystemExit(extract_main())
    raise SystemExit(main())