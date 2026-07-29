#!/usr/bin/env python3
"""Face detection, tracking, and alignment pipeline for extracted Celeb-DF frames.

Outputs:
- annotations/<split>/<video>.json
- faces/<split>/<video>/frame_000001.jpg
- debug/face_detection_samples/ and debug/alignment_samples/
- faces/reports/

The detector is an OpenCV Haar cascade. It is lightweight, ships with the OpenCV wheel,
and works well enough here when combined with temporal smoothing, low-confidence retry,
and eye-based alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Optional, Sequence

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore[assignment]


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = WORKSPACE_ROOT / "frames"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT
SPLITS = ("train", "val", "test")
FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class FrameEntry:
    sequence_index: int
    split: str
    video_name: str
    frame_dir: Path
    frame_files: list[Path]
    source_manifest_path: Optional[Path]
    source_path: Optional[str]
    class_name: Optional[str]
    label: Optional[str]
    frame_width: Optional[int]
    frame_height: Optional[int]


@dataclass(frozen=True)
class DetectionCandidate:
    bbox: tuple[float, float, float, float]
    confidence: float
    pass_label: str
    keypoints: dict[str, tuple[float, float]]


@dataclass
class FrameDecision:
    raw_bbox: Optional[tuple[float, float, float, float]]
    final_bbox: tuple[float, float, float, float]
    confidence: float
    status: str
    bbox_source: str
    failure_reason: str
    keypoints: Optional[dict[str, tuple[float, float]]]
    alignment_status: str = "pending"
    rotation_degrees: float = 0.0


@dataclass
class FrameAnnotation:
    frame_name: str
    crop_name: str
    bbox: list[int]
    raw_bbox: Optional[list[int]]
    confidence: float
    status: str
    bbox_source: str
    alignment_status: str
    rotation_degrees: float
    failure_reason: str


@dataclass
class VideoRecord:
    sequence_index: int
    split: str
    class_name: Optional[str]
    label: Optional[str]
    video_name: str
    source_path: Optional[str]
    frame_dir: str
    annotation_path: str
    crop_dir: str
    total_frames: int
    detected_frames: int
    fallback_frames: int
    invalid_frames: int
    aligned_frames: int
    detection_success_rate: float
    usable_roi_rate: float
    alignment_success_rate: float
    mean_confidence: float
    raw_center_jitter_px: float
    stabilized_center_jitter_px: float
    jitter_reduction_pct: float
    raw_area_cv: float
    stabilized_area_cv: float
    mean_abs_rotation_deg: float
    processing_seconds: float
    status: str
    error_message: str
    frame_width: Optional[int]
    frame_height: Optional[int]
    output_size: int


@dataclass(frozen=True)
class Config:
    input_root: Path
    output_root: Path
    faces_root: Path
    annotations_root: Path
    debug_root: Path
    reports_root: Path
    padding: float
    smoothing_alpha: float
    max_temporal_gap: int
    min_bbox_side: int
    min_bbox_area_ratio: float
    accept_confidence: float
    initial_accept_confidence: float
    output_size: int
    workers: int
    overwrite: bool
    continue_on_error: bool
    debug_success_samples: int
    debug_fallback_samples: int
    debug_missed_samples: int
    debug_alignment_samples: int
    debug_alignment_fallback_samples: int


@dataclass
class TrackState:
    smoothed_bbox: Optional[tuple[float, float, float, float]] = None
    last_confidence: float = 0.0
    missed_frames: int = 0


class HaarFaceBackend:
    def __init__(self) -> None:
        cv2_mod, _ = require_dependencies()
        self._cv2 = cv2_mod
        self._face = cv2_mod.CascadeClassifier(cv2_mod.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_candidates = [
            Path(cv2_mod.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml",
            Path(cv2_mod.data.haarcascades) / "haarcascade_eye.xml",
        ]
        eye_path = next((path for path in eye_candidates if path.is_file()), eye_candidates[-1])
        self._eyes = cv2_mod.CascadeClassifier(str(eye_path))
        if self._face.empty() or self._eyes.empty():
            raise RuntimeError("Failed to load OpenCV Haar cascades")

    def close(self) -> None:
        return None

    def detect_best_face(self, frame_bgr: Any, retry_rank: int = 0) -> Optional[DetectionCandidate]:
        gray = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2GRAY)
        strict_passes = [("strict", 6), ("retry", 4), ("relaxed", 3)]
        pass_label, min_neighbors = strict_passes[min(retry_rank, len(strict_passes) - 1)]
        candidates: list[DetectionCandidate] = []

        for x, y, w, h, confidence in self._detect_faces(gray, min_neighbors=min_neighbors):
            keypoints = self._detect_eye_keypoints(gray, (x, y, w, h))
            candidates.append(
                DetectionCandidate(
                    bbox=(float(x), float(y), float(x + w), float(y + h)),
                    confidence=confidence,
                    pass_label=pass_label,
                    keypoints=keypoints,
                )
            )

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item.confidence, bbox_area(item.bbox)), reverse=True)
        return candidates[0]

    def _detect_faces(self, gray: Any, min_neighbors: int) -> list[tuple[int, int, int, int, float]]:
        min_side = max(24, min(gray.shape[0], gray.shape[1]) // 10)
        try:
            faces, _reject_levels, weights = self._face.detectMultiScale3(
                gray,
                scaleFactor=1.08,
                minNeighbors=min_neighbors,
                flags=0,
                minSize=(min_side, min_side),
                outputRejectLevels=True,
            )
            faces_list = list(faces)
            weights_list = list(weights) if weights is not None else [0.0] * len(faces_list)
            return [
                (int(x), int(y), int(w), int(h), sigmoid_confidence(float(weight)))
                for (x, y, w, h), weight in zip(faces_list, weights_list)
            ]
        except Exception:
            faces = self._face.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=min_neighbors,
                flags=0,
                minSize=(min_side, min_side),
            )
            return [(int(x), int(y), int(w), int(h), 0.5) for x, y, w, h in faces]

    def _detect_eye_keypoints(self, gray: Any, face_rect: tuple[int, int, int, int]) -> dict[str, tuple[float, float]]:
        x, y, w, h = face_rect
        roi_top = y
        roi_bottom = y + int(h * 0.65)
        roi = gray[roi_top:roi_bottom, x:x + w]
        if roi.size == 0:
            return {}

        eye_min = max(8, min(w, h) // 6)
        eyes = self._eyes.detectMultiScale(roi, scaleFactor=1.05, minNeighbors=3, flags=0, minSize=(eye_min, eye_min))
        if len(eyes) < 2:
            eyes = self._eyes.detectMultiScale(roi, scaleFactor=1.03, minNeighbors=2, flags=0, minSize=(max(6, eye_min // 2), max(6, eye_min // 2)))
        if len(eyes) < 2:
            return {}

        eye_boxes = sorted((int(ex), int(ey), int(ew), int(eh)) for ex, ey, ew, eh in eyes)
        left_eye = eye_boxes[0]
        right_eye = eye_boxes[-1]

        def center(box: tuple[int, int, int, int]) -> tuple[float, float]:
            bx, by, bw, bh = box
            return float(x + bx + bw / 2.0), float(roi_top + by + bh / 2.0)

        return {"left_eye": center(left_eye), "right_eye": center(right_eye)}


class DebugSampler:
    def __init__(self, root: Path, success_limit: int, fallback_limit: int, missed_limit: int, alignment_limit: int, alignment_fallback_limit: int) -> None:
        self._root = root
        self._limits = {"correct": success_limit, "fallback": fallback_limit, "missed": missed_limit, "aligned": alignment_limit, "alignment_fallback": alignment_fallback_limit}
        self._counts = {key: 0 for key in self._limits}
        self._lock = threading.Lock()

    def maybe_save_detection(self, cv2_mod, split: str, video_name: str, frame_name: str, frame_bgr: Any, bbox: tuple[float, float, float, float], decision: FrameDecision) -> None:
        category = "missed" if decision.status == "invalid" else ("fallback" if decision.bbox_source == "temporal_fallback" else "correct")
        if not self._reserve(category):
            return
        sample = render_detection_sample(cv2_mod, frame_bgr, bbox, decision, frame_name)
        path = self._root / "face_detection_samples" / category / split / f"{video_name}_{Path(frame_name).stem}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2_mod.imwrite(str(path), sample, [int(cv2_mod.IMWRITE_JPEG_QUALITY), 92])

    def maybe_save_alignment(self, cv2_mod, split: str, video_name: str, frame_name: str, before_bgr: Any, after_bgr: Any, decision: FrameDecision) -> None:
        category = "aligned" if decision.alignment_status == "aligned" else "alignment_fallback"
        if not self._reserve(category):
            return
        sample = render_alignment_sample(cv2_mod, before_bgr, after_bgr, decision, frame_name)
        path = self._root / "alignment_samples" / category / split / f"{video_name}_{Path(frame_name).stem}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2_mod.imwrite(str(path), sample, [int(cv2_mod.IMWRITE_JPEG_QUALITY), 92])

    def _reserve(self, category: str) -> bool:
        with self._lock:
            if self._counts[category] >= self._limits[category]:
                return False
            self._counts[category] += 1
            return True


def require_dependencies():
    missing = []
    if cv2 is None:
        missing.append("opencv-python-headless")
    if np is None:
        missing.append("numpy")
    if missing:
        raise RuntimeError("Missing runtime dependencies: " + ", ".join(missing))
    return cv2, np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect, stabilize, and align faces from extracted frames.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Process extracted frames into stabilized face crops and annotations.")
    run.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--videos", nargs="*")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--continue-on-error", action="store_true")
    run.add_argument("--output-size", type=int, default=224)
    run.add_argument("--padding", type=float, default=0.25)
    run.add_argument("--smoothing-alpha", type=float, default=0.65)
    run.add_argument("--max-temporal-gap", type=int, default=3)
    run.add_argument("--min-bbox-side", type=int, default=48)
    run.add_argument("--min-bbox-area-ratio", type=float, default=0.01)
    run.add_argument("--accept-confidence", type=float, default=0.35)
    run.add_argument("--initial-accept-confidence", type=float, default=0.25)
    run.add_argument("--debug-success-samples", type=int, default=8)
    run.add_argument("--debug-fallback-samples", type=int, default=8)
    run.add_argument("--debug-missed-samples", type=int, default=8)
    run.add_argument("--debug-alignment-samples", type=int, default=8)
    run.add_argument("--debug-alignment-fallback-samples", type=int, default=8)
    return parser


def configure_logging(reports_root: Path) -> logging.Logger:
    reports_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("face_roi_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(reports_root / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def normalize_video_folder(input_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
        raise FileNotFoundError(f"Video folder not found: {resolved}")
    direct = (input_root / candidate).resolve()
    if direct.is_dir():
        return direct
    matches = []
    for split in SPLITS:
        split_root = input_root / split
        if not split_root.is_dir():
            continue
        nested = (split_root / candidate).resolve()
        if nested.is_dir():
            matches.append(nested)
        for child in split_root.iterdir():
            if child.is_dir() and child.name == candidate.name:
                matches.append(child.resolve())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous video folder selection: {value}")
    raise FileNotFoundError(f"Could not resolve video folder: {value}")


def load_manifest(input_root: Path, split: str, video_name: str) -> dict[str, object]:
    manifest_path = input_root / "reports" / "manifests" / split / f"{video_name}.json"
    if not manifest_path.is_file():
        return {"manifest_path": None}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {"manifest_path": manifest_path}
    return {
        "manifest_path": manifest_path,
        "source_path": payload.get("source_path"),
        "class_name": payload.get("class_name"),
        "label": payload.get("label"),
        "frame_width": parse_int(payload.get("width")),
        "frame_height": parse_int(payload.get("height")),
    }


def discover_frame_entries(input_root: Path, videos: Optional[Sequence[str]]) -> list[FrameEntry]:
    entries: list[FrameEntry] = []
    if videos:
        candidates = [normalize_video_folder(input_root, value) for value in videos]
    else:
        candidates = []
        for split in SPLITS:
            split_root = input_root / split
            if split_root.is_dir():
                candidates.extend(sorted(path for path in split_root.iterdir() if path.is_dir() and path.name != "reports"))

    for sequence_index, frame_dir in enumerate(sorted({path.resolve() for path in candidates}), start=1):
        parts = frame_dir.relative_to(input_root.resolve()).parts
        if len(parts) < 2 or parts[0] not in SPLITS:
            raise ValueError(f"Frame folder is not inside train/val/test: {frame_dir}")
        split = parts[0]
        frame_files = sorted(path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in FRAME_EXTENSIONS)
        if not frame_files:
            raise FileNotFoundError(f"No extracted frames found in: {frame_dir}")
        manifest = load_manifest(input_root, split, frame_dir.name)
        entries.append(
            FrameEntry(
                sequence_index=sequence_index,
                split=split,
                video_name=frame_dir.name,
                frame_dir=frame_dir,
                frame_files=frame_files,
                source_manifest_path=manifest.get("manifest_path"),
                source_path=manifest.get("source_path"),
                class_name=manifest.get("class_name"),
                label=manifest.get("label"),
                frame_width=manifest.get("frame_width"),
                frame_height=manifest.get("frame_height"),
            )
        )
    return entries


def parse_int(value: object) -> Optional[int]:
    try:
        if value in (None, "", "null"):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def sigmoid_confidence(weight: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-weight))
    except OverflowError:
        return 0.0 if weight < 0 else 1.0


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_side_lengths(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1])


def clamp_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(width - 1), bbox[0]))
    y1 = max(0.0, min(float(height - 1), bbox[1]))
    x2 = max(0.0, min(float(width), bbox[2]))
    y2 = max(0.0, min(float(height), bbox[3]))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return x1, y1, x2, y2


def expand_bbox(bbox: tuple[float, float, float, float], padding: float, width: int, height: int) -> tuple[float, float, float, float]:
    bw, bh = bbox_side_lengths(bbox)
    if bw <= 0 or bh <= 0:
        return clamp_bbox(bbox, width, height)
    return clamp_bbox((bbox[0] - bw * padding, bbox[1] - bh * padding, bbox[2] + bw * padding, bbox[3] + bh * padding), width, height)


def bbox_to_int_list(bbox: tuple[float, float, float, float]) -> list[int]:
    return [int(math.floor(bbox[0])), int(math.floor(bbox[1])), int(math.ceil(bbox[2])), int(math.ceil(bbox[3]))]


def ema_bbox(previous: tuple[float, float, float, float], current: tuple[float, float, float, float], alpha: float) -> tuple[float, float, float, float]:
    return tuple(alpha * cur + (1.0 - alpha) * prev for prev, cur in zip(previous, current))  # type: ignore[return-value]


def min_face_size(width: int, height: int, min_side: int) -> tuple[int, int]:
    side = max(min_side, min(width, height) // 10)
    return side, side


def center_placeholder_bbox(width: int, height: int, coverage: float = 0.8) -> tuple[float, float, float, float]:
    side = min(width, height) * coverage
    side = max(1.0, min(side, float(min(width, height))))
    cx, cy = width / 2.0, height / 2.0
    half = side / 2.0
    return clamp_bbox((cx - half, cy - half, cx + half, cy + half), width, height)


def acceptable_bbox(bbox: tuple[float, float, float, float], width: int, height: int, config: Config) -> bool:
    bw, bh = bbox_side_lengths(bbox)
    if bw < config.min_bbox_side or bh < config.min_bbox_side:
        return False
    return bbox_area(bbox) / float(width * height) >= config.min_bbox_area_ratio


def read_frame_or_placeholder(frame_path: Path, output_size: int) -> tuple[Any, bool]:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return np.zeros((output_size, output_size, 3), dtype=np.uint8), False
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame, True


def align_crop(crop_bgr: Any, keypoints: Optional[dict[str, tuple[float, float]]], output_size: int) -> tuple[Any, str, float]:
    if keypoints is None or not {"left_eye", "right_eye"}.issubset(keypoints):
        return cv2.resize(crop_bgr, (output_size, output_size), interpolation=cv2.INTER_LINEAR), "resized_only", 0.0

    source = np.float32([[keypoints["right_eye"][0], keypoints["right_eye"][1]], [keypoints["left_eye"][0], keypoints["left_eye"][1]]])
    target = np.float32([[output_size * 0.68, output_size * 0.35], [output_size * 0.32, output_size * 0.35]])
    matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
    eye_tilt = abs(math.degrees(math.atan2(keypoints["left_eye"][1] - keypoints["right_eye"][1], keypoints["left_eye"][0] - keypoints["right_eye"][0])))
    if matrix is None:
        return cv2.resize(crop_bgr, (output_size, output_size), interpolation=cv2.INTER_LINEAR), "resized_only", eye_tilt
    aligned = cv2.warpAffine(crop_bgr, matrix, (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if aligned is None or aligned.size == 0:
        return cv2.resize(crop_bgr, (output_size, output_size), interpolation=cv2.INTER_LINEAR), "resized_only", eye_tilt
    return aligned, "aligned", eye_tilt


def render_detection_sample(cv2_mod, frame_bgr: Any, bbox: tuple[float, float, float, float], decision: FrameDecision, frame_name: str) -> Any:
    canvas = frame_bgr.copy()
    color = (0, 220, 0) if decision.status == "valid" and decision.bbox_source != "temporal_fallback" else (0, 180, 255)
    if decision.status == "invalid":
        color = (0, 0, 255)
    x1, y1, x2, y2 = bbox_to_int_list(bbox)
    cv2_mod.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    text = f"{decision.bbox_source} | {decision.status} | conf={decision.confidence:.2f}"
    if decision.failure_reason:
        text = f"{text} | {decision.failure_reason}"
    cv2_mod.putText(canvas, frame_name, (12, 24), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2_mod.LINE_AA)
    cv2_mod.putText(canvas, frame_name, (12, 24), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 1, cv2_mod.LINE_AA)
    cv2_mod.putText(canvas, text, (12, max(48, canvas.shape[0] - 18)), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2_mod.LINE_AA)
    cv2_mod.putText(canvas, text, (12, max(48, canvas.shape[0] - 18)), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2_mod.LINE_AA)
    return canvas


def render_alignment_sample(cv2_mod, before_bgr: Any, after_bgr: Any, decision: FrameDecision, frame_name: str) -> Any:
    height = max(before_bgr.shape[0], after_bgr.shape[0])
    before = resize_to_height(cv2_mod, before_bgr, height)
    after = resize_to_height(cv2_mod, after_bgr, height)
    separator = np.full((height, 12, 3), 28, dtype=np.uint8)
    canvas = cv2_mod.hconcat([before, separator, after])
    text = f"{frame_name} | {decision.alignment_status} | rot={decision.rotation_degrees:.1f} deg"
    if decision.failure_reason:
        text = f"{text} | {decision.failure_reason}"
    cv2_mod.putText(canvas, text, (12, 28), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2_mod.LINE_AA)
    cv2_mod.putText(canvas, text, (12, 28), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 1, cv2_mod.LINE_AA)
    return canvas


def resize_to_height(cv2_mod, image_bgr: Any, target_height: int) -> Any:
    if image_bgr.shape[0] == target_height:
        return image_bgr
    scale = target_height / float(image_bgr.shape[0])
    target_width = max(1, int(round(image_bgr.shape[1] * scale)))
    return cv2_mod.resize(image_bgr, (target_width, target_height), interpolation=cv2_mod.INTER_LINEAR)


def safe_mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def coefficient_of_variation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    if avg == 0:
        return 0.0
    return pstdev(values) / avg


def average_shift(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return safe_mean(math.dist(points[i - 1], points[i]) for i in range(1, len(points)))


def write_csv_rows(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_existing_annotation(path: Path) -> Optional[dict[str, object]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sequential_face_crops_valid(crop_dir: Path, expected_count: int, output_size: int) -> bool:
    if not crop_dir.is_dir():
        return False
    files = sorted(path for path in crop_dir.glob("frame_*.jpg") if path.is_file())
    if len(files) != expected_count:
        return False
    for index, file in enumerate(files, start=1):
        if file.name != f"frame_{index:06d}.jpg":
            return False
    sample = cv2.imread(str(files[0]))
    return sample is not None and sample.shape[:2] == (output_size, output_size)


def should_skip_video(entry: FrameEntry, config: Config) -> bool:
    annotation_path = config.annotations_root / entry.split / f"{entry.video_name}.json"
    crop_dir = config.faces_root / entry.split / entry.video_name
    existing = load_existing_annotation(annotation_path)
    if not existing:
        return False
    if existing.get("source_frame_dir") != str(entry.frame_dir):
        return False
    if parse_int(existing.get("frame_count")) != len(entry.frame_files):
        return False
    if parse_int(existing.get("output_size")) != config.output_size:
        return False
    return sequential_face_crops_valid(crop_dir, len(entry.frame_files), config.output_size)


def process_video(entry: FrameEntry, config: Config, logger: logging.Logger, sampler: DebugSampler) -> VideoRecord:
    cv2_mod, _ = require_dependencies()
    start = time.perf_counter()
    annotation_path = config.annotations_root / entry.split / f"{entry.video_name}.json"
    crop_dir = config.faces_root / entry.split / entry.video_name

    if not config.overwrite and should_skip_video(entry, config):
        return VideoRecord(
            sequence_index=entry.sequence_index,
            split=entry.split,
            class_name=entry.class_name,
            label=entry.label,
            video_name=entry.video_name,
            source_path=entry.source_path,
            frame_dir=str(entry.frame_dir),
            annotation_path=str(annotation_path),
            crop_dir=str(crop_dir),
            total_frames=len(entry.frame_files),
            detected_frames=0,
            fallback_frames=0,
            invalid_frames=0,
            aligned_frames=0,
            detection_success_rate=0.0,
            usable_roi_rate=0.0,
            alignment_success_rate=0.0,
            mean_confidence=0.0,
            raw_center_jitter_px=0.0,
            stabilized_center_jitter_px=0.0,
            jitter_reduction_pct=0.0,
            raw_area_cv=0.0,
            stabilized_area_cv=0.0,
            mean_abs_rotation_deg=0.0,
            processing_seconds=0.0,
            status="skipped",
            error_message="Already processed and validated.",
            frame_width=entry.frame_width,
            frame_height=entry.frame_height,
            output_size=config.output_size,
        )

    if config.overwrite:
        shutil.rmtree(crop_dir, ignore_errors=True)
        shutil.rmtree(annotation_path.parent, ignore_errors=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)

    backend = HaarFaceBackend()
    state = TrackState()
    annotations: list[FrameAnnotation] = []
    raw_centers: list[tuple[float, float]] = []
    stable_centers: list[tuple[float, float]] = []
    raw_areas: list[float] = []
    stable_areas: list[float] = []
    confidences: list[float] = []
    rotations: list[float] = []
    detected = fallback = invalid = aligned = 0

    try:
        for index, frame_path in enumerate(entry.frame_files, start=1):
            frame_name = frame_path.name
            crop_name = f"frame_{index:06d}.jpg"
            frame_bgr, readable = read_frame_or_placeholder(frame_path, config.output_size)
            height, width = frame_bgr.shape[:2]

            candidate = None
            if readable:
                for retry_rank in range(3):
                    candidate = backend.detect_best_face(frame_bgr, retry_rank=retry_rank)
                    if candidate is not None:
                        break

            decision = resolve_decision(candidate, state, width, height, config)
            if decision.status == "valid" and decision.bbox_source in {"detected", "retry"}:
                detected += 1
                if decision.raw_bbox is not None:
                    raw_centers.append(bbox_center(decision.raw_bbox))
                    raw_areas.append(bbox_area(decision.raw_bbox))
            elif decision.status == "valid":
                fallback += 1
            else:
                invalid += 1

            padded = expand_bbox(decision.final_bbox, config.padding, width, height)
            crop = crop_from_bbox(frame_bgr, padded, config.output_size)
            aligned_crop, alignment_status, rotation = align_crop(crop, decision.keypoints, config.output_size)
            decision.alignment_status = alignment_status
            decision.rotation_degrees = rotation

            validate_crop(aligned_crop, config.output_size)
            cv2_mod.imwrite(str(crop_dir / crop_name), aligned_crop, [int(cv2_mod.IMWRITE_JPEG_QUALITY), 92])

            if decision.status == "valid":
                stable_centers.append(bbox_center(padded))
                stable_areas.append(bbox_area(padded))
                if alignment_status == "aligned":
                    aligned += 1
            confidences.append(decision.confidence)
            if alignment_status == "aligned":
                rotations.append(abs(rotation))

            annotations.append(
                FrameAnnotation(
                    frame_name=frame_name,
                    crop_name=crop_name,
                    bbox=bbox_to_int_list(padded),
                    raw_bbox=bbox_to_int_list(decision.raw_bbox) if decision.raw_bbox is not None else None,
                    confidence=round(float(decision.confidence), 6),
                    status=decision.status,
                    bbox_source=decision.bbox_source,
                    alignment_status=alignment_status,
                    rotation_degrees=round(float(rotation), 4),
                    failure_reason=decision.failure_reason,
                )
            )

            sampler.maybe_save_detection(cv2_mod, entry.split, entry.video_name, frame_name, frame_bgr, padded, decision)
            sampler.maybe_save_alignment(cv2_mod, entry.split, entry.video_name, frame_name, crop, aligned_crop, decision)

        valid = detected + fallback
        detection_rate = detected / len(entry.frame_files) if entry.frame_files else 0.0
        usable_rate = valid / len(entry.frame_files) if entry.frame_files else 0.0
        alignment_rate = aligned / valid if valid else 0.0
        raw_jitter = average_shift(raw_centers)
        stable_jitter = average_shift(stable_centers)
        jitter_reduction = 0.0 if raw_jitter == 0 else max(0.0, (raw_jitter - stable_jitter) / raw_jitter * 100.0)
        record = VideoRecord(
            sequence_index=entry.sequence_index,
            split=entry.split,
            class_name=entry.class_name,
            label=entry.label,
            video_name=entry.video_name,
            source_path=entry.source_path,
            frame_dir=str(entry.frame_dir),
            annotation_path=str(annotation_path),
            crop_dir=str(crop_dir),
            total_frames=len(entry.frame_files),
            detected_frames=detected,
            fallback_frames=fallback,
            invalid_frames=invalid,
            aligned_frames=aligned,
            detection_success_rate=detection_rate,
            usable_roi_rate=usable_rate,
            alignment_success_rate=alignment_rate,
            mean_confidence=safe_mean(confidences),
            raw_center_jitter_px=raw_jitter,
            stabilized_center_jitter_px=stable_jitter,
            jitter_reduction_pct=jitter_reduction,
            raw_area_cv=coefficient_of_variation(raw_areas),
            stabilized_area_cv=coefficient_of_variation(stable_areas),
            mean_abs_rotation_deg=safe_mean(rotations),
            processing_seconds=time.perf_counter() - start,
            status="success" if invalid == 0 else "degraded",
            error_message="" if invalid == 0 else f"{invalid} frames used placeholder ROIs.",
            frame_width=entry.frame_width,
            frame_height=entry.frame_height,
            output_size=config.output_size,
        )

        write_annotation_json(
            annotation_path,
            {
                "split": entry.split,
                "class_name": entry.class_name,
                "label": entry.label,
                "video_name": entry.video_name,
                "source_frame_dir": str(entry.frame_dir),
                "source_manifest_path": str(entry.source_manifest_path) if entry.source_manifest_path else None,
                "source_path": entry.source_path,
                "frame_width": entry.frame_width,
                "frame_height": entry.frame_height,
                "frame_count": record.total_frames,
                "valid_frame_count": record.detected_frames + record.fallback_frames,
                "invalid_frame_count": record.invalid_frames,
                "detection_success_rate": record.detection_success_rate,
                "usable_roi_rate": record.usable_roi_rate,
                "alignment_success_rate": record.alignment_success_rate,
                "output_size": record.output_size,
                "frames": [asdict(item) for item in annotations],
            },
        )
        return record
    except Exception as exc:
        logger.error("Failed %s | %s", entry.video_name, exc)
        shutil.rmtree(crop_dir, ignore_errors=True)
        annotation_path.unlink(missing_ok=True)
        raise
    finally:
        backend.close()


def resolve_decision(candidate: Optional[DetectionCandidate], state: TrackState, width: int, height: int, config: Config) -> FrameDecision:
    if candidate is not None:
        accepts = config.initial_accept_confidence if state.smoothed_bbox is None else config.accept_confidence
        if candidate.confidence >= accepts and acceptable_bbox(candidate.bbox, width, height, config):
            stabilized = candidate.bbox if state.smoothed_bbox is None else ema_bbox(state.smoothed_bbox, candidate.bbox, config.smoothing_alpha)
            state.smoothed_bbox = stabilized
            state.last_confidence = candidate.confidence
            state.missed_frames = 0
            return FrameDecision(candidate.bbox, stabilized, candidate.confidence, "valid", candidate.pass_label, "", candidate.keypoints)

        reason = []
        if candidate.confidence < accepts:
            reason.append("low_confidence")
        if not acceptable_bbox(candidate.bbox, width, height, config):
            reason.append("tiny_bbox")
        failure_reason = "+".join(reason) if reason else "rejected_detection"
        if state.smoothed_bbox is not None and state.missed_frames < config.max_temporal_gap:
            state.missed_frames += 1
            return FrameDecision(candidate.bbox, state.smoothed_bbox, state.last_confidence, "valid", "temporal_fallback", failure_reason, None)
        state.missed_frames += 1
        return FrameDecision(candidate.bbox, center_placeholder_bbox(width, height), candidate.confidence, "invalid", "center_placeholder", failure_reason, None)

    if state.smoothed_bbox is not None and state.missed_frames < config.max_temporal_gap:
        state.missed_frames += 1
        return FrameDecision(None, state.smoothed_bbox, state.last_confidence, "valid", "temporal_fallback", "no_face_detected", None)
    state.missed_frames += 1
    return FrameDecision(None, center_placeholder_bbox(width, height), 0.0, "invalid", "center_placeholder", "no_face_detected", None)


def crop_from_bbox(frame_bgr: Any, bbox: tuple[float, float, float, float], output_size: int) -> Any:
    width = frame_bgr.shape[1]
    height = frame_bgr.shape[0]
    x1, y1, x2, y2 = bbox_to_int_list(clamp_bbox(bbox, width, height))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    return crop


def validate_crop(image: Any, output_size: int) -> None:
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"Invalid face crop shape: {None if image is None else image.shape}")
    if image.shape[:2] != (output_size, output_size):
        raise RuntimeError(f"Unexpected crop size: {image.shape[:2]} != {(output_size, output_size)}")


def write_annotation_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_failure_record(entry: FrameEntry, config: Config, error_message: str) -> VideoRecord:
    annotation_path = config.annotations_root / entry.split / f"{entry.video_name}.json"
    crop_dir = config.faces_root / entry.split / entry.video_name
    return VideoRecord(
        sequence_index=entry.sequence_index,
        split=entry.split,
        class_name=entry.class_name,
        label=entry.label,
        video_name=entry.video_name,
        source_path=entry.source_path,
        frame_dir=str(entry.frame_dir),
        annotation_path=str(annotation_path),
        crop_dir=str(crop_dir),
        total_frames=len(entry.frame_files),
        detected_frames=0,
        fallback_frames=0,
        invalid_frames=len(entry.frame_files),
        aligned_frames=0,
        detection_success_rate=0.0,
        usable_roi_rate=0.0,
        alignment_success_rate=0.0,
        mean_confidence=0.0,
        raw_center_jitter_px=0.0,
        stabilized_center_jitter_px=0.0,
        jitter_reduction_pct=0.0,
        raw_area_cv=0.0,
        stabilized_area_cv=0.0,
        mean_abs_rotation_deg=0.0,
        processing_seconds=0.0,
        status="failure",
        error_message=error_message,
        frame_width=entry.frame_width,
        frame_height=entry.frame_height,
        output_size=config.output_size,
    )


def write_reports(records: Sequence[VideoRecord], config: Config) -> None:
    metrics_fields = [
        "sequence_index", "split", "class_name", "label", "video_name", "source_path", "frame_dir", "annotation_path", "crop_dir",
        "total_frames", "detected_frames", "fallback_frames", "invalid_frames", "aligned_frames", "detection_success_rate",
        "usable_roi_rate", "alignment_success_rate", "mean_confidence", "raw_center_jitter_px", "stabilized_center_jitter_px",
        "jitter_reduction_pct", "raw_area_cv", "stabilized_area_cv", "mean_abs_rotation_deg", "processing_seconds", "status",
        "error_message", "frame_width", "frame_height", "output_size",
    ]
    write_csv_rows(config.reports_root / "face_detection_metrics.csv", [asdict(record) for record in records], metrics_fields)
    (config.reports_root / "pipeline_summary.md").write_text(build_summary(records, config), encoding="utf-8")
    (config.reports_root / "roi_consistency_report.md").write_text(build_consistency_report(records), encoding="utf-8")
    (config.reports_root / "failure_analysis.md").write_text(build_failure_report(records), encoding="utf-8")


def build_summary(records: Sequence[VideoRecord], config: Config) -> str:
    total_frames = sum(record.total_frames for record in records)
    detected = sum(record.detected_frames for record in records)
    fallback = sum(record.fallback_frames for record in records)
    invalid = sum(record.invalid_frames for record in records)
    aligned = sum(record.aligned_frames for record in records)
    valid = sum(record.detected_frames + record.fallback_frames for record in records)
    lines = [
        "# Face Detection, Tracking, and Alignment Pipeline Summary",
        "",
        f"Input root: `{config.input_root}`",
        f"Output root: `{config.output_root}`",
        f"Faces root: `{config.faces_root}`",
        f"Annotations root: `{config.annotations_root}`",
        "",
        "## Frame Totals",
        "",
        f"- Total frames processed: {total_frames}",
        f"- Successful detections: {detected}",
        f"- Temporal fallback frames: {fallback}",
        f"- Invalid frames: {invalid}",
        f"- Valid ROI frames: {valid}",
        f"- Detection success rate: {safe_pct(detected, total_frames)}",
        f"- Usable ROI rate: {safe_pct(valid, total_frames)}",
        f"- Alignment success rate: {safe_pct(aligned, valid)}",
        "",
    ]
    return "\n".join(lines)


def build_consistency_report(records: Sequence[VideoRecord]) -> str:
    usable = [record for record in records if record.detection_success_rate > 0 or record.usable_roi_rate > 0]
    if not usable:
        return "# ROI Consistency Report\n\nNo usable ROI sequences were produced.\n"
    raw_jitter = [record.raw_center_jitter_px for record in usable if record.raw_center_jitter_px > 0]
    stable_jitter = [record.stabilized_center_jitter_px for record in usable if record.stabilized_center_jitter_px > 0]
    raw_cv = [record.raw_area_cv for record in usable if record.raw_area_cv > 0]
    stable_cv = [record.stabilized_area_cv for record in usable if record.stabilized_area_cv > 0]
    rotation = [record.mean_abs_rotation_deg for record in usable if record.mean_abs_rotation_deg > 0]
    jitter_gain = 0.0 if safe_mean(raw_jitter) == 0 else max(0.0, (safe_mean(raw_jitter) - safe_mean(stable_jitter)) / safe_mean(raw_jitter) * 100.0)
    cv_gain = 0.0 if safe_mean(raw_cv) == 0 else max(0.0, (safe_mean(raw_cv) - safe_mean(stable_cv)) / safe_mean(raw_cv) * 100.0)
    return "\n".join([
        "# ROI Consistency Report",
        "",
        f"- Videos analyzed: {len(usable)}",
        f"- Mean raw center jitter: {safe_mean(raw_jitter):.2f} px",
        f"- Mean stabilized center jitter: {safe_mean(stable_jitter):.2f} px",
        f"- Jitter reduction: {jitter_gain:.2f}%",
        f"- Mean raw bbox area CV: {safe_mean(raw_cv):.4f}",
        f"- Mean stabilized bbox area CV: {safe_mean(stable_cv):.4f}",
        f"- Area CV reduction: {cv_gain:.2f}%",
        f"- Mean absolute eye-line correction: {safe_mean(rotation):.2f} degrees",
        "",
    ])


def build_failure_report(records: Sequence[VideoRecord]) -> str:
    degraded = [record for record in records if record.status in {"degraded", "failure"}]
    if not degraded:
        return "# Face Failure Analysis\n\nNo degraded face sequences were observed.\n"
    reasons: dict[str, int] = {}
    for record in degraded:
        reasons[record.error_message or record.status] = reasons.get(record.error_message or record.status, 0) + 1
    lines = ["# Face Failure Analysis", "", f"Degraded or failed sequences: {len(degraded)}", "", "## Common Failure Cases", ""]
    for reason, count in sorted(reasons.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {reason}: {count}")
    lines.extend([
        "",
        "## Suggested Manual Inspection Focus",
        "",
        "- Frames with repeated placeholder ROIs and low-confidence detections.",
        "- Videos where temporal fallback dominates because detection fails on many consecutive frames.",
        "- Sequences where alignment falls back to simple resizing instead of keypoint-based normalization.",
        "",
    ])
    return "\n".join(lines)


def safe_pct(numerator: float, denominator: float) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{(numerator / denominator) * 100.0:.2f}%"


def run_pipeline(args: argparse.Namespace) -> int:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.is_relative_to(input_root):
        raise ValueError("Refusing to write outputs inside the frame input root.")

    faces_root = output_root / "faces"
    annotations_root = output_root / "annotations"
    debug_root = output_root / "debug"
    reports_root = faces_root / "reports"
    for folder in (faces_root, annotations_root, debug_root, reports_root):
        folder.mkdir(parents=True, exist_ok=True)

    config = Config(
        input_root=input_root,
        output_root=output_root,
        faces_root=faces_root,
        annotations_root=annotations_root,
        debug_root=debug_root,
        reports_root=reports_root,
        padding=float(args.padding),
        smoothing_alpha=float(args.smoothing_alpha),
        max_temporal_gap=int(args.max_temporal_gap),
        min_bbox_side=int(args.min_bbox_side),
        min_bbox_area_ratio=float(args.min_bbox_area_ratio),
        accept_confidence=float(args.accept_confidence),
        initial_accept_confidence=float(args.initial_accept_confidence),
        output_size=int(args.output_size),
        workers=max(1, int(args.workers)),
        overwrite=bool(args.overwrite),
        continue_on_error=bool(args.continue_on_error),
        debug_success_samples=int(args.debug_success_samples),
        debug_fallback_samples=int(args.debug_fallback_samples),
        debug_missed_samples=int(args.debug_missed_samples),
        debug_alignment_samples=int(args.debug_alignment_samples),
        debug_alignment_fallback_samples=int(args.debug_alignment_fallback_samples),
    )

    logger = configure_logging(reports_root)
    logger.info("Input root: %s", input_root)
    logger.info("Output root: %s", output_root)

    entries = discover_frame_entries(input_root, args.videos)
    logger.info("Videos selected: %d", len(entries))

    sampler = DebugSampler(
        debug_root,
        config.debug_success_samples,
        config.debug_fallback_samples,
        config.debug_missed_samples,
        config.debug_alignment_samples,
        config.debug_alignment_fallback_samples,
    )

    records: list[VideoRecord] = []
    if config.workers <= 1:
        for index, entry in enumerate(entries, start=1):
            logger.info("[%d/%d] Processing %s", index, len(entries), entry.frame_dir.relative_to(input_root))
            try:
                record = process_video(entry, config, logger, sampler)
            except Exception as exc:
                logger.error("Failed %s | %s", entry.video_name, exc)
                record = build_failure_record(entry, config, str(exc))
                if not config.continue_on_error:
                    records.append(record)
                    break
            records.append(record)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            futures = {executor.submit(process_video, entry, config, logger, sampler): entry for entry in entries}
            for completed_index, future in enumerate(as_completed(futures), start=1):
                entry = futures[future]
                logger.info("[%d/%d] Processing %s", completed_index, len(entries), entry.frame_dir.relative_to(input_root))
                try:
                    records.append(future.result())
                except Exception as exc:
                    logger.error("Failed %s | %s", entry.video_name, exc)
                    records.append(build_failure_record(entry, config, str(exc)))
                    if not config.continue_on_error:
                        for pending in futures:
                            pending.cancel()
                        break

    records.sort(key=lambda record: (record.sequence_index, record.split, record.video_name))
    write_reports(records, config)
    logger.info("Reports written to %s", reports_root)
    return 0 if all(record.status in {"success", "degraded", "skipped"} for record in records) else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command != "run":
        raise RuntimeError(f"Unknown command: {args.command}")
    return run_pipeline(args)


def write_annotation_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
