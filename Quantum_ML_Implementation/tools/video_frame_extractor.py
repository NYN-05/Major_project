#!/usr/bin/env python3
"""Video-to-frame extraction pipeline for Celeb-DF-style datasets.

The script preserves the required hierarchy:

    /frames/
        /train/<video_name>/frame_000001.jpg
        /val/<video_name>/frame_000001.jpg
        /test/<video_name>/frame_000001.jpg

It also writes structured reports under `/frames/reports/`:
- extraction_log.csv
- frame_metadata.csv
- per-video JSON manifests
- optional sampling_benchmark.csv and sampling_benchmark_summary.md

Sampling supports either target FPS mode or fixed frame-interval mode.
Frames are written in temporal order and named sequentially with zero-padded indices.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from statistics import mean
from typing import Iterable, Iterator, Optional, Sequence


SPLITS = ("train", "val", "test")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FRAME_EXTENSION = ".jpg"
EPSILON = 1e-9
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_FPS = 5.0


@dataclass(frozen=True)
class VideoEntry:
    split: str
    class_name: str
    label: str
    source_path: Path
    video_name: str
    relative_path: str


@dataclass
class ProbeResult:
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    duration_seconds: Optional[float]
    frame_count: Optional[int]
    method: str


@dataclass
class ExtractionRecord:
    split: str
    class_name: str
    label: str
    video_name: str
    source_path: str
    output_dir: str
    sampling_strategy: str
    requested_sampling_value: float
    effective_sampling_rate: Optional[float]
    original_fps: Optional[float]
    duration_seconds: Optional[float]
    source_frame_count: Optional[int]
    extracted_frame_count: int
    width: Optional[int]
    height: Optional[int]
    status: str
    error_message: str
    processing_seconds: float
    validation_method: str


@dataclass
class BenchmarkRecord:
    split: str
    class_name: str
    label: str
    video_name: str
    source_path: str
    sampling_strategy: str
    requested_sampling_value: float
    original_fps: Optional[float]
    duration_seconds: Optional[float]
    source_frame_count: Optional[int]
    extracted_frame_count: int
    effective_sampling_rate: Optional[float]
    processing_seconds: float
    width: Optional[int]
    height: Optional[int]
    validation_method: str
    error_message: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract temporally ordered video frames with strict dataset split separation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract frames into /frames/<split>/<video_name>/")
    add_common_input_arguments(extract)
    extract.add_argument(
        "--sampling-strategy",
        choices=("fps", "interval"),
        default="fps",
        help="Use target-FPS sampling or fixed frame-interval sampling.",
    )
    extract.add_argument(
        "--target-fps",
        type=float,
        default=DEFAULT_TARGET_FPS,
        help="Target sampling rate when using FPS-based sampling. Edit DEFAULT_TARGET_FPS in this file to change the default.",
    )
    extract.add_argument(
        "--frame-interval",
        type=int,
        default=10,
        help="Sample every Nth frame when using interval-based sampling.",
    )
    extract.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved frames.",
    )
    extract.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted frames for a video if the output folder already exists.",
    )
    extract.add_argument(
        "--videos",
        nargs="*",
        help="Optional list of specific videos to process (relative to input root or absolute paths).",
    )
    extract.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining videos if one video fails.",
    )
    extract.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker threads to use for extraction.",
    )

    benchmark = subparsers.add_parser("benchmark", help="Compare multiple FPS sampling rates without writing frames.")
    add_common_input_arguments(benchmark)
    benchmark.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=[1.0, 5.0, 10.0],
        help="FPS rates to compare in dry-run benchmark mode.",
    )
    benchmark.add_argument(
        "--videos",
        nargs="*",
        help="Optional list of specific videos to benchmark (relative to input root or absolute paths).",
    )

    return parser


def add_common_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Dataset root that contains train/val/test folders. Defaults to the discovered Celeb-DF split root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root for frames. Defaults to <input-root>/frames.",
    )


def configure_logging(reports_root: Path) -> logging.Logger:
    reports_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("video_frame_extractor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(reports_root / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def resolve_dataset_root(explicit_root: Optional[Path]) -> Path:
    if explicit_root is not None:
        explicit_root = explicit_root.resolve()
        if (explicit_root / "train").is_dir() and (explicit_root / "val").is_dir() and (explicit_root / "test").is_dir():
            return explicit_root
        raise FileNotFoundError(f"Input root does not contain train/val/test folders: {explicit_root}")

    cwd = Path.cwd()
    if (cwd / "train").is_dir() and (cwd / "val").is_dir() and (cwd / "test").is_dir():
        return cwd

    script_root = Path(__file__).resolve().parent.parent
    if (script_root / "train").is_dir() and (script_root / "val").is_dir() and (script_root / "test").is_dir():
        return script_root

    nested = cwd / "celebdf_split"
    if (nested / "train").is_dir() and (nested / "val").is_dir() and (nested / "test").is_dir():
        return nested

    raise FileNotFoundError("Could not locate a dataset root with train/val/test folders.")


def label_from_class_name(class_name: str) -> str:
    lowered = class_name.lower()
    if "real" in lowered:
        return "real"
    if "fake" in lowered or "synth" in lowered or "manip" in lowered:
        return "fake"
    return "unknown"


def normalize_video_path(input_root: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = input_root / candidate
    return candidate.resolve()


def discover_videos(input_root: Path, video_values: Optional[Sequence[str]]) -> list[VideoEntry]:
    if video_values:
        candidates = [normalize_video_path(input_root, value) for value in video_values]
    else:
        candidates = []
        for split in SPLITS:
            split_dir = input_root / split
            if not split_dir.is_dir():
                continue
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                for video_path in sorted(class_dir.rglob("*")):
                    if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                        candidates.append(video_path.resolve())

    entries: list[VideoEntry] = []
    for source_path in candidates:
        if not source_path.exists():
            raise FileNotFoundError(f"Video not found: {source_path}")
        try:
            relative_parts = source_path.relative_to(input_root).parts
        except ValueError as exc:
            raise ValueError(f"Video is not inside the input root: {source_path}") from exc

        if len(relative_parts) < 3:
            raise ValueError(f"Expected split/class/video structure, got: {source_path.relative_to(input_root)}")

        split = relative_parts[0]
        if split not in SPLITS:
            raise ValueError(f"Video is not inside one of the expected split folders: {source_path}")

        class_name = relative_parts[1]
        video_name = source_path.stem
        entries.append(
            VideoEntry(
                split=split,
                class_name=class_name,
                label=label_from_class_name(class_name),
                source_path=source_path,
                video_name=video_name,
                relative_path=str(source_path.relative_to(input_root)).replace(os.sep, "/"),
            )
        )

    return entries


def run_ffprobe(video_path: Path) -> ProbeResult:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is not available")

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}
    format_info = payload.get("format") or {}

    width = stream.get("width")
    height = stream.get("height")

    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))

    duration_text = stream.get("duration") or format_info.get("duration")
    duration_seconds = parse_float(duration_text)

    frame_count_text = stream.get("nb_frames")
    frame_count = int(frame_count_text) if isinstance(frame_count_text, str) and frame_count_text.isdigit() else None

    return ProbeResult(
        width=width,
        height=height,
        fps=fps,
        duration_seconds=duration_seconds,
        frame_count=frame_count,
        method="ffprobe",
    )


def run_opencv_probe(video_path: Path) -> ProbeResult:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is not available") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    fps_value = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps_value if fps_value > 0 else None
    frame_count_value = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_count = frame_count_value if frame_count_value > 0 else None
    duration_seconds = None
    if fps and frame_count:
        duration_seconds = frame_count / fps
    capture.release()
    return ProbeResult(width=width, height=height, fps=fps, duration_seconds=duration_seconds, frame_count=frame_count, method="opencv")


def probe_video(video_path: Path) -> ProbeResult:
    try:
        return run_ffprobe(video_path)
    except Exception:
        return run_opencv_probe(video_path)


def parse_rate(value: object) -> Optional[float]:
    if not isinstance(value, str) or not value or value == "0/0":
        return None
    if "/" in value:
        numerator_text, denominator_text = value.split("/", 1)
        denominator = parse_float(denominator_text)
        if denominator in (None, 0.0):
            return None
        numerator = parse_float(numerator_text)
        if numerator is None:
            return None
        return numerator / denominator
    return parse_float(value)


def parse_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def ensure_clean_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"Output directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)


def promote_directory(temp_dir: Path, final_output_dir: Path, attempts: int = 3) -> None:
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        if final_output_dir.exists():
            shutil.rmtree(final_output_dir, ignore_errors=True)
        try:
            temp_dir.rename(final_output_dir)
            return
        except PermissionError as exc:
            last_error = exc
        except OSError as exc:
            last_error = exc

        if attempt < attempts:
            time.sleep(0.1 * attempt)

    if final_output_dir.exists():
        shutil.rmtree(final_output_dir, ignore_errors=True)
    try:
        shutil.copytree(temp_dir, final_output_dir)
    except Exception:
        if last_error is not None:
            raise last_error
        raise


def verify_sequential_frames(frame_dir: Path, video_name: str, expected_count: int) -> bool:
    frame_files = sorted(frame_dir.glob(f"{video_name}_frame_*.jpg"))
    if len(frame_files) != expected_count:
        return False
    for index, frame_file in enumerate(frame_files, start=1):
        expected_name = f"{video_name}_frame_{index:06d}.jpg"
        if frame_file.name != expected_name:
            return False
    return True


def write_csv_rows(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest(manifest_path: Path, record: ExtractionRecord) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    payload["source_path"] = str(record.source_path)
    payload["output_dir"] = str(record.output_dir)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def ensure_output_outside_dataset(output_root: Path, input_root: Path) -> None:
    resolved_output = output_root.resolve()
    resolved_input = input_root.resolve()
    if resolved_output.is_relative_to(resolved_input):
        raise ValueError("Refusing to write extracted frames inside the dataset folder.")


def sample_iterator(strategy: str, requested_value: float, source_fps: Optional[float]) -> Iterator[bool]:
    if strategy == "interval":
        interval = max(1, int(round(requested_value)))
        frame_index = 0
        while True:
            yield frame_index % interval == 0
            frame_index += 1
    if source_fps is None or source_fps <= 0:
        raise ValueError("A valid source FPS is required for FPS-based sampling.")
    if requested_value <= 0:
        raise ValueError("Target FPS must be positive.")

    next_sample_time = 0.0
    frame_index = 0
    time_step = 1.0 / requested_value
    while True:
        current_time = frame_index / source_fps
        should_sample = current_time + EPSILON >= next_sample_time
        if should_sample:
            next_sample_time += time_step
        frame_index += 1
        yield should_sample
def extract_video(
    entry: VideoEntry,
    input_root: Path,
    output_root: Path,
    reports_root: Path,
    sampling_strategy: str,
    requested_sampling_value: float,
    jpeg_quality: int,
    overwrite: bool,
    dry_run: bool = False,
) -> ExtractionRecord:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required for frame extraction") from exc

    probe = probe_video(entry.source_path)
    output_dir = output_root / entry.split / entry.video_name
    manifest_dir = reports_root / "manifests" / entry.split
    manifest_path = manifest_dir / f"{entry.video_name}.json"

    if not dry_run and output_dir.exists():
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    existing_manifest = json.load(handle)

                if str(existing_manifest.get("source_path")) == str(entry.source_path):
                    expected_count = int(existing_manifest.get("extracted_frame_count") or 0)
                    if expected_count > 0 and verify_sequential_frames(output_dir, entry.video_name, expected_count) and not overwrite:
                        return ExtractionRecord(
                            split=entry.split,
                            class_name=entry.class_name,
                            label=entry.label,
                            video_name=entry.video_name,
                            source_path=str(entry.source_path),
                            output_dir=str(output_dir),
                            sampling_strategy=sampling_strategy,
                            requested_sampling_value=requested_sampling_value,
                            effective_sampling_rate=parse_float(existing_manifest.get("effective_sampling_rate")),
                            original_fps=parse_float(existing_manifest.get("original_fps")),
                            duration_seconds=parse_float(existing_manifest.get("duration_seconds")),
                            source_frame_count=int(existing_manifest.get("source_frame_count")) if existing_manifest.get("source_frame_count") is not None else probe.frame_count,
                            extracted_frame_count=expected_count,
                            width=int(existing_manifest.get("width")) if existing_manifest.get("width") is not None else probe.width,
                            height=int(existing_manifest.get("height")) if existing_manifest.get("height") is not None else probe.height,
                            status="skipped",
                            error_message="Already extracted and validated.",
                            processing_seconds=0.0,
                            validation_method=probe.method,
                        )
            except Exception:
                pass
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
        manifest_path.unlink(missing_ok=True)

    if not dry_run:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)

    capture = cv2.VideoCapture(str(entry.source_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {entry.source_path}")

    source_fps = probe.fps
    sampler = sample_iterator(sampling_strategy, requested_sampling_value, source_fps)
    extracted_count = 0
    start_time = time.perf_counter()
    temp_dir = None

    if not dry_run:
        temp_dir = output_dir.parent / f".{entry.video_name}__tmp_{os.getpid()}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if next(sampler):
                extracted_count += 1
                if not dry_run and temp_dir is not None:
                    frame_name = f"{entry.video_name}_frame_{extracted_count:06d}{FRAME_EXTENSION}"
                    frame_path = temp_dir / frame_name
                    saved = cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                    if not saved:
                        raise RuntimeError(f"Failed to write frame: {frame_path}")

        if extracted_count == 0:
            raise RuntimeError("No frames were extracted.")

        duration_seconds = probe.duration_seconds
        if duration_seconds is None and probe.frame_count and probe.fps and probe.fps > 0:
            duration_seconds = probe.frame_count / probe.fps
        effective_rate = extracted_count / duration_seconds if duration_seconds and duration_seconds > 0 else None
        elapsed_seconds = time.perf_counter() - start_time

        if not dry_run and temp_dir is not None:
            if not verify_sequential_frames(temp_dir, entry.video_name, extracted_count):
                raise RuntimeError("Frame sequence verification failed.")
            promote_directory(temp_dir, output_dir)

        record = ExtractionRecord(
            split=entry.split,
            class_name=entry.class_name,
            label=entry.label,
            video_name=entry.video_name,
            source_path=str(entry.source_path),
            output_dir=str(output_dir),
            sampling_strategy=sampling_strategy,
            requested_sampling_value=requested_sampling_value,
            effective_sampling_rate=effective_rate,
            original_fps=probe.fps,
            duration_seconds=duration_seconds,
            source_frame_count=probe.frame_count,
            extracted_frame_count=extracted_count,
            width=probe.width,
            height=probe.height,
            status="success",
            error_message="",
            processing_seconds=elapsed_seconds,
            validation_method="opencv",
        )

        if not dry_run:
            write_manifest(manifest_path, record)
        return record

    finally:
        capture.release()
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def benchmark_video(
    entry: VideoEntry,
    rates: Sequence[float],
) -> list[BenchmarkRecord]:
    probe = probe_video(entry.source_path)
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required for benchmarking") from exc

    results: list[BenchmarkRecord] = []
    for rate in rates:
        capture = cv2.VideoCapture(str(entry.source_path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {entry.source_path}")

        start_time = time.perf_counter()
        extracted_count = 0
        source_index = -1
        error_message = ""
        try:
            sampler = sample_iterator("fps", float(rate), probe.fps)
            while True:
                ok, _frame = capture.read()
                if not ok:
                    break
                source_index += 1
                if next(sampler):
                    extracted_count += 1

            if extracted_count == 0:
                raise RuntimeError("No frames were extracted.")

            elapsed_seconds = time.perf_counter() - start_time
            effective_rate = None
            if probe.duration_seconds and probe.duration_seconds > 0:
                effective_rate = extracted_count / probe.duration_seconds

            results.append(
                BenchmarkRecord(
                    split=entry.split,
                    class_name=entry.class_name,
                    label=entry.label,
                    video_name=entry.video_name,
                    source_path=str(entry.source_path),
                    sampling_strategy="fps",
                    requested_sampling_value=float(rate),
                    original_fps=probe.fps,
                    duration_seconds=probe.duration_seconds,
                    source_frame_count=probe.frame_count,
                    extracted_frame_count=extracted_count,
                    effective_sampling_rate=effective_rate,
                    processing_seconds=elapsed_seconds,
                    width=probe.width,
                    height=probe.height,
                    validation_method=probe.method,
                    error_message=error_message,
                )
            )
        except Exception as exc:
            elapsed_seconds = time.perf_counter() - start_time
            results.append(
                BenchmarkRecord(
                    split=entry.split,
                    class_name=entry.class_name,
                    label=entry.label,
                    video_name=entry.video_name,
                    source_path=str(entry.source_path),
                    sampling_strategy="fps",
                    requested_sampling_value=float(rate),
                    original_fps=probe.fps,
                    duration_seconds=probe.duration_seconds,
                    source_frame_count=probe.frame_count,
                    extracted_frame_count=0,
                    effective_sampling_rate=None,
                    processing_seconds=elapsed_seconds,
                    width=probe.width,
                    height=probe.height,
                    validation_method=probe.method,
                    error_message=str(exc),
                )
            )
        finally:
            capture.release()

    return results


def build_summary_markdown(records: Sequence[BenchmarkRecord]) -> str:
    lines = [
        "# Sampling Benchmark Summary",
        "",
        "This summary compares FPS-based sampling rates using the same source videos.",
        "Lower FPS means fewer frames, less compute, and larger temporal gaps.",
        "Higher FPS means denser temporal evidence and higher storage/compute cost.",
        "",
        "| Rate (FPS) | Videos | Avg Extracted Frames | Avg Processing Time (s) | Avg Effective FPS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    grouped: dict[float, list[BenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault(record.requested_sampling_value, []).append(record)

    for rate in sorted(grouped):
        group = [item for item in grouped[rate] if item.error_message == ""]
        if not group:
            continue
        avg_frames = mean(item.extracted_frame_count for item in group)
        avg_time = mean(item.processing_seconds for item in group)
        effective_rates = [item.effective_sampling_rate for item in group if item.effective_sampling_rate is not None]
        avg_effective_rate = mean(effective_rates) if effective_rates else 0.0
        lines.append(f"| {rate:g} | {len(group)} | {avg_frames:.2f} | {avg_time:.2f} | {avg_effective_rate:.2f} |")

    lines.extend(
        [
            "",
            "Recommended baseline for the project: 5 FPS.",
            "It is the most practical middle ground for low-resolution KYC clips because it keeps temporal coverage while avoiding the cost of denser extraction.",
            "",
        ]
    )
    return "\n".join(lines)


def run_extract(args: argparse.Namespace) -> int:
    input_root = resolve_dataset_root(args.input_root)
    output_root = args.output_root.resolve() if args.output_root else (WORKSPACE_ROOT / "frames")
    ensure_output_outside_dataset(output_root, input_root)
    reports_root = output_root / "reports"
    logger = configure_logging(reports_root)

    logger.info("Input root: %s", input_root)
    logger.info("Output root: %s", output_root)

    entries = discover_videos(input_root, args.videos)
    logger.info("Videos selected: %d", len(entries))

    records: list[ExtractionRecord] = []
    sampling_value = args.target_fps if args.sampling_strategy == "fps" else float(args.frame_interval)
    extractor = partial(
        extract_video,
        input_root=input_root,
        output_root=output_root,
        reports_root=reports_root,
        sampling_strategy=args.sampling_strategy,
        requested_sampling_value=sampling_value,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
        dry_run=False,
    )

    if args.workers <= 1:
        for index, entry in enumerate(entries, start=1):
            logger.info("[%d/%d] Processing %s", index, len(entries), entry.relative_path)
            try:
                record = extractor(entry=entry)
                logger.info(
                    "Completed %s | extracted=%d | status=%s | effective_fps=%s",
                    entry.video_name,
                    record.extracted_frame_count,
                    record.status,
                    f"{record.effective_sampling_rate:.2f}" if record.effective_sampling_rate is not None else "n/a",
                )
                records.append(record)
            except Exception as exc:
                logger.error("Failed %s | %s", entry.video_name, exc)
                records.append(build_failure_record(entry, output_root, args, str(exc)))
                if not args.continue_on_error:
                    break
    else:
        logger.info("Parallel workers: %d", args.workers)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(extractor, entry=entry): entry for entry in entries}
            completed_count = 0
            for future in as_completed(future_map):
                entry = future_map[future]
                completed_count += 1
                logger.info("[%d/%d] Processing %s", completed_count, len(entries), entry.relative_path)
                try:
                    record = future.result()
                    logger.info(
                        "Completed %s | extracted=%d | status=%s | effective_fps=%s",
                        entry.video_name,
                        record.extracted_frame_count,
                        record.status,
                        f"{record.effective_sampling_rate:.2f}" if record.effective_sampling_rate is not None else "n/a",
                    )
                    records.append(record)
                except Exception as exc:
                    logger.error("Failed %s | %s", entry.video_name, exc)
                    records.append(build_failure_record(entry, output_root, args, str(exc)))
                    if not args.continue_on_error:
                        for pending in future_map:
                            pending.cancel()
                        break

    write_extract_reports(records, reports_root)
    write_pipeline_summary(records, output_root)
    return 0 if all(record.status in {"success", "skipped"} for record in records) else 1


def build_failure_record(entry: VideoEntry, output_root: Path, args: argparse.Namespace, error_message: str) -> ExtractionRecord:
    return ExtractionRecord(
        split=entry.split,
        class_name=entry.class_name,
        label=entry.label,
        video_name=entry.video_name,
        source_path=str(entry.source_path),
        output_dir=str(output_root / entry.split / entry.video_name),
        sampling_strategy=args.sampling_strategy,
        requested_sampling_value=args.target_fps if args.sampling_strategy == "fps" else float(args.frame_interval),
        effective_sampling_rate=None,
        original_fps=None,
        duration_seconds=None,
        source_frame_count=None,
        extracted_frame_count=0,
        width=None,
        height=None,
        status="failure",
        error_message=error_message,
        processing_seconds=0.0,
        validation_method="unknown",
    )


def write_extract_reports(records: Sequence[ExtractionRecord], reports_root: Path) -> None:
    metadata_fields = [
        "split",
        "class_name",
        "label",
        "video_name",
        "source_path",
        "output_dir",
        "sampling_strategy",
        "requested_sampling_value",
        "effective_sampling_rate",
        "original_fps",
        "duration_seconds",
        "source_frame_count",
        "extracted_frame_count",
        "width",
        "height",
        "status",
        "error_message",
        "processing_seconds",
        "validation_method",
    ]
    log_fields = [
        "split",
        "class_name",
        "label",
        "video_name",
        "source_path",
        "sampling_strategy",
        "requested_sampling_value",
        "extracted_frame_count",
        "status",
        "error_message",
        "processing_seconds",
    ]

    metadata_rows = [asdict(record) for record in records]
    log_rows = [
        {
            "split": record.split,
            "class_name": record.class_name,
            "label": record.label,
            "video_name": record.video_name,
            "source_path": record.source_path,
            "sampling_strategy": record.sampling_strategy,
            "requested_sampling_value": record.requested_sampling_value,
            "extracted_frame_count": record.extracted_frame_count,
            "status": record.status,
            "error_message": record.error_message,
            "processing_seconds": record.processing_seconds,
        }
        for record in records
    ]

    write_csv_rows(reports_root / "frame_metadata.csv", metadata_rows, metadata_fields)
    write_csv_rows(reports_root / "extraction_log.csv", log_rows, log_fields)

    manifest_root = reports_root / "manifests"
    for record in records:
        if record.status == "failure":
            continue
        manifest_path = manifest_root / record.split / f"{record.video_name}.json"
        write_manifest(manifest_path, record)


def write_pipeline_summary(records: Sequence[ExtractionRecord], output_root: Path) -> None:
    summary_lines = [
        "# Frame Extraction Pipeline Summary",
        "",
        f"Output root: `{output_root}`",
        "",
        "## Status Counts",
        "",
    ]

    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1

    for status in sorted(status_counts):
        summary_lines.append(f"- {status}: {status_counts[status]}")

    summary_lines.extend(
        [
            "",
            "## Structure Guarantees",
            "",
            "- Split folders are kept separate under frames/train, frames/val, and frames/test.",
            "- Each video gets its own folder named exactly after the video filename without extension.",
            "- Frames are saved in temporal order and named sequentially with zero-padded indices.",
            "- Reports and manifests are stored under frames/reports/.",
            "",
        ]
    )
    (output_root / "reports" / "pipeline_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> int:
    input_root = resolve_dataset_root(args.input_root)
    output_root = args.output_root.resolve() if args.output_root else (WORKSPACE_ROOT / "frames")
    ensure_output_outside_dataset(output_root, input_root)
    reports_root = output_root / "reports"
    logger = configure_logging(reports_root)

    logger.info("Input root: %s", input_root)
    logger.info("Output root: %s", output_root)

    entries = discover_videos(input_root, args.videos)
    logger.info("Benchmark videos selected: %d", len(entries))

    benchmark_records: list[BenchmarkRecord] = []
    for index, entry in enumerate(entries, start=1):
        logger.info("[%d/%d] Benchmarking %s", index, len(entries), entry.relative_path)
        try:
            benchmark_records.extend(benchmark_video(entry, args.rates))
        except Exception as exc:
            logger.error("Benchmark failed for %s | %s", entry.video_name, exc)
            benchmark_records.append(
                BenchmarkRecord(
                    split=entry.split,
                    class_name=entry.class_name,
                    label=entry.label,
                    video_name=entry.video_name,
                    source_path=str(entry.source_path),
                    sampling_strategy="fps",
                    requested_sampling_value=float(args.rates[0]),
                    original_fps=None,
                    duration_seconds=None,
                    source_frame_count=None,
                    extracted_frame_count=0,
                    effective_sampling_rate=None,
                    processing_seconds=0.0,
                    width=None,
                    height=None,
                    validation_method="unknown",
                    error_message=str(exc),
                )
            )

    benchmark_fields = [
        "split",
        "class_name",
        "label",
        "video_name",
        "source_path",
        "sampling_strategy",
        "requested_sampling_value",
        "original_fps",
        "duration_seconds",
        "source_frame_count",
        "extracted_frame_count",
        "effective_sampling_rate",
        "processing_seconds",
        "width",
        "height",
        "validation_method",
        "error_message",
    ]
    benchmark_rows = [asdict(record) for record in benchmark_records]
    write_csv_rows(reports_root / "sampling_benchmark.csv", benchmark_rows, benchmark_fields)
    (reports_root / "sampling_benchmark_summary.md").write_text(build_summary_markdown(benchmark_records), encoding="utf-8")

    logger.info("Benchmark reports written to %s", reports_root)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        if args.sampling_strategy == "fps" and args.target_fps <= 0:
            raise ValueError("--target-fps must be positive.")
        if args.sampling_strategy == "interval" and args.frame_interval <= 0:
            raise ValueError("--frame-interval must be positive.")
        return run_extract(args)

    if args.command == "benchmark":
        if not args.rates:
            raise ValueError("At least one sampling rate is required for benchmarking.")
        if any(rate <= 0 for rate in args.rates):
            raise ValueError("All benchmark rates must be positive.")
        return run_benchmark(args)

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())