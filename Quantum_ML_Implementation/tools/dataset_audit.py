#!/usr/bin/env python3
"""Audit a Celeb-DF style dataset split.

The script scans all video files under train/val/test, validates that each file can be
opened, collects metadata, and writes shareable CSV/Markdown reports.

Reports generated in `dataset_audit_reports/`:
- `corrupt_unreadable_files.csv`
- `video_metadata.csv`
- `dataset_inventory.csv`
- `dataset_structure_report.md`

The script prefers `ffprobe` when available, and falls back to OpenCV if needed.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SPLITS = ("train", "val", "test")
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class VideoRecord:
    split: str
    class_name: str
    relative_path: str
    filename: str
    label: str
    size_bytes: int
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    duration_seconds: Optional[float]
    readable: bool
    validation_method: str
    error_message: str


def find_dataset_root(start: Path) -> Path:
    if (start / "train").is_dir() and (start / "val").is_dir() and (start / "test").is_dir():
        return start
    nested = start / "celebdf_split"
    if (nested / "train").is_dir() and (nested / "val").is_dir() and (nested / "test").is_dir():
        return nested
    raise FileNotFoundError("Could not find a dataset root with train/val/test folders.")


def iter_video_files(dataset_root: Path) -> Iterable[Path]:
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for video_path in sorted(class_dir.rglob("*")):
                if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                    yield video_path


def run_ffprobe(video_path: Path) -> tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
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
        "stream=width,height,r_frame_rate,duration",
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
    formats = payload.get("format") or {}

    width = stream.get("width")
    height = stream.get("height")

    fps = None
    rate = stream.get("r_frame_rate")
    if isinstance(rate, str) and rate and rate != "0/0":
        numerator, denominator = rate.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value != 0:
            fps = float(numerator) / denominator_value

    duration_text = stream.get("duration") or formats.get("duration")
    duration = float(duration_text) if duration_text not in (None, "", "N/A") else None
    return width, height, fps, duration


def run_opencv(video_path: Path) -> tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is not available") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    fps_value = capture.get(cv2.CAP_PROP_FPS)
    fps = float(fps_value) if fps_value and fps_value > 0 else None
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = None
    if fps and frame_count and frame_count > 0:
        duration = float(frame_count) / fps
    capture.release()
    return width, height, fps, duration


def inspect_video(video_path: Path) -> tuple[bool, str, Optional[int], Optional[int], Optional[float], Optional[float], str]:
    validation_method = "ffprobe"
    try:
        width, height, fps, duration = run_ffprobe(video_path)
        readable = True
        error_message = ""
    except Exception as ffprobe_error:
        validation_method = "opencv"
        try:
            width, height, fps, duration = run_opencv(video_path)
            readable = True
            error_message = ""
        except Exception as opencv_error:
            readable = False
            width = height = None
            fps = duration = None
            error_message = f"ffprobe: {ffprobe_error}; opencv: {opencv_error}"

    return readable, validation_method, width, height, fps, duration, error_message


def build_records(dataset_root: Path) -> list[VideoRecord]:
    records: list[VideoRecord] = []

    for video_path in iter_video_files(dataset_root):
        relative_parts = video_path.relative_to(dataset_root).parts
        split = relative_parts[0]
        class_name = relative_parts[1]
        label = "real" if "real" in class_name.lower() else "fake"
        readable, validation_method, width, height, fps, duration, error_message = inspect_video(video_path)
        records.append(
            VideoRecord(
                split=split,
                class_name=class_name,
                relative_path=str(video_path.relative_to(dataset_root)).replace(os.sep, "/"),
                filename=video_path.name,
                label=label,
                size_bytes=video_path.stat().st_size,
                width=width,
                height=height,
                fps=fps,
                duration_seconds=duration,
                readable=readable,
                validation_method=validation_method,
                error_message=error_message,
            )
        )

    return records


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_structure_report(dataset_root: Path, records: list[VideoRecord]) -> str:
    lines = [
        "# Dataset Structure Report",
        "",
        f"Dataset root: `{dataset_root}`",
        "",
        "## Split Summary",
        "",
    ]

    for split in SPLITS:
        split_records = [record for record in records if record.split == split]
        class_counts: dict[str, int] = {}
        for record in split_records:
            class_counts[record.class_name] = class_counts.get(record.class_name, 0) + 1
        lines.append(f"- **{split}**: {len(split_records)} videos")
        for class_name in sorted(class_counts):
            lines.append(f"  - {class_name}: {class_counts[class_name]}")

    lines.extend(
        [
            "",
            "## Compliance Notes",
            "",
            "- Master structure present: yes",
            "- Train/val/test split present: yes",
            "- Inventory generated: yes",
            "- Metadata extracted: yes",
            "- Unreadable file scan completed: yes",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    start_dir = Path.cwd()
    try:
        dataset_root = find_dataset_root(start_dir)
    except FileNotFoundError:
        dataset_root = find_dataset_root(Path(__file__).resolve().parent.parent)

    records = build_records(dataset_root)
    output_dir = WORKSPACE_ROOT / "dataset_audit_reports"
    if output_dir.resolve().is_relative_to(dataset_root.resolve()):
        raise ValueError("Refusing to write audit reports inside the dataset folder.")
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows = [
        {
            "split": record.split,
            "class_name": record.class_name,
            "relative_path": record.relative_path,
            "filename": record.filename,
            "label": record.label,
        }
        for record in records
    ]

    metadata_rows = [asdict(record) for record in records]
    unreadable_rows = [asdict(record) for record in records if not record.readable]

    write_csv(
        output_dir / "dataset_inventory.csv",
        inventory_rows,
        ["split", "class_name", "relative_path", "filename", "label"],
    )
    write_csv(
        output_dir / "video_metadata.csv",
        metadata_rows,
        [
            "split",
            "class_name",
            "relative_path",
            "filename",
            "label",
            "size_bytes",
            "width",
            "height",
            "fps",
            "duration_seconds",
            "readable",
            "validation_method",
            "error_message",
        ],
    )
    write_csv(
        output_dir / "corrupt_unreadable_files.csv",
        unreadable_rows,
        [
            "split",
            "class_name",
            "relative_path",
            "filename",
            "label",
            "size_bytes",
            "width",
            "height",
            "fps",
            "duration_seconds",
            "readable",
            "validation_method",
            "error_message",
        ],
    )

    structure_report = build_structure_report(dataset_root, records)
    (output_dir / "dataset_structure_report.md").write_text(structure_report, encoding="utf-8")

    summary = {
        "dataset_root": str(dataset_root),
        "total_videos": len(records),
        "unreadable_videos": len(unreadable_rows),
        "reports": [
            str(output_dir / "dataset_inventory.csv"),
            str(output_dir / "video_metadata.csv"),
            str(output_dir / "corrupt_unreadable_files.csv"),
            str(output_dir / "dataset_structure_report.md"),
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())