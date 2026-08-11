"""
extract_dataset_features.py
===========================
Batch processes the available real and deepfake datasets to extract rPPG
features. The script now supports the new folder-based DFDC layout at
archive/DFDC_Dataset/ in addition to the older CSV-based archive (1)/ layout.

Outputs a CSV file dataset_features.csv for training the classifier.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rppg import RPPGPipeline


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _iter_video_files(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def _add_sample(samples: List[Tuple[int, Path, str]], label: int, path: Path, source: str) -> None:
    if path.exists() and path.is_file():
        samples.append((label, path, source))


def collect_samples(max_per_class: Optional[int] = None) -> List[Tuple[int, Path, str]]:
    root = _repo_root()
    samples: List[Tuple[int, Path, str]] = []

    new_dataset_root = root / "archive" / "DFDC_Dataset"
    if new_dataset_root.exists():
        fake_dir = new_dataset_root / "Fake"
        real_dir = new_dataset_root / "Real"

        fake_files = list(_iter_video_files(fake_dir)) if fake_dir.exists() else []
        real_files = list(_iter_video_files(real_dir)) if real_dir.exists() else []

        if fake_files or real_files:
            if max_per_class is not None:
                fake_files = fake_files[:max_per_class]
                real_files = real_files[:max_per_class]

            for path in fake_files:
                _add_sample(samples, 1, path, "archive/DFDC_Dataset/Fake")
            for path in real_files:
                _add_sample(samples, 0, path, "archive/DFDC_Dataset/Real")

    legacy_root = root / "archive (1)"
    legacy_csv = legacy_root / "DeepFake Videos Dataset.csv"
    if legacy_csv.exists():
        legacy_df = pd.read_csv(legacy_csv)

        if "deepfake" in legacy_df.columns:
            fake_paths = [legacy_root / str(value) for value in legacy_df["deepfake"].dropna().tolist()]
            if max_per_class is not None:
                fake_paths = fake_paths[:max_per_class]
            for path in fake_paths:
                _add_sample(samples, 1, path, "archive (1)/DeepFake Videos Dataset.csv")

        if "video" in legacy_df.columns:
            real_paths = [legacy_root / str(value) for value in legacy_df["video"].dropna().tolist()]
            if max_per_class is not None:
                real_paths = real_paths[:max_per_class]
            for path in real_paths:
                _add_sample(samples, 0, path, "archive (1)/DeepFake Videos Dataset.csv")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract rPPG features from deepfake and real video datasets.")
    parser.add_argument("--method", default="POS", choices=["POS", "CHROM"], help="rPPG reconstruction method")
    parser.add_argument("--target-fps", type=float, default=None, help="Optional target FPS for sampling")
    parser.add_argument("--blur-threshold", type=float, default=15.0, help="Minimum Laplacian variance to keep a frame")
    parser.add_argument("--min-usable-frames", type=int, default=48, help="Minimum usable frames required per clip")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap for each label when extracting features")
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    samples = collect_samples(max_per_class=args.max_per_class)
    if not samples:
        print("No dataset videos were found. Check archive/DFDC_Dataset or archive (1).")
        return

    root = _repo_root()
    out_csv_path = Path(args.output) if args.output else root / "dataset_features.csv"

    pipeline = RPPGPipeline(
        method=args.method,
        target_fps=args.target_fps,
        blur_threshold=args.blur_threshold,
        min_usable_frames=args.min_usable_frames,
    )

    features_list = []
    for label, video_path, source in samples:
        label_name = "Fake" if label == 1 else "Real"
        print(f"Processing {label_name}: {video_path}")
        try:
            result = pipeline.process_video(str(video_path))
        except Exception as exc:
            print(f"  -> Error processing {video_path.name}: {exc}")
            continue

        if result.features is None:
            print("  -> Failed to extract features (insufficient frames or no face).")
            continue

        feat_dict = result.features.to_dict()
        feat_dict["label"] = label
        feat_dict["video_path"] = str(video_path.relative_to(root)) if video_path.is_relative_to(root) else str(video_path)
        feat_dict["source"] = source
        features_list.append(feat_dict)

    if not features_list:
        print("No features extracted from any videos.")
        return

    out_df = pd.DataFrame(features_list)
    out_df.to_csv(out_csv_path, index=False)
    print(f"\nSuccessfully extracted features for {len(out_df)} videos.")
    print(f"Features saved to {out_csv_path}")


if __name__ == "__main__":
    main()
