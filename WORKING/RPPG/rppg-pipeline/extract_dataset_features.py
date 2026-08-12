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
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rppg import RPPGPipeline  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

_WORKER: dict = {}


def _init_worker(method: str, target_fps: Optional[float], blur_threshold: float, min_usable_frames: int) -> None:
    """Per-process initializer: creates one RPPGPipeline per worker."""
    _WORKER["pipeline"] = RPPGPipeline(
        method=method,
        target_fps=target_fps,
        blur_threshold=blur_threshold,
        min_usable_frames=min_usable_frames,
    )


def _process_one(item: Tuple[int, Path, str, Path]) -> dict:
    """Process a single video inside a worker process. Returns a feature
    dict, or a dict with an 'error'/'no_features' marker."""
    label, video_path, source, root = item
    entry: dict = {"label": label, "video_path": str(video_path), "source": source}
    try:
        result = _WORKER["pipeline"].process_video(str(video_path))
    except Exception as exc:  # noqa: BLE001 - record and continue
        entry["error"] = f"{type(exc).__name__}: {exc}"
        return entry
    if result.features is None:
        entry["no_features"] = True
        entry["usable_frames"] = result.n_frames_usable
        return entry
    feat = result.features.to_dict()
    feat["label"] = label
    feat["video_path"] = str(video_path.relative_to(root)) if video_path.is_relative_to(root) else str(video_path)
    feat["source"] = source
    return feat


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _output_dir() -> Path:
    return _repo_root().parent / "output" / "rppg"


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
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes (0 = all CPU cores)")
    args = parser.parse_args()

    samples = collect_samples(max_per_class=args.max_per_class)
    if not samples:
        print("No dataset videos were found. Check archive/DFDC_Dataset or archive (1).")
        return

    root = _repo_root()
    out_csv_path = Path(args.output) if args.output else _output_dir() / "dataset_features.csv"
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if args.workers == 0:
        n_workers = max(1, os.cpu_count() or 1)
    else:
        n_workers = args.workers

    items = [(label, video_path, source, root) for label, video_path, source in samples]

    features_list = []
    stats = {"processed": 0, "no_features": 0, "failed": 0}
    t_start = time.time()

    if n_workers <= 1:
        pipeline = RPPGPipeline(
            method=args.method,
            target_fps=args.target_fps,
            blur_threshold=args.blur_threshold,
            min_usable_frames=args.min_usable_frames,
        )
        for label, video_path, source in samples:
            label_name = "Fake" if label == 1 else "Real"
            print(f"Processing {label_name}: {video_path}")
            try:
                result = pipeline.process_video(str(video_path))
            except Exception as exc:
                print(f"  -> Error processing {video_path.name}: {exc}")
                stats["failed"] += 1
                continue
            if result.features is None:
                print("  -> Failed to extract features (insufficient frames or no face).")
                stats["no_features"] += 1
                continue
            feat_dict = result.features.to_dict()
            feat_dict["label"] = label
            feat_dict["video_path"] = str(video_path.relative_to(root)) if video_path.is_relative_to(root) else str(video_path)
            feat_dict["source"] = source
            features_list.append(feat_dict)
            stats["processed"] += 1
    else:
        print(f"Extracting with {n_workers} parallel workers ...")
        with mp.Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(args.method, args.target_fps, args.blur_threshold, args.min_usable_frames),
        ) as pool:
            for entry in pool.imap_unordered(_process_one, items, chunksize=1):
                error = entry.pop("error", None)
                no_features = entry.pop("no_features", None)
                if error:
                    stats["failed"] += 1
                    print(f"  -> Error processing {Path(entry['video_path']).name}: {error}")
                elif no_features:
                    stats["no_features"] += 1
                    print(f"  -> Failed to extract features: {Path(entry['video_path']).name} (usable={entry.get('usable_frames')})")
                else:
                    features_list.append(entry)
                    stats["processed"] += 1
                if stats["processed"] > 0 and stats["processed"] % 100 == 0:
                    rate = stats["processed"] / (time.time() - t_start)
                    remain = int((stats["processed"] + stats["failed"] + stats["no_features"]) / rate) if rate > 0 else 0
                    print(f"    ... {stats['processed']} ok / {stats['failed']} err / {stats['no_features']} no-feat | "
                          f"{rate:.2f} vid/s | ETA ~{remain/60:.0f} min")
    # NOTE: only the initializer pipeline processes all legacy CSV samples when
    # run sequentially; the parallel branch uses the same items list above.

    if not features_list:
        print("No features extracted from any videos.")
        return

    out_df = pd.DataFrame(features_list)
    out_df.to_csv(out_csv_path, index=False)
    print(f"\nSuccessfully extracted features for {len(out_df)} videos.")
    print(f"Features saved to {out_csv_path}")


if __name__ == "__main__":
    main()
