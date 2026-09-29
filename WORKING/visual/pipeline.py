"""
visual/pipeline.py
==================
Visual feature extraction pipeline for batch processing.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd

# Add project roots to path
WORKING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKING_ROOT))

from visual.extractor import compute_visual_features, compute_visual_features_from_video
from visual.features import VisualFeatures, VISUAL_FEATURE_NAMES


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _iter_video_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])


def extract_visual_features_dataset(
    frames_root: Path,
    output_csv: Path,
    metadata_csv: Optional[Path] = None,
    max_frames_per_video: Optional[int] = None,
    backbone: str = "resnet50",
    device: str = "auto",
    deep_feature_dim: int = 16,
) -> None:
    """
    Extract visual features for all videos that have stage-1 frame output.

    Args:
        frames_root: Root directory containing frame_sequences/<video>/frames/
        output_csv: Output CSV path for visual features
        metadata_csv: Optional existing metadata CSV to join with (for labels)
        max_frames_per_video: Max frames to process per video
        backbone: CNN backbone to use
        device: Torch device
        deep_feature_dim: PCA-reduced dimension for deep features
    """
    video_dirs = [d for d in frames_root.iterdir() if d.is_dir()]
    if not video_dirs:
        print(f"No video directories found in {frames_root}")
        return

    # Load metadata for labels if provided
    metadata_labels = {}
    if metadata_csv and metadata_csv.exists():
        meta_df = pd.read_csv(metadata_csv)
        for _, row in meta_df.iterrows():
            metadata_labels[row["video_path"]] = row.get("label", -1)

    features_list = []
    stats = {"processed": 0, "failed": 0, "no_frames": 0}

    for video_dir in video_dirs:
        video_name = video_dir.name
        frames_dir = video_dir / "frames"
        if not frames_dir.exists():
            stats["no_frames"] += 1
            continue

        try:
            visual_features = compute_visual_features(
                frames_dir=str(frames_dir),
                max_frames=max_frames_per_video,
                backbone=backbone,
                device=device,
                deep_feature_dim=deep_feature_dim,
            )

            feat_dict = visual_features.to_dict()
            feat_dict["video_id"] = video_name
            feat_dict["video_path"] = video_name

            # Add label from metadata if available
            label = metadata_labels.get(video_name, -1)
            feat_dict["label"] = label

            features_list.append(feat_dict)
            stats["processed"] += 1

        except Exception as e:
            print(f"  Failed to process {video_name}: {e}")
            stats["failed"] += 1

    if not features_list:
        print("No visual features extracted.")
        return

    out_df = pd.DataFrame(features_list)
    out_df.to_csv(output_csv, index=False)
    print(f"Visual features saved to {output_csv} ({stats['processed']} videos)")
    print(f"Failed: {stats['failed']}, No frames: {stats['no_frames']}")


def extract_visual_features_from_videos(
    video_root: Path,
    output_csv: Path,
    labels_csv: Optional[Path] = None,
    max_frames_per_video: Optional[int] = 100,
    sample_fps: float = 10.0,
    backbone: str = "resnet50",
    device: str = "auto",
    deep_feature_dim: int = 16,
) -> None:
    """
    Extract visual features directly from video files (when stage-1 frames unavailable).
    """
    video_files = _iter_video_files(video_root)
    if not video_files:
        print(f"No videos found in {video_root}")
        return

    # Load labels if provided; restrict the scan to videos that have rPPG
    # rows (fuse inner-joins on them anyway) — otherwise a capped extraction
    # (e.g. setup_and_run.ps1 -Quick) still grinds the whole video root.
    labels = {}
    if labels_csv and labels_csv.exists():
        labels_df = pd.read_csv(labels_csv)
        for _, row in labels_df.iterrows():
            labels[str(row["video_path"])] = row.get("label", -1)
        wanted = {Path(p).stem.lower() for p in labels}
        video_files = [p for p in video_files if p.stem.lower() in wanted]
    if not video_files:
        print("No videos matched the rPPG CSV under the given video root — check paths.")
        return

    features_list = []
    stats = {"processed": 0, "failed": 0}

    for video_path in video_files:
        try:
            visual_features = compute_visual_features_from_video(
                video_path=str(video_path),
                max_frames=max_frames_per_video,
                sample_fps=sample_fps,
                backbone=backbone,
                device=device,
                deep_feature_dim=deep_feature_dim,
            )

            feat_dict = visual_features.to_dict()
            feat_dict["video_id"] = video_path.stem
            feat_dict["video_path"] = str(video_path)

            label = labels.get(str(video_path), -1)
            feat_dict["label"] = label

            features_list.append(feat_dict)
            stats["processed"] += 1

        except Exception as e:
            print(f"  Failed to process {video_path.name}: {e}")
            stats["failed"] += 1

    if not features_list:
        print("No visual features extracted.")
        return

    out_df = pd.DataFrame(features_list)
    out_df.to_csv(output_csv, index=False)
    print(f"Visual features saved to {output_csv} ({stats['processed']} videos)")
    print(f"Failed: {stats['failed']}")


def fuse_features(
    rppg_csv: Path,
    visual_csv: Path,
    output_csv: Path,
    join_key: str = "video_path",
) -> None:
    """
    Fuse rPPG and visual feature tables on a common key.

    Args:
        rppg_csv: Path to rPPG features CSV (with label column)
        visual_csv: Path to visual features CSV
        output_csv: Output path for fused features
        join_key: Column name to join on (default: "video_path")
                  Will create normalized "video_stem" key if paths don't match.
    """
    rppg_df = pd.read_csv(rppg_csv)
    visual_df = pd.read_csv(visual_csv)

    # Create normalized join key (video stem without extension) for both tables
    def normalize_key(path):
        """Extract video stem from path."""
        return Path(str(path)).stem

    # Add video_stem column to both dataframes
    rppg_df["video_stem"] = rppg_df["video_path"].apply(normalize_key)
    visual_df["video_stem"] = visual_df["video_path"].apply(normalize_key)

    # Use video_stem as join key
    actual_join_key = "video_stem"

    # Merge on video_stem
    fused = pd.merge(rppg_df, visual_df, on=actual_join_key, how="inner", suffixes=("_rppg", "_visual"))

    # Handle duplicate label columns
    if "label_rppg" in fused.columns and "label_visual" in fused.columns:
        # They should be the same; use rPPG label
        fused["label"] = fused["label_rppg"]
        fused = fused.drop(columns=["label_rppg", "label_visual"])
    elif "label_rppg" in fused.columns:
        fused["label"] = fused["label_rppg"]
        fused = fused.drop(columns=["label_rppg"])
    elif "label_visual" in fused.columns:
        fused["label"] = fused["label_visual"]
        fused = fused.drop(columns=["label_visual"])

    # Remove duplicate video_id columns
    if "video_id_rppg" in fused.columns and "video_id_visual" in fused.columns:
        fused = fused.drop(columns=["video_id_visual"])

    # Keep original video_path from rPPG (more informative)
    if "video_path_rppg" in fused.columns:
        fused = fused.rename(columns={"video_path_rppg": "video_path"})
        fused = fused.drop(columns=["video_path_visual"], errors="ignore")

    fused.to_csv(output_csv, index=False)
    print(f"Fused features saved to {output_csv} ({len(fused)} videos)")
    print(f"  rPPG features: {len([c for c in rppg_df.columns if c not in ['label', 'video_path', 'video_id', 'source', 'video_stem']])}")
    print(f"  Visual features: {len([c for c in visual_df.columns if c not in ['label', 'video_path', 'video_id', 'source', 'video_stem']])}")
    print(f"  Fused total: {len([c for c in fused.columns if c not in ['label', 'video_path', 'video_id', 'source', 'video_stem']])}")


def create_experiment_splits(
    fused_csv: Path,
    output_dir: Path,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Create train/val/test splits for fused features only.

    Args:
        fused_csv: Path to fused features CSV
        output_dir: Output directory for split files
        test_ratio: Test split ratio
        val_ratio: Validation split ratio
        seed: Random seed
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(fused_csv)
    if "label" not in df.columns:
        raise ValueError("Fused CSV must have 'label' column")

    # Create subject groups for grouped splitting
    # Use video_id as group if available, otherwise video_path
    if "video_id" in df.columns:
        groups = df["video_id"].astype(str)
    else:
        groups = df["video_path"].astype(str)

    y = df["label"].values

    # Simple random split with stratification (for now)
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_ratio, stratify=y, random_state=seed
    )

    # Second split: train vs val
    y_train_val = y[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio / (1 - test_ratio), stratify=y_train_val, random_state=seed
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Only fused feature set
    feature_cols = [c for c in df.columns if c not in ["label", "video_path", "video_id", "source", "video_stem"]]
    
    if not feature_cols:
        print("  No features available for fused set")
        return

    X = df[feature_cols].values
    # Save splits
    np.savez_compressed(
        output_dir / "fused_split.npz",
        X_train=X[train_idx], y_train=y[train_idx],
        X_val=X[val_idx], y_val=y[val_idx],
        X_test=X[test_idx], y_test=y[test_idx],
        feature_names=np.array(feature_cols),
    )

    # Save split indices for reproducibility
    np.savez_compressed(
        output_dir / "split_indices.npz",
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )
    print(f"Experiment splits saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visual feature extraction and fusion pipeline")
    parser.add_argument("--frames-root", type=str, default=None, help="Root of stage-1 frame sequences")
    parser.add_argument("--video-root", type=str, default=None, help="Root of raw videos (fallback)")
    parser.add_argument("--rppg-csv", type=str, default=None, help="Path to rPPG features CSV")
    parser.add_argument("--output-dir", type=str, default=str(
        Path(os.environ.get("MAJ_OUTPUT_ROOT", WORKING_ROOT.parent / "Scrape" / "output")) / "visual"
    ), help="Output directory")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames per video")
    parser.add_argument("--backbone", type=str, default="resnet50", help="CNN backbone")
    parser.add_argument("--device", type=str, default="auto", help="Torch device")
    parser.add_argument("--deep-dim", type=int, default=16, help="Deep feature PCA dimension")
    parser.add_argument("--fuse", action="store_true", help="Fuse with rPPG features")
    parser.add_argument("--create-splits", action="store_true", help="Create experiment splits")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.frames_root:
        visual_csv = output_dir / "visual_features.csv"
        extract_visual_features_dataset(
            frames_root=Path(args.frames_root),
            output_csv=visual_csv,
            metadata_csv=Path(args.rppg_csv) if args.rppg_csv else None,
            max_frames_per_video=args.max_frames,
            backbone=args.backbone,
            device=args.device,
            deep_feature_dim=args.deep_dim,
        )
    elif args.video_root:
        visual_csv = output_dir / "visual_features.csv"
        extract_visual_features_from_videos(
            video_root=Path(args.video_root),
            output_csv=visual_csv,
            labels_csv=Path(args.rppg_csv) if args.rppg_csv else None,
            max_frames_per_video=args.max_frames,
            sample_fps=10.0,
            backbone=args.backbone,
            device=args.device,
            deep_feature_dim=args.deep_dim,
        )
    else:
        print("Must provide --frames-root or --video-root")
        sys.exit(1)

    if args.fuse and args.rppg_csv:
        fused_csv = output_dir / "fused_features.csv"
        fuse_features(
            rppg_csv=Path(args.rppg_csv),
            visual_csv=visual_csv,
            output_csv=fused_csv,
        )

        if args.create_splits:
            create_experiment_splits(
                fused_csv=fused_csv,
                output_dir=output_dir / "experiments",
            )