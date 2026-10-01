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

from visual.extractor import (
    VisualFeatureExtractor,
    load_face_crops_from_frames,
)
from visual.features import VisualFeatures, VISUAL_FEATURE_NAMES


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _iter_video_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])


def _training_video_stems(labels_csv: Path) -> set[str]:
    """Return the deterministic training partition used by quantum training."""
    from quantum.config import DataConfig, RPPG_FEATURE_NAMES
    from quantum.data import _grouped_train_val_test_split, _load_feature_rows

    cfg = DataConfig(csv_file=labels_csv)
    X, y, groups, paths, _, _ = _load_feature_rows(labels_csv, RPPG_FEATURE_NAMES, cfg)
    split = _grouped_train_val_test_split(X, y, groups, paths, cfg)
    return {Path(str(path)).stem.lower() for path in split["paths_train"]}


def _fit_shared_pca(
    raw_features: dict[str, np.ndarray],
    training_stems: set[str],
    extractor: VisualFeatureExtractor,
    artifact_path: Path,
) -> None:
    training = [
        features
        for video_id, features in raw_features.items()
        if Path(video_id).stem.lower() in training_stems
    ]
    if not training:
        raise ValueError("No extracted visual videos matched the training partition")
    pooled = np.concatenate(training, axis=0)
    extractor.fit_pca(pooled, str(artifact_path))
    print(f"Training PCA fitted once on {pooled.shape[0]} frame features: {artifact_path}")


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

    if not metadata_csv or not metadata_csv.exists():
        raise ValueError("A labelled rPPG CSV is required to fit training-only visual PCA")
    training_stems = _training_video_stems(metadata_csv)
    extractor = VisualFeatureExtractor(
        backbone=backbone, device=device, deep_feature_dim=deep_feature_dim
    )
    video_crops: dict[str, list[np.ndarray]] = {}
    raw_features: dict[str, np.ndarray] = {}
    stats = {"processed": 0, "failed": 0, "no_frames": 0}

    for video_dir in video_dirs:
        video_name = video_dir.name
        frames_dir = video_dir / "frames"
        if not frames_dir.exists():
            stats["no_frames"] += 1
            continue

        try:
            crops = load_face_crops_from_frames(str(frames_dir), max_frames=max_frames_per_video)
            if not crops:
                raise ValueError("No face crops found")
            video_crops[video_name] = crops
            raw_features[video_name] = extractor.extract_deep_features(crops)
        except Exception as e:
            print(f"  Failed to process {video_name}: {e}")
            stats["failed"] += 1

    pca_path = output_csv.parent / "visual_pca.npz"
    _fit_shared_pca(raw_features, training_stems, extractor, pca_path)
    features_list = []
    for video_name, crops in video_crops.items():
        try:
            visual_features = extractor.extract_from_crops(
                crops, deep_features=raw_features[video_name]
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

    if not labels_csv or not labels_csv.exists():
        raise ValueError("A labelled rPPG CSV is required to fit training-only visual PCA")
    training_stems = _training_video_stems(labels_csv)
    extractor = VisualFeatureExtractor(
        backbone=backbone, device=device, deep_feature_dim=deep_feature_dim
    )
    video_crops: dict[str, list[np.ndarray]] = {}
    raw_features: dict[str, np.ndarray] = {}
    stats = {"processed": 0, "failed": 0}

    for video_path in video_files:
        try:
            # Use the existing video helper once for crops, then reuse raw
            # features after the shared PCA is fitted.
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise IOError(f"Could not open video: {video_path}")
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            stride = max(1, round(native_fps / sample_fps))
            crops = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % stride == 0:
                    h, w = frame.shape[:2]
                    crop = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
                    if crop.size > 0:
                        crops.append(crop)
                    if max_frames_per_video and len(crops) >= max_frames_per_video:
                        break
                frame_idx += 1
            cap.release()
            if not crops:
                raise ValueError("No usable frames found")
            video_crops[video_path.stem] = crops
            raw_features[video_path.stem] = extractor.extract_deep_features(crops)
        except Exception as e:
            print(f"  Failed to process {video_path.name}: {e}")
            stats["failed"] += 1

    pca_path = output_csv.parent / "visual_pca.npz"
    _fit_shared_pca(raw_features, training_stems, extractor, pca_path)
    features_list = []
    for video_path in video_files:
        if video_path.stem not in video_crops:
            continue
        try:
            visual_features = extractor.extract_from_crops(
                video_crops[video_path.stem],
                deep_features=raw_features[video_path.stem],
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
    from quantum.config import DataConfig
    from quantum.data import (
        _assert_no_group_leakage,
        _grouped_train_val_test_split,
        _infer_subject_key,
    )

    df = pd.read_csv(fused_csv)
    if "label" not in df.columns:
        raise ValueError("Fused CSV must have 'label' column")

    if df["video_path"].astype(str).str.strip().duplicated().any():
        raise ValueError("Duplicate video_path values found; refusing to create experiment splits")
    feature_cols = [
        c for c in df.columns
        if c not in ["label", "video_path", "video_id", "source", "video_stem"]
    ]
    if not feature_cols:
        print("  No features available for fused set")
        return
    groups = np.asarray(
        [_infer_subject_key(row.to_dict()) for _, row in df.iterrows()],
        dtype=object,
    )
    y = df["label"].astype(int).values
    paths = df["video_path"].astype(str).values
    split = _grouped_train_val_test_split(
        df[feature_cols].to_numpy(),
        y,
        groups,
        paths,
        DataConfig(seed=seed, val_ratio=val_ratio, test_ratio=test_ratio),
    )
    _assert_no_group_leakage(split)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    np.savez_compressed(
        output_dir / "fused_split.npz",
        X_train=split["X_train"], y_train=split["y_train"],
        X_val=split["X_val"], y_val=split["y_val"],
        X_test=split["X_test"], y_test=split["y_test"],
        feature_names=np.array(feature_cols),
        groups_train=np.asarray(split["groups_train"], dtype=str),
        groups_val=np.asarray(split["groups_val"], dtype=str),
        groups_test=np.asarray(split["groups_test"], dtype=str),
    )

    # Save split indices for reproducibility
    np.savez_compressed(
        output_dir / "split_indices.npz",
        train_idx=np.asarray(
            [df.index[df["video_path"].astype(str) == p][0] for p in split["paths_train"]]
        ),
        val_idx=np.asarray(
            [df.index[df["video_path"].astype(str) == p][0] for p in split["paths_val"]]
        ),
        test_idx=np.asarray(
            [df.index[df["video_path"].astype(str) == p][0] for p in split["paths_test"]]
        ),
    )
    manifest = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "grouping": "metadata-first-grouped-split",
        "rows": {
            str(path): {"split": split_name, "group": str(group)}
            for split_name in ("train", "val", "test")
            for path, group in zip(
                split[f"paths_{split_name}"], split[f"groups_{split_name}"]
            )
        },
    }
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Experiment splits saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    import json

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