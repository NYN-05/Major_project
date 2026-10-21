"""
visual/pipeline.py
==================
Streaming, checkpoint-resumable visual feature extraction for batch processing.

Design notes
------------
Dataset extraction processes one video at a time. Face crops for that video
are materialised, the deep + handcrafted aggregates are computed, and a
compact per-video record (mean 2048-dim deep vector, 23-dim handcrafted
aggregate, frame count) is written to ``parts/<stem>.npz`` before the crops
are released. Peak RAM therefore stays bounded to a single video no matter
how large the dataset is. The previous implementation accumulated every
video's crops and per-frame deep features in memory until the very end,
exhausted the system commit limit about halfway through the full DFDC run,
and lost all completed work on failure.

Videos that already have a valid record are skipped (auto-resume), so an
interrupted run only processes the remainder. Training-partition videos
additionally persist their per-frame ResNet50 rows to
``parts/deep_train/<stem>.npy`` so the shared training PCA can still be
fitted once, at the end, on exactly the same pooled matrix as before.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allocator pressure mitigation - must be set before torch is imported
# through visual.extractor. Windows CUDA allocations fragment badly over the
# very long free/alloc cycles of a dataset run; capping the chunk split size
# keeps sizeable requests satisfiable near commit-limit pressure.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# Add project roots to path
WORKING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKING_ROOT))

from visual.extractor import (
    HANDCRAFTED_DIM,
    RESNET_FEATURE_DIM,
    VisualFeatureExtractor,
    iter_video_center_crops,
    load_face_crops_from_frames,
)
from quantum.operations import (
    OPERATIONS_VERSION,
    cache_key,
    dataset_fingerprint,
    stable_hash,
)

import torch


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
REC_VERSION = "visual-parts-v1"


def _norm_stem(path_or_name) -> str:
    """Lower-cased video stem used for membership/join checks."""
    return Path(str(path_or_name)).stem.lower()


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


def _label_lookups(labels_csv: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build label lookups from the rPPG CSV: by full path and by stem.

    The fused table takes its label from the rPPG columns, so visual row
    labels are informational; the stem fallback exists because frame-sequence
    directories are named by stem while the CSV stores full paths.
    """
    # dtype=str keeps zero-padded numeric stems (DFDC Real/00000.mp4) intact.
    labels_df = pd.read_csv(labels_csv, dtype={"video_path": str})
    by_path: Dict[str, int] = {}
    by_stem: Dict[str, int] = {}
    for _, row in labels_df.iterrows():
        raw = str(row["video_path"])
        try:
            label = int(row["label"])
        except (TypeError, ValueError):
            label = -1
        by_path[raw] = label
        by_stem[_norm_stem(raw)] = label
    return by_path, by_stem


def _assert_training_coverage(item_stems: List[str], training_stems: set, source_hint: str) -> None:
    """Fail fast (within seconds) when no available video can feed the PCA fit.

    Without this guard a misconfigured source (for example a frames root that
    only holds leftover single-video extractions) would run the whole
    extraction before dying at the PCA fit.
    """
    matched = sum(1 for stem in item_stems if stem in training_stems)
    if matched == 0:
        print(
            "ERROR: no available visual videos overlap the rPPG training split; "
            "the shared PCA cannot be fitted "
            f"(videos={len(item_stems)}, training_stems={len(training_stems)}, matched=0)."
        )
        print(f"       source mode: {source_hint}")
        print(
            "       Hint: --frames-root must hold frame sequences for the rPPG CSV "
            "videos, or --video-root must cover the labelled DFDC videos."
        )
        raise SystemExit(2)
def _attempt_video(
    extractor: VisualFeatureExtractor,
    crops_loader: Callable[[], List[np.ndarray]],
    empty_message: str,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """One extraction pass for a single video; crops stay local to this call."""
    crops = crops_loader()
    if not crops:
        raise ValueError(empty_message)
    rows = extractor.extract_deep_features(crops)
    if rows.shape[0] == 0:
        raise ValueError(empty_message)
    handcrafted = extractor.aggregate_handcrafted(crops)
    return rows.shape[0], rows.mean(axis=0), handcrafted, rows


def _process_video(
    extractor: VisualFeatureExtractor,
    crops_loader: Callable[[], List[np.ndarray]],
    empty_message: str,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Process one video with a single retry after dropping memory caches.

    Allocation-style failures (CUDA OOM while another process holds GPU
    memory, cv OutOfMemoryError under system commit pressure) are usually
    transient once the allocator caches are released; the whole extraction,
    including crop loading, is retried once. On a second failure the original
    error surfaces so the log keeps the same "Failed to process ..." shape.
    """
    try:
        return _attempt_video(extractor, crops_loader, empty_message)
    except Exception as first_error:
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            return _attempt_video(extractor, crops_loader, empty_message)
        except Exception:
            raise first_error from None


class _RecordStore:
    """Checkpoint store backing the streaming visual stage.

    One record per completed video under ``parts/<stem>.npz``, plus optional
    per-frame training rows under ``parts/deep_train/<stem>.npy``. Writes are
    atomic (tmp file + os.replace), so an interrupted run never leaves a
    half-written record that would look complete on resume.
    """

    def __init__(self, output_dir: Path) -> None:
        self.parts_dir = Path(output_dir) / "parts"
        self.deep_train_dir = self.parts_dir / "deep_train"
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.deep_train_dir.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, dict] = {}

    def record_path(self, stem: str) -> Path:
        return self.parts_dir / f"{stem}.npz"

    def deep_train_path(self, stem: str) -> Path:
        return self.deep_train_dir / f"{stem}.npy"

    def load_existing(self, training_stems: set, video_stems: set) -> int:
        """Adopt valid persisted records for the current video set.

        Stale partials/mismatches are deleted so those videos are reprocessed.
        Returns the number of records adopted.
        """
        for tmp in list(self.parts_dir.glob("*.tmp")) + list(self.deep_train_dir.glob("*.tmp")):
            try:
                tmp.unlink()
            except OSError:
                pass
        adopted = 0
        for stem in sorted(video_stems):
            record_path = self.record_path(stem)
            if not record_path.exists():
                continue
            try:
                with np.load(record_path, allow_pickle=False) as artifact:
                    if str(artifact["version"]) != REC_VERSION:
                        raise ValueError("record version mismatch")
                    n_frames = int(artifact["n_frames"])
                    deep_mean = np.asarray(artifact["deep_mean"], dtype=np.float32)
                    handcrafted = np.asarray(artifact["handcrafted"], dtype=np.float64)
                    if (
                        n_frames < 1
                        or deep_mean.shape != (RESNET_FEATURE_DIM,)
                        or handcrafted.shape != (HANDCRAFTED_DIM,)
                    ):
                        raise ValueError("record aggregates have unexpected shapes")
                if stem in training_stems:
                    rows = np.load(self.deep_train_path(stem), allow_pickle=False, mmap_mode="r")
                    expected = (n_frames, RESNET_FEATURE_DIM)
                    if rows.shape != expected:
                        raise ValueError(f"deep rows shape {rows.shape} != {expected}")
                    del rows
                self.records[stem] = {
                    "n_frames": n_frames,
                    "deep_mean": deep_mean,
                    "handcrafted": handcrafted,
                }
                adopted += 1
            except (OSError, ValueError, KeyError, EOFError) as exc:
                print(f"  Discarding stale visual record for {stem}: {exc}")
                try:
                    record_path.unlink()
                except OSError:
                    pass
        return adopted

    def save(
        self,
        *,
        stem: str,
        display_path: str,
        label: int,
        n_frames: int,
        deep_mean: np.ndarray,
        handcrafted: np.ndarray,
        deep_rows: Optional[np.ndarray] = None,
    ) -> None:
        """Atomically persist one completed video (and its training rows)."""
        record_path = self.record_path(stem)
        deep_path = self.deep_train_path(stem)
        if deep_rows is not None:
            tmp = deep_path.with_name(deep_path.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.save(fh, np.asarray(deep_rows, dtype=np.float32))
            os.replace(tmp, deep_path)
        tmp = record_path.with_name(record_path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                version=REC_VERSION,
                video=stem,
                video_path=str(display_path),
                label=int(label),
                n_frames=int(n_frames),
                deep_mean=np.asarray(deep_mean, dtype=np.float32),
                handcrafted=np.asarray(handcrafted, dtype=np.float64),
            )
        os.replace(tmp, record_path)
        self.records[stem] = {
            "n_frames": int(n_frames),
            "deep_mean": np.asarray(deep_mean, dtype=np.float32),
            "handcrafted": np.asarray(handcrafted, dtype=np.float64),
        }
def _fit_training_pca(
    store: _RecordStore,
    video_items: List,
    training_stems: set,
    extractor: VisualFeatureExtractor,
    pca_path: Path,
) -> None:
    """Fit the shared training PCA once from persisted per-frame rows.

    Rows are restricted to the training partition, read back in video_items
    order (deterministic), and assembled into one preallocated matrix, so the
    peak RAM is a single (n_rows, 2048) float32 array instead of the whole
    crop tree. Numerically this pools exactly the same per-frame deep
    features as the pre-refactor implementation.
    """
    slots: List[Tuple[str, int]] = []
    total_rows = 0
    for item in video_items:
        stem = _norm_stem(item)
        if stem not in training_stems:
            continue
        record = store.records.get(stem)
        if record is None:
            continue
        rows_path = store.deep_train_path(stem)
        if not rows_path.exists():
            continue
        rows = np.load(rows_path, allow_pickle=False, mmap_mode="r")
        if rows.shape != (record["n_frames"], RESNET_FEATURE_DIM):
            raise ValueError(
                f"Persisted deep rows for {stem} have shape {rows.shape}, "
                f"expected {(record['n_frames'], RESNET_FEATURE_DIM)}"
            )
        slots.append((stem, int(rows.shape[0])))
        total_rows += int(rows.shape[0])
        del rows

    if not slots:
        raise ValueError("No extracted visual videos matched the training partition")

    pooled = np.empty((total_rows, RESNET_FEATURE_DIM), dtype=np.float32)
    offset = 0
    for stem, n in slots:
        rows = np.load(store.deep_train_path(stem), allow_pickle=False, mmap_mode="r")
        pooled[offset:offset + n] = rows
        offset += n
        del rows
    del slots
    gc.collect()
    extractor.fit_pca(pooled, str(pca_path))
    print(f"Training PCA fitted once on {pooled.shape[0]} frame features: {pca_path}")
    del pooled
    gc.collect()


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
    Extract visual features for all rPPG-labelled videos that have stage-1
    frame output.

    Args:
        frames_root: Root directory containing frame_sequences/<video>/frames/
        output_csv: Output CSV path for visual features
        metadata_csv: Labelled rPPG CSV (labels + training partition)
        max_frames_per_video: Max frames to process per video
        backbone: CNN backbone to use
        device: Torch device
        deep_feature_dim: PCA-reduced dimension for deep features
    """
    if not metadata_csv or not metadata_csv.exists():
        raise ValueError("A labelled rPPG CSV is required to fit training-only visual PCA")
    training_stems = _training_video_stems(metadata_csv)
    labels_by_path, labels_by_stem = _label_lookups(metadata_csv)

    frames_root = Path(frames_root)
    all_dirs = [d for d in frames_root.iterdir() if d.is_dir()]
    # Only labelled rPPG videos are processed; stray stage-1 leftovers (for
    # example single-video inference extractions) are skipped.
    video_dirs = [d for d in all_dirs if _norm_stem(d.name) in labels_by_stem]
    filtered = len(all_dirs) - len(video_dirs)
    if filtered:
        print(f"Skipping {filtered} frame directories without rPPG labels in {frames_root}")
    if not video_dirs:
        print(f"No video directories matched the rPPG CSV in {frames_root}")
        return
    item_stems = [_norm_stem(d.name) for d in video_dirs]
    _assert_training_coverage(item_stems, training_stems, "frames-root (stage-1 frame sequences)")

    extractor = VisualFeatureExtractor(
        backbone=backbone, device=device, deep_feature_dim=deep_feature_dim
    )
    store = _RecordStore(output_csv.parent)
    adopted = store.load_existing(training_stems, set(item_stems))
    if adopted:
        print(f"Resuming: {adopted} of {len(video_dirs)} videos already completed in {store.parts_dir}")

    pending = [d for d in video_dirs if _norm_stem(d.name) not in store.records]
    stats = {"processed": 0, "failed": 0, "no_frames": 0}
    total_videos = len(video_dirs)
    received = 0
    t_start = time.time()
    # Machine-readable total for live progress wrappers (setup_progress.py).
    print(f"TOTAL samples={total_videos} pending={len(pending)}", flush=True)

    for video_dir in pending:
        video_name = video_dir.name
        stem = _norm_stem(video_name)
        frames_dir = video_dir / "frames"
        is_training = stem in training_stems
        try:
            if not frames_dir.exists():
                stats["no_frames"] += 1
            else:
                def crops_loader(frames_dir=frames_dir):
                    return load_face_crops_from_frames(str(frames_dir), max_frames=max_frames_per_video)
                n_frames, deep_mean, handcrafted, deep_rows = _process_video(
                    extractor, crops_loader, "No face crops found"
                )
                store.save(
                    stem=stem,
                    display_path=video_name,
                    label=labels_by_stem.get(stem, -1),
                    n_frames=n_frames,
                    deep_mean=deep_mean,
                    handcrafted=handcrafted,
                    deep_rows=deep_rows if is_training else None,
                )
                del deep_rows, deep_mean, handcrafted
        except Exception as e:
            print(f"  Failed to process {video_name}: {e}")
            stats["failed"] += 1
        received += 1
        ok = received - stats["failed"] - stats["no_frames"]
        print(
            f"PROGRESS received={received} total={len(pending)} ok={ok} "
            f"err={stats['failed']} nofeat={stats['no_frames']} gated=0 "
            f"elapsed={time.time() - t_start:.1f}",
            flush=True,
        )

    pca_path = output_csv.parent / "visual_pca.npz"
    _fit_training_pca(store, video_dirs, training_stems, extractor, pca_path)

    features_list = []
    for video_dir in video_dirs:
        stem = _norm_stem(video_dir.name)
        record = store.records.get(stem)
        if record is None:
            continue
        visual_features = extractor.build_features_from_aggregates(
            record["deep_mean"], record["handcrafted"]
        )
        feat_dict = visual_features.to_dict()
        feat_dict["video_id"] = video_dir.name
        feat_dict["video_path"] = video_dir.name

        # Add label from metadata if available
        label = labels_by_stem.get(stem, labels_by_path.get(video_dir.name, -1))
        feat_dict["label"] = label

        features_list.append(feat_dict)
        stats["processed"] += 1

    if not features_list:
        print("No visual features extracted.")
        return

    out_df = pd.DataFrame(features_list)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    cache_metadata = {
        "cache_key": cache_key(
            dataset_hash=dataset_fingerprint(frames_root),
            preprocessing_version="visual-pca-resnet50-v2",
            model_version=backbone,
            feature_version="visual-39-v1",
            config_hash=stable_hash({
                "deep_feature_dim": deep_feature_dim,
                "max_frames_per_video": max_frames_per_video,
                "device": str(extractor.device),
            }),
        ),
        "operations_version": OPERATIONS_VERSION,
        "source": str(frames_root),
        "output": str(output_csv),
    }
    output_csv.with_suffix(".metadata.json").write_text(
        json.dumps(cache_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
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
    Extract visual features directly from video files (when stage-1 frames
    unavailable). Streaming + checkpoint-resumable; see module docstring.
    """
    video_files = _iter_video_files(Path(video_root))
    if not video_files:
        print(f"No videos found in {video_root}")
        return

    if not labels_csv or not labels_csv.exists():
        raise ValueError("A labelled rPPG CSV is required to fit training-only visual PCA")
    training_stems = _training_video_stems(labels_csv)
    labels_by_path, labels_by_stem = _label_lookups(labels_csv)

    # Restrict the scan to videos that have rPPG rows (fuse inner-joins on
    # them anyway) - otherwise a capped extraction (e.g. setup_and_run.ps1
    # -Quick) still grinds the whole video root.
    wanted = set(labels_by_stem)
    video_files = [p for p in video_files if _norm_stem(p) in wanted]
    if not video_files:
        print("No videos matched the rPPG CSV under the given video root — check paths.")
        return
    item_stems = [_norm_stem(p) for p in video_files]
    _assert_training_coverage(item_stems, training_stems, "video-root (direct video extraction)")

    extractor = VisualFeatureExtractor(
        backbone=backbone, device=device, deep_feature_dim=deep_feature_dim
    )
    store = _RecordStore(output_csv.parent)
    adopted = store.load_existing(training_stems, set(item_stems))
    if adopted:
        print(f"Resuming: {adopted} of {len(video_files)} videos already completed in {store.parts_dir}")

    pending = [p for p in video_files if _norm_stem(p) not in store.records]
    stats = {"processed": 0, "failed": 0}
    total_videos = len(video_files)
    received = 0
    t_start = time.time()
    # Machine-readable total for live progress wrappers (setup_progress.py).
    print(f"TOTAL samples={total_videos} pending={len(pending)}", flush=True)

    for video_path in pending:
        stem = _norm_stem(video_path)
        try:
            def crops_loader(video_path=video_path):
                return iter_video_center_crops(
                    video_path, sample_fps=sample_fps, max_frames=max_frames_per_video
                )
            n_frames, deep_mean, handcrafted, deep_rows = _process_video(
                extractor, crops_loader, "No usable frames found"
            )
            store.save(
                stem=stem,
                display_path=str(video_path),
                label=labels_by_path.get(str(video_path), labels_by_stem.get(stem, -1)),
                n_frames=n_frames,
                deep_mean=deep_mean,
                handcrafted=handcrafted,
                deep_rows=deep_rows if stem in training_stems else None,
            )
            del deep_rows, deep_mean, handcrafted
        except Exception as e:
            print(f"  Failed to process {video_path.name}: {e}")
            stats["failed"] += 1
        received += 1
        ok = received - stats["failed"]
        print(
            f"PROGRESS received={received} total={len(pending)} ok={ok} "
            f"err={stats['failed']} nofeat=0 gated=0 "
            f"elapsed={time.time() - t_start:.1f}",
            flush=True,
        )

    pca_path = output_csv.parent / "visual_pca.npz"
    _fit_training_pca(store, video_files, training_stems, extractor, pca_path)

    features_list = []
    for video_path in video_files:
        stem = _norm_stem(video_path)
        record = store.records.get(stem)
        if record is None:
            continue
        visual_features = extractor.build_features_from_aggregates(
            record["deep_mean"], record["handcrafted"]
        )
        feat_dict = visual_features.to_dict()
        feat_dict["video_id"] = video_path.stem
        feat_dict["video_path"] = str(video_path)

        label = labels_by_path.get(str(video_path), labels_by_stem.get(stem, -1))
        feat_dict["label"] = label

        features_list.append(feat_dict)
        stats["processed"] += 1

    if not features_list:
        print("No visual features extracted.")
        return

    out_df = pd.DataFrame(features_list)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
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
    # dtype=str keeps zero-padded numeric stems (DFDC Real/00000.mp4)
    # intact; default int coercion would turn stem "00001" into 1 and
    # silently drop those videos from the stem join.
    rppg_df = pd.read_csv(rppg_csv, dtype={"video_path": str, "video_id": str})
    visual_df = pd.read_csv(visual_csv, dtype={"video_path": str, "video_id": str})

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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
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

    df = pd.read_csv(fused_csv, dtype={"video_path": str, "video_id": str})
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
