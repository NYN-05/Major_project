"""
extract_dataset_features.py
===========================
Batch processes the available real and deepfake datasets to extract rPPG
features. The script now supports the DFDC dataset located at the path
specified by the DFDC_DATASET_PATH environment variable (set in .env).

Outputs a CSV file dataset_features.csv for training the classifier.

Data-quality caveat (measured on the DFDC-derived 16-row table): every row
carries negative SNR (pulse buried in noise), HR features sit on a coarse
~24.3 BPM grid set by short-clip spectral binning at 10 fps, and the train
split is 10 rows — treat any metrics trained on this table as indicative
only, not deployment guarantees.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Silence MediaPipe / TensorFlow Lite C++ logging (GLOG + TF) that would
# otherwise flood stderr from every worker process. Set before the
# RPPGPipeline import and inside each worker so spawned children are quiet.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import pandas as pd  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from quantum.config import get_dfdc_dataset_path  # noqa: E402
from RPPG import RPPGPipeline  # noqa: E402
from RPPG.face_roi import FaceROIExtractor  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Upper bound (seconds) for one clip before the parent gives up on the worker.
ITEM_TIMEOUT_S = 600

_WORKER: dict = {}

# Global pause flag for graceful pause handling
_PAUSE_REQUESTED = False


@dataclass
class CheckpointState:
    """Persistent checkpoint state for dataset generation."""
    last_completed_index: int = -1
    total_samples: int = 0
    processed_count: int = 0
    failed_count: int = 0
    no_features_count: int = 0
    gated_count: int = 0
    config_hash: str = ""
    sample_hashes: List[str] = None  # Hash of each sample path for verification
    completed_samples: List[str] = None  # List of video paths already processed
    completed_indices: set = None  # Set of sample indices already processed
    paused: bool = False
    resume_from_index: int = -1  # Start index from --start parameter; -1 means use last_completed_index + 1

    def __post_init__(self):
        if self.sample_hashes is None:
            self.sample_hashes = []
        if self.completed_samples is None:
            self.completed_samples = []
        if self.completed_indices is None:
            self.completed_indices = set()


class CheckpointManager:
    """Manages checkpoint state with atomic writes and resume capability."""

    def __init__(self, checkpoint_path: Path, samples: List[Tuple[int, Path, str, Path]], config: dict):
        self.checkpoint_path = checkpoint_path
        self.samples = samples
        self.config = config
        self.state = CheckpointState()
        self._config_hash = self._compute_config_hash(config)
        self._sample_hashes = [self._hash_sample(s) for s in samples]
        self._sample_paths = [str(s[1]) for s in samples]  # for quick lookup
        self._index_to_path = {i: str(s[1]) for i, s in enumerate(samples)}
        self._lock = mp.Lock()

    def _compute_config_hash(self, config: dict) -> str:
        """Compute a hash of the configuration for validation."""
        import hashlib
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _hash_sample(self, sample: Tuple[int, Path, str, Path]) -> str:
        """Compute a hash of a sample for integrity checking."""
        import hashlib
        label, path, source, root = sample
        sample_str = f"{label}|{path}|{source}"
        return hashlib.sha256(sample_str.encode()).hexdigest()[:16]

    def _get_index_for_path(self, path: str) -> Optional[int]:
        """Get the index for a sample path, or None if not found."""
        for i, s in enumerate(self.samples):
            if str(s[1]) == path:
                return i
        return None

    def load(self) -> bool:
        """Load checkpoint from disk. Returns True if valid and compatible."""
        if not self.checkpoint_path.exists():
            return False
        try:
            with open(self.checkpoint_path, "r") as f:
                data = json.load(f)
            # Verify config compatibility
            if data.get("config_hash") != self._config_hash:
                print(f"[checkpoint] Config mismatch, starting fresh (hash: {data.get('config_hash')} vs {self._config_hash})")
                return False
            # Verify sample list integrity
            saved_hashes = data.get("sample_hashes", [])
            if saved_hashes != self._sample_hashes:
                print(f"[checkpoint] Sample list changed, starting fresh")
                return False
            # Restore state
            self.state = CheckpointState(
                last_completed_index=data.get("last_completed_index", -1),
                total_samples=data.get("total_samples", len(self.samples)),
                processed_count=data.get("processed_count", 0),
                failed_count=data.get("failed_count", 0),
                no_features_count=data.get("no_features_count", 0),
                gated_count=data.get("gated_count", 0),
                config_hash=data.get("config_hash", ""),
                sample_hashes=data.get("sample_hashes", []),
                completed_samples=data.get("completed_samples", []),
                completed_indices=set(data.get("completed_indices", [])),
                paused=data.get("paused", False),
                resume_from_index=data.get("resume_from_index", 0),
            )
            # Validate state consistency
            if self.state.last_completed_index >= len(self.samples) and self.state.last_completed_index != -1:
                print(f"[checkpoint] Invalid state (index out of bounds), starting fresh")
                return False
            print(f"[checkpoint] Resumed from index {self.state.last_completed_index} "
                  f"({self.state.processed_count} processed, {self.state.failed_count} failed, "
                  f"{self.state.no_features_count} no-features, {self.state.gated_count} gated)")
            return True
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"[checkpoint] Failed to load checkpoint ({e}), starting fresh")
            return False

    def save(self, index: int, result_type: str, sample_path: str = None) -> None:
        """Atomically save checkpoint after processing a sample."""
        with self._lock:
            if result_type == "processed":
                self.state.processed_count += 1
            elif result_type == "failed":
                self.state.failed_count += 1
            elif result_type == "no_features":
                self.state.no_features_count += 1
            elif result_type == "gated":
                self.state.gated_count += 1

            if sample_path and index not in self.state.completed_indices:
                self.state.completed_samples.append(sample_path)
            self.state.last_completed_index = index
            self.state.completed_indices.add(index)
            self._write_locked()

    def _write_locked(self) -> None:
        """Serialize state to the checkpoint file atomically (lock held)."""
        data = {
            "last_completed_index": self.state.last_completed_index,
            "total_samples": self.state.total_samples,
            "processed_count": self.state.processed_count,
            "failed_count": self.state.failed_count,
            "no_features_count": self.state.no_features_count,
            "gated_count": self.state.gated_count,
            "config_hash": self._config_hash,
            "sample_hashes": self._sample_hashes,
            "completed_samples": self.state.completed_samples,
            "completed_indices": sorted(self.state.completed_indices),
            "paused": self.state.paused,
            "resume_from_index": self.state.resume_from_index,
        }
        # Atomic write: write to temp then rename
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(data, f)
            os.replace(temp_path, self.checkpoint_path)
        except OSError:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def mark_paused(self) -> None:
        """Mark checkpoint as paused without double-counting progress."""
        with self._lock:
            self.state.paused = True
            self._write_locked()

    def get_start_index(self) -> int:
        """Get the index to resume from."""
        return self.state.resume_from_index if self.state.resume_from_index >= 0 else self.state.last_completed_index + 1

    def is_sample_completed(self, sample_path: str) -> bool:
        """Check if a sample has already been completed."""
        return sample_path in self.state.completed_samples

    def is_index_completed(self, index: int) -> bool:
        """Check if a sample index has already been processed."""
        return index in self.state.completed_indices

    def should_process(self, index: int) -> bool:
        """Check if a sample at the given index should be processed (not already done)."""
        # If resume_from_index is set, only process from that index onwards
        if self.state.resume_from_index >= 0 and index < self.state.resume_from_index:
            return False
        # If this index is already completed, skip it
        if self.is_index_completed(index):
            return False
        return True


def _setup_signal_handlers():
    """Set up signal handlers for graceful pause/stop.

    First Ctrl+C/SIGTERM requests a pause: the current sample finishes,
    its result and checkpoint are saved, then the run stops cleanly.
    A second Ctrl+C forces an immediate exit (checkpoint is already
    consistent — it is written atomically after every sample).
    """
    global _PAUSE_REQUESTED

    def _handle_signal(signum, frame):
        global _PAUSE_REQUESTED
        if _PAUSE_REQUESTED:
            print("\n[pause] Forced exit (second signal).")
            os._exit(130)
        _PAUSE_REQUESTED = True
        print("\n[pause] Pause requested, finishing current sample... "
              "(press Ctrl+C again to force quit)")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def _ignore_sigint() -> None:
    """Workers ignore SIGINT so a console Ctrl+C only reaches the parent,
    which then drains in-flight results and checkpoints before stopping."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass


def _init_worker(
    method: str,
    target_fps: Optional[float],
    blur_threshold: float,
    brightness_min: int,
    brightness_max: int,
    min_usable_frames: int,
    min_sqi: float,
    max_nan_features: int,
    roi_weights: tuple = (0.35, 0.35, 0.30),
    # Phase 2
    use_quality_weighting: bool = False,
    quality_weight_min: float = 0.05,
) -> None:
    """Per-process initializer: creates one RPPGPipeline per worker.

    MediaPipe/TFLite log to C++ stderr from every worker; fd 2 is
    redirected to NUL so the console stays readable (progress lines are
    printed by the parent, so workers need no stderr).
    """
    _ignore_sigint()
    try:
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    except OSError:
        pass
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    _WORKER["pipeline"] = RPPGPipeline(
        method=method,
        target_fps=target_fps,
        blur_threshold=blur_threshold,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        min_usable_frames=min_usable_frames,
        roi_weights=roi_weights,
        # Phase 2
        use_quality_weighting=use_quality_weighting,
        quality_weight_min=quality_weight_min,
    )
    _WORKER["min_sqi"] = min_sqi
    _WORKER["max_nan_features"] = max_nan_features


def _init_worker_gpu(
    method: str,
    target_fps: Optional[float],
    blur_threshold: float,
    brightness_min: int,
    brightness_max: int,
    min_usable_frames: int,
    min_sqi: float,
    max_nan_features: int,
    roi_weights: tuple = (0.35, 0.35, 0.30),
    # Phase 2
    use_quality_weighting: bool = False,
    quality_weight_min: float = 0.05,
) -> None:
    """Per-process initializer for GPU workers: creates RPPGPipeline +
    GPUFaceDetector + FaceROIExtractor (for trace accumulation)."""
    _ignore_sigint()
    try:
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    except OSError:
        pass
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # Ensure torch lib (cuDNN) is on PATH before ORT loads CUDA provider
    try:
        import torch as _torch  # noqa: F401
        _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            os.add_dll_directory(_torch_lib)
            os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass
    _WORKER["pipeline"] = RPPGPipeline(
        method=method,
        target_fps=target_fps,
        blur_threshold=blur_threshold,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        min_usable_frames=min_usable_frames,
        roi_weights=roi_weights,
        # Phase 2
        use_quality_weighting=use_quality_weighting,
        quality_weight_min=quality_weight_min,
    )
    _WORKER["min_sqi"] = min_sqi
    _WORKER["max_nan_features"] = max_nan_features
    from WORKING.RPPG.gpu_face_detector import GPUFaceDetector
    _WORKER["gpu_detector"] = GPUFaceDetector(conf_threshold=0.25)
    _WORKER["roi_extractor"] = FaceROIExtractor()


def _gate_result(result, min_sqi: float, max_nan_features: int) -> Optional[str]:
    """Return a gate reason if this clip's features are too poor to keep.

    Rows with an essentially absent pulse (low SQI) or multiple raw NaNs
    (median-filled) are garbage for the classifier; gating them at the
    dataset build is stricter than the downstream implausibility filter.
    Returns None when the clip passes both gates.
    """
    raw_nan = int(getattr(result.features, "_raw_nan_count", 0))
    if raw_nan > max_nan_features:
        return f"nan_features={raw_nan} > {max_nan_features}"
    sqi = float(result.features.signal_quality_index)
    if sqi < min_sqi:
        return f"sqi={sqi:.4f} < {min_sqi}"
    return None


def _process_one(item: Tuple[int, Path, str, Path]) -> dict:
    """Process a single video inside a worker process. Returns a feature
    dict, or a dict with an 'error'/'no_features'/'gated' marker."""
    label, video_path, source, root = item
    entry: dict = {"label": label, "video_path": str(video_path), "source": source}
    try:
        result = _WORKER["pipeline"].process_video(str(video_path))
    except Exception as exc:  # noqa: BLE001 - record and continue
        entry["error"] = f"{type(exc).__name__} (pid {os.getpid()}): {exc}"
        return entry
    if result.features is None:
        entry["no_features"] = True
        entry["usable_frames"] = result.n_frames_usable
        entry["total_frames"] = len(result.quality_log)
        entry["no_face"] = sum(1 for q in result.quality_log if not q.face_found)
        return entry
    gate = _gate_result(result, _WORKER["min_sqi"], _WORKER["max_nan_features"])
    if gate is not None:
        entry["gated"] = True
        entry["gate_reason"] = gate
        return entry
    return _feature_entry(result, label, video_path, root, source)


def _process_one_gpu(item: Tuple[int, Path, str, Path]) -> dict:
    """GPU-accelerated variant of _process_one.

    Uses GPUFaceDetector (YuNet via ONNX Runtime CUDA) for face
    detection instead of MediaPipe, then falls back to the existing
    ROI extraction and signal/feature computation.
    """
    import cv2
    label, video_path, source, root = item
    entry: dict = {"label": label, "video_path": str(video_path), "source": source}
    pipeline = _WORKER["pipeline"]
    gpu_det = _WORKER["gpu_detector"]
    roi_ext = _WORKER["roi_extractor"]

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            entry["error"] = f"Cannot open video: {video_path}"
            return entry
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_stride = max(1, round(fps / pipeline.target_fps)) if pipeline.target_fps else 1

        left_trace: list = []
        right_trace: list = []
        forehead_trace: list = []
        quality_log: list = []
        warnings: list = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_stride != 0:
                frame_idx += 1
                continue

            # GPU face detection (YuNet ONNX Runtime CUDA)
            face = gpu_det.detect(frame, frame_idx)

            # Reuse pipeline's quality assessment
            q = pipeline._assess_frame(frame, frame_idx, face.found)
            quality_log.append(q)

            if q.is_usable:
                rois = roi_ext.extract_rois(frame, face)
                left_trace.append(roi_ext.mean_rgb(frame, rois.left_cheek))
                right_trace.append(roi_ext.mean_rgb(frame, rois.right_cheek))
                forehead_trace.append(roi_ext.mean_rgb(frame, rois.forehead))
            else:
                left_trace.append(None)
                right_trace.append(None)
                forehead_trace.append(None)

            frame_idx += 1

        cap.release()

        result = pipeline._finalize(
            left_trace, right_trace, forehead_trace, fps, warnings, quality_log,
        )
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"{type(exc).__name__} (pid {os.getpid()}): {exc}"
        return entry

    if result.features is None:
        entry["no_features"] = True
        entry["usable_frames"] = result.n_frames_usable
        entry["total_frames"] = len(result.quality_log)
        entry["no_face"] = sum(1 for q in result.quality_log if not q.face_found)
        return entry
    gate = _gate_result(result, _WORKER["min_sqi"], _WORKER["max_nan_features"])
    if gate is not None:
        entry["gated"] = True
        entry["gate_reason"] = gate
        return entry
    return _feature_entry(result, label, video_path, root, source)


def _feature_entry(result, label: int, video_path: Path, root: Path, source: str) -> dict:
    """Build one CSV row from a successful RPPG result."""
    feat = result.features.to_dict()
    feat["label"] = label
    feat["video_path"] = str(video_path.relative_to(root)) if video_path.is_relative_to(root) else str(video_path)
    feat["source"] = source
    return feat


def _fail_summary(result) -> str:
    """Human-readable reason breakdown for clips with features=None.

    Uses the frame quality log: usable frames, frames where no face was
    found, and frames rejected by the blur/brightness gate.
    """
    n_usable = result.n_frames_usable
    n_no_face = sum(1 for q in result.quality_log if not q.face_found)
    n_total = len(result.quality_log)
    n_quality = max(0, n_total - n_usable - n_no_face)
    return f"(usable={n_usable}/{n_total}: no_face={n_no_face}, quality_rejected={n_quality})"


def _write_features_csv(features_list: List[dict], out_csv_path: Path) -> None:
    """Write the accumulated feature rows to CSV, sorted by video_path.

    Called after every successful extraction, so the CSV is created after
    the first video and refreshed after each subsequent one. A full rewrite
    of a few thousand rows costs tens of ms — negligible next to the
    per-clip MediaPipe runtime.
    """
    out_df = pd.DataFrame(features_list)
    out_df = out_df.sort_values("video_path").reset_index(drop=True)
    os.makedirs(out_csv_path.parent, exist_ok=True)
    out_df.to_csv(str(out_csv_path), index=False)


def _write_features_csv_atomic(features_list: List[dict], out_csv_path: Path) -> None:
    """Atomically write the accumulated feature rows to CSV, sorted by video_path.

    Writes to a temp file first, then renames to the target path to prevent
    partial/corrupt CSV files on interruption.
    """
    out_df = pd.DataFrame(features_list)
    out_df = out_df.sort_values("video_path").reset_index(drop=True)
    os.makedirs(out_csv_path.parent, exist_ok=True)
    temp_path = out_csv_path.with_suffix(".csv.tmp")
    try:
        temp_df = out_df.sort_values("video_path").reset_index(drop=True)
        temp_df.to_csv(str(temp_path), index=False)
        os.replace(temp_path, out_csv_path)
    except OSError:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _append_entry_to_csv_atomic(entry: dict, out_csv_path: Path) -> None:
    """Append a single entry to the CSV atomically.

    Reads the existing CSV (if any), appends the new entry, sorts by video_path,
    and writes atomically via temp file + rename.
    """
    out_df = pd.DataFrame([entry])
    os.makedirs(out_csv_path.parent, exist_ok=True)
    temp_path = out_csv_path.with_suffix(".csv.tmp")

    # Read existing data if CSV exists
    if out_csv_path.exists():
        try:
            existing_df = pd.read_csv(str(out_csv_path))
            out_df = pd.concat([existing_df, out_df], ignore_index=True)
        except Exception:
            pass  # Start fresh if read fails

    # Sort by video_path for consistent ordering
    out_df = out_df.sort_values("video_path").reset_index(drop=True)
    try:
        temp_df = out_df.sort_values("video_path").reset_index(drop=True)
        temp_df.to_csv(str(temp_path), index=False)
        os.replace(temp_path, out_csv_path)
    except OSError:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _output_dir() -> Path:
    return Path(os.environ.get("MAJ_OUTPUT_ROOT", _repo_root().parent / "Scrape" / "output")) / "rppg"


def _find_item_index(items: List[Tuple[int, Path, str, Path]], video_path: Path) -> Optional[int]:
    """Find the index of an item in the list matching the given video path.

    imap_unordered returns results as they complete, so we need to identify
    which sample this is by matching the video path. Matches full path first,
    then the root-relative form (_feature_entry stores video_path relative
    to root when possible). No basename-only fallback: a wrong match would
    corrupt the checkpoint; an unmatched result simply stays pending and is
    retried on the next run.
    """
    video_str = str(video_path)
    for i, (label, vp, source, root) in enumerate(items):
        if str(vp) == video_str:
            return i
    for i, (label, vp, source, root) in enumerate(items):
        try:
            if vp.is_relative_to(root) and str(vp.relative_to(root)) == video_str:
                return i
        except (ValueError, OSError):
            continue
    return None


def _iter_video_files(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def _add_sample(samples: List[Tuple[int, Path, str]], label: int, path: Path, source: str) -> None:
    if path.exists() and path.is_file():
        samples.append((label, path, source))


def _cap_per_class(groups: List[list], max_per_class: Optional[int]) -> None:
    """Deterministically cap per-class file lists.

    A plain sorted()[...] slice would systematically pick the
    alphabetically-first N files (e.g. the first FF++ actor pairs);
    instead the sorted lists are seeded-shuffled (fixed seed 0) before
    truncation, so caps are unbiased AND reproducible across runs.
    """
    if max_per_class is None:
        return
    rng = np.random.RandomState(0)
    for files in groups:
        rng.shuffle(files)
        del files[max_per_class:]


def collect_samples(max_per_class: Optional[int] = None) -> List[Tuple[int, Path, str]]:
    samples: List[Tuple[int, Path, str]] = []

    dfdc_root = get_dfdc_dataset_path()
    if dfdc_root.exists():
        fake_dir = dfdc_root / "Fake"
        real_dir = dfdc_root / "Real"

        fake_files = list(_iter_video_files(fake_dir)) if fake_dir.exists() else []
        real_files = list(_iter_video_files(real_dir)) if real_dir.exists() else []

        if fake_files or real_files:
            _cap_per_class([fake_files, real_files], max_per_class)

            for path in fake_files:
                _add_sample(samples, 1, path, "DFDC_Dataset/Fake")
            for path in real_files:
                _add_sample(samples, 0, path, "DFDC_Dataset/Real")

    # Legacy archive (1) dataset support (optional, if CSV exists)
    legacy_root = _repo_root().parent / "archive (1)"
    legacy_csv = legacy_root / "DeepFake Videos Dataset.csv"
    if legacy_csv.exists():
        legacy_df = pd.read_csv(legacy_csv)

        if "deepfake" in legacy_df.columns:
            fake_paths = [legacy_root / str(value) for value in legacy_df["deepfake"].dropna().tolist()]
            if max_per_class is not None:
                _cap_per_class([fake_paths], max_per_class)
            for path in fake_paths:
                _add_sample(samples, 1, path, "archive (1)/DeepFake Videos Dataset.csv")

        if "video" in legacy_df.columns:
            real_paths = [legacy_root / str(value) for value in legacy_df["video"].dropna().tolist()]
            if max_per_class is not None:
                _cap_per_class([real_paths], max_per_class)
            for path in real_paths:
                _add_sample(samples, 0, path, "archive (1)/DeepFake Videos Dataset.csv")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract rPPG features from deepfake and real video datasets (DFDC only).")
    parser.add_argument("--method", default="POS", choices=["POS", "CHROM"], help="rPPG reconstruction method")
    parser.add_argument("--target-fps", type=float, default=None, help="Optional target FPS for sampling")
    parser.add_argument("--blur-threshold", type=float, default=3.0, help="Minimum Laplacian variance to keep a frame (lowered for compressed video)")
    parser.add_argument("--brightness-min", type=int, default=15, help="Minimum mean pixel intensity to keep a frame (0-255)")
    parser.add_argument("--brightness-max", type=int, default=245, help="Maximum mean pixel intensity to keep a frame (0-255)")
    parser.add_argument("--min-usable-frames", type=int, default=24, help="Minimum usable frames required per clip (lowered for short videos)")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap for each label when extracting features")
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes (0 = all CPU cores)")
    parser.add_argument("--min-sqi", type=float, default=0.05, help="Drop clips whose signal_quality_index is below this (0 disables)")
    parser.add_argument("--max-nan-features", type=int, default=2, help="Drop clips with more than this many median-filled (raw-NaN) features")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU-accelerated face detection (YuNet via ONNX Runtime CUDA) instead of MediaPipe (default: on)")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Disable GPU face detection, use CPU (MediaPipe)")
    parser.add_argument("--gpu-workers", type=int, default=None, help="Number of GPU worker processes (default: 8 when --gpu is set)")
    parser.add_argument("--roi-weights", type=float, nargs=3, default=None, metavar=("LEFT", "RIGHT", "FOREHEAD"), help="ROI weights for left cheek, right cheek, forehead (default: 0.35 0.35 0.30)")
    # Phase 2: Quality-weighted rPPG
    parser.add_argument("--use-quality-weighting", action="store_true", help="Use continuous quality weights instead of binary frame rejection (Phase 2)")
    parser.add_argument("--quality-weight-min", type=float, default=0.05, help="Minimum weight for poor-quality frames (default: 0.05)")
    # Checkpoint/Resume
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--start", type=int, default=-1, help="Start processing from this sample index (0-based; default 0 — with --resume, completed indices are skipped automatically)")
    parser.add_argument("--end", type=int, default=None, help="Stop processing at this sample index (exclusive)")
    parser.add_argument("--checkpoint-file", default=None, help="Path to checkpoint file (default: alongside output CSV)")
    args = parser.parse_args()

    samples = collect_samples(max_per_class=args.max_per_class)
    if not samples:
        print("No dataset videos were found. Check DFDC_DATASET_PATH in .env or archive (1).")
        return

    root = _repo_root()
    out_csv_path = Path(args.output) if args.output else _output_dir() / "dataset_features.csv"
    os.makedirs(out_csv_path.parent, exist_ok=True)

    # Configure checkpoint path
    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else out_csv_path.parent / "checkpoint.json"

    # Set up checkpoint manager
    config = {
        "method": args.method,
        "target_fps": args.target_fps,
        "blur_threshold": args.blur_threshold,
        "brightness_min": args.brightness_min,
        "brightness_max": args.brightness_max,
        "min_usable_frames": args.min_usable_frames,
        "max_per_class": args.max_per_class,
        "use_quality_weighting": args.use_quality_weighting,
        "quality_weight_min": args.quality_weight_min,
        "min_sqi": args.min_sqi,
        "max_nan_features": args.max_nan_features,
        "gpu": args.gpu,
    }

    # Items are 4-tuples: (label, video_path, source, root) — the shape
    # CheckpointManager._hash_sample, the worker functions, and
    # _find_item_index all expect. items is NEVER sliced: checkpoint
    # completed_indices are positions in this full list, so range
    # handling must not shift them.
    items = [(label, video_path, source, root) for label, video_path, source in samples]

    ckpt_mgr = CheckpointManager(checkpoint_path, items, config)

    # Determine the [range_start, range_end) window to process.
    # --resume loads the checkpoint; already-completed indices are skipped
    # via completed_indices (not via last_completed_index + 1, which would
    # be wrong when imap_unordered completed higher indices first).
    range_start = args.start if args.start >= 0 else 0
    range_end = args.end if args.end is not None else len(items)
    if args.resume:
        if not ckpt_mgr.load():
            raise SystemExit(
                f"[checkpoint] --resume requested but no valid checkpoint was found at "
                f"{checkpoint_path}. Run without --resume to start fresh."
            )
        if ckpt_mgr.state.completed_indices and not out_csv_path.exists():
            raise SystemExit(
                f"[checkpoint] Checkpoint marks {len(ckpt_mgr.state.completed_indices)} samples "
                f"completed but the CSV {out_csv_path} is missing. Restore the CSV or run "
                f"without --resume to start fresh."
            )
        print(f"[checkpoint] Resuming: processing indices [{range_start}, {range_end}), "
              f"{len(ckpt_mgr.state.completed_indices)} already completed")

    # If resuming, don't delete the existing CSV - instead, we'll rely on the checkpoint
    # to prevent duplicates. Only delete on fresh extraction (CSV and checkpoint
    # together, so a stale checkpoint can never claim rows that don't exist).
    if not args.resume:
        if out_csv_path.exists():
            out_csv_path.unlink()
            print(f"Fresh extraction: removed existing {out_csv_path}")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"Fresh extraction: removed existing {checkpoint_path}")

    # Set up signal handlers for pause
    _setup_signal_handlers()

    use_gpu = args.gpu
    if use_gpu:
        # Validate the actual YuNet session before spawning workers.
        try:
            from RPPG.gpu_face_detector import GPUFaceDetector
            with GPUFaceDetector() as _gpu_detector:
                if not _gpu_detector.gpu_active:
                    print(
                        "[gpu] WARNING: CUDA provider unavailable; "
                        "falling back to CPU (slow). GPU requested but not "
                        "available."
                    )
                    use_gpu = False
                else:
                    print(
                        "[gpu] Preflight passed: YuNet providers="
                        f"{_gpu_detector._session.get_providers()}"
                    )
        except Exception as exc:
            print(
                f"[gpu] WARNING: GPU preflight failed ({type(exc).__name__}: {exc}); "
                "falling back to CPU."
            )
            use_gpu = False

    roi_weights = tuple(args.roi_weights) if args.roi_weights else (0.35, 0.35, 0.30)

    if use_gpu:
        n_workers = args.gpu_workers if args.gpu_workers else min(8, os.cpu_count() or 8)
        print(f"[gpu] Using GPU face detection (YuNet ONNX Runtime CUDA) with {n_workers} workers")
    elif args.workers == 0:
        n_workers = max(1, os.cpu_count() or 1)
    else:
        n_workers = args.workers

    stats = {"processed": 0, "no_features": 0, "failed": 0, "gated": 0}
    t_start = time.time()

    # Single-worker path (workers == 1 and not use_gpu)
    if n_workers <= 1 and not use_gpu:
        pipeline = RPPGPipeline(
            method=args.method,
            target_fps=args.target_fps,
            blur_threshold=args.blur_threshold,
            brightness_min=args.brightness_min,
            brightness_max=args.brightness_max,
            min_usable_frames=args.min_usable_frames,
            roi_weights=roi_weights,
            # Phase 2
            use_quality_weighting=args.use_quality_weighting,
            quality_weight_min=args.quality_weight_min,
        )
        features_list = []
        # Resume: seed from the existing CSV so rewritten rows are not lost
        if args.resume and out_csv_path.exists():
            try:
                features_list = pd.read_csv(str(out_csv_path)).to_dict("records")
                print(f"[checkpoint] Loaded {len(features_list)} existing rows from {out_csv_path}")
            except (OSError, ValueError):
                features_list = []

        for idx, (label, video_path, source, _root) in enumerate(items):
            # Range window: indices outside [--start, --end) are not ours
            if not (range_start <= idx < range_end):
                continue

            # Check if we should pause
            if _PAUSE_REQUESTED:
                print("\n[pause] Stopping between samples.")
                # Mark as paused in checkpoint
                ckpt_mgr.mark_paused()
                print(f"[pause] Processed {ckpt_mgr.state.processed_count} samples total. "
                      f"Resume with --resume to continue.")
                return

            # Check if this sample should be processed (not already done)
            if not ckpt_mgr.should_process(idx):
                continue

            label_name = "Fake" if label == 1 else "Real"
            print(f"Processing {label_name}: {video_path}")
            try:
                result = pipeline.process_video(str(video_path))
            except Exception as exc:
                print(f"  -> Error processing {video_path.name}: {exc}")
                stats["failed"] += 1
                ckpt_mgr.save(idx, "failed", str(video_path))
                continue
            if result.features is None:
                print(f"  -> Failed to extract features: {video_path.name}")
                stats["no_features"] += 1
                ckpt_mgr.save(idx, "no_features", str(video_path))
                continue
            gate = _gate_result(result, args.min_sqi, args.max_nan_features)
            if gate is not None:
                print(f"  -> Gated (low signal quality): {video_path.name} ({gate})")
                stats["gated"] += 1
                ckpt_mgr.save(idx, "gated", str(video_path))
                continue
            entry = _feature_entry(result, label, video_path, root, source)
            features_list.append(entry)
            _write_features_csv_atomic(features_list, out_csv_path)
            stats["processed"] += 1
            ckpt_mgr.save(idx, "processed", str(video_path))

            # Check pause request after each sample
            if _PAUSE_REQUESTED:
                print("\n[pause] Pausing pipeline after current sample. "
                      "Resume with --resume to continue.")
                ckpt_mgr.mark_paused()
                return

        if not features_list:
            print("No features extracted from any videos.")
            return

        print(f"\nSuccessfully extracted features for {len(features_list)} videos "
              f"({stats['gated']} gated by signal quality, {stats['no_features']} with no features, "
              f"{stats['failed']} errors).")
        print(f"Features saved to {out_csv_path}")

    else:
        # Multi-worker path (GPU or multiple CPU workers)
        worker_fn = _process_one_gpu if use_gpu else _process_one
        init_fn = _init_worker_gpu if use_gpu else _init_worker
        print(f"Extracting with {n_workers} parallel workers {'(GPU)' if use_gpu else '(CPU)'} ...")

        # Only send samples inside the range window that the checkpoint
        # has not already completed — this is what prevents reprocessing
        # and duplicate rows after a resume.
        pending = [
            item for idx, item in enumerate(items)
            if range_start <= idx < range_end and not ckpt_mgr.is_index_completed(idx)
        ]
        print(f"[checkpoint] {len(pending)} of {len(items)} samples to process "
              f"in index range [{range_start}, {range_end})")
        if not pending:
            print("Nothing to do: every sample in the requested range is already completed.")
            return

        def _record_result(entry: dict) -> None:
            """Persist one worker result: append to CSV (success only) and
            save the checkpoint atomically. Skips duplicates."""
            error = entry.pop("error", None)
            no_features = entry.pop("no_features", None)
            gated = entry.pop("gated", None)
            video_path = Path(entry.get("video_path", ""))
            idx = _find_item_index(items, video_path)
            if idx is None or not ckpt_mgr.should_process(idx):
                return
            if error:
                stats["failed"] += 1
                ckpt_mgr.save(idx, "failed", str(video_path))
                print(f"  -> Error processing {video_path.name}: {error}")
            elif no_features:
                stats["no_features"] += 1
                ckpt_mgr.save(idx, "no_features", str(video_path))
                print(f"  -> Failed to extract features: {video_path.name} "
                      f"(usable={entry.get('usable_frames', 0)}/{entry.get('total_frames', 0)})")
            elif gated:
                stats["gated"] += 1
                ckpt_mgr.save(idx, "gated", str(video_path))
                print(f"  -> Gated (low signal quality): {video_path.name} "
                      f"({entry.get('gate_reason', 'unknown')})")
            else:
                _append_entry_to_csv_atomic(entry, out_csv_path)
                ckpt_mgr.save(idx, "processed", str(video_path))
                stats["processed"] += 1

        with mp.Pool(
            n_workers,
            initializer=init_fn,
            initargs=(args.method, args.target_fps, args.blur_threshold, args.brightness_min, args.brightness_max, args.min_usable_frames, args.min_sqi, args.max_nan_features, roi_weights, args.use_quality_weighting, args.quality_weight_min),
        ) as pool:
            results = iter(pool.imap_unordered(worker_fn, pending, chunksize=1))
            received = 0

            while received < len(pending):
                # Pause requested: drain one in-flight result so its work
                # is saved, checkpoint, then stop cleanly.
                if _PAUSE_REQUESTED:
                    print("\n[pause] Pausing pipeline. Finishing current sample then stopping.")
                    try:
                        entry = results.next(timeout=5.0)
                        received += 1
                        _record_result(entry)
                    except (mp.TimeoutError, StopIteration):
                        pass
                    ckpt_mgr.mark_paused()
                    print(f"[pause] {ckpt_mgr.state.processed_count} samples completed overall. "
                          f"Resume with --resume to continue.")
                    break

                try:
                    entry = results.next(timeout=ITEM_TIMEOUT_S)
                except mp.TimeoutError:
                    pool.terminate()
                    raise SystemExit(
                        f"FATAL: worker hung for > {ITEM_TIMEOUT_S}s "
                        f"(no result received after {received}/{len(pending)}). "
                        f"Pool terminated; re-run with --workers 1 to isolate."
                    ) from None
                except StopIteration:
                    break
                received += 1
                _record_result(entry)

                if stats["processed"] > 0 and stats["processed"] % 100 == 0:
                    done = stats["processed"] + stats["failed"] + stats["no_features"] + stats["gated"]
                    rate = done / (time.time() - t_start)
                    if rate > 0:
                        remain = int((len(pending) - received) / rate)
                        print(f"    ... {stats['processed']} ok / {stats['failed']} err / "
                              f"{stats['no_features']} no-feat / {stats['gated']} gated | "
                              f"{rate:.2f} vid/s | ETA ~{remain / 60:.0f} min")

        if stats["processed"] == 0 and not out_csv_path.exists():
            print("No features extracted from any videos.")
            return

        print(f"\nThis run: {stats['processed']} videos extracted "
              f"({stats['gated']} gated by signal quality, {stats['no_features']} with no features, "
              f"{stats['failed']} errors).")
        print(f"Features saved to {out_csv_path}")


if __name__ == "__main__":
    main()
