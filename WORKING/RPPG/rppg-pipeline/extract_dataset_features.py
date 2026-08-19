"""
extract_dataset_features.py
===========================
Batch processes the available real and deepfake datasets to extract rPPG
features. The script now supports the new folder-based DFDC layout at
archive/DFDC_Dataset/ in addition to the older CSV-based archive (1)/ layout.

Outputs a CSV file dataset_features.csv for training the classifier.

Data-quality caveat (measured on the DFDC-derived 16-row table): every row
carries negative SNR (pulse buried in noise), HR features sit on a coarse
~24.3 BPM grid set by short-clip spectral binning at 10 fps, and the train
split is 10 rows — treat any metrics trained on this table as indicative
only, not deployment guarantees.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rppg import RPPGPipeline  # noqa: E402
from rppg.face_roi import FaceROIExtractor  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Upper bound (seconds) for one clip before the parent gives up on the worker.
ITEM_TIMEOUT_S = 600

_WORKER: dict = {}


def _init_worker(
    method: str,
    target_fps: Optional[float],
    blur_threshold: float,
    min_usable_frames: int,
    min_sqi: float,
    max_nan_features: int,
) -> None:
    """Per-process initializer: creates one RPPGPipeline per worker.

    MediaPipe/TFLite log to C++ stderr from every worker; fd 2 is
    redirected to NUL so the console stays readable (progress lines are
    printed by the parent, so workers need no stderr).
    """
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
        min_usable_frames=min_usable_frames,
    )
    _WORKER["min_sqi"] = min_sqi
    _WORKER["max_nan_features"] = max_nan_features


def _init_worker_gpu(
    method: str,
    target_fps: Optional[float],
    blur_threshold: float,
    min_usable_frames: int,
    min_sqi: float,
    max_nan_features: int,
) -> None:
    """Per-process initializer for GPU workers: creates RPPGPipeline +
    GPUFaceDetector + FaceROIExtractor (for trace accumulation)."""
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
        min_usable_frames=min_usable_frames,
    )
    _WORKER["min_sqi"] = min_sqi
    _WORKER["max_nan_features"] = max_nan_features
    from rppg.gpu_face_detector import GPUFaceDetector
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
    out_df.to_csv(out_csv_path, index=False)


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


def collect_samples(max_per_class: Optional[int] = None, include_ffpp: bool = False) -> List[Tuple[int, Path, str]]:
    root = _repo_root()
    samples: List[Tuple[int, Path, str]] = []

    new_dataset_root = root / "archive" / "DFDC_Dataset"
    if new_dataset_root.exists():
        fake_dir = new_dataset_root / "Fake"
        real_dir = new_dataset_root / "Real"

        fake_files = list(_iter_video_files(fake_dir)) if fake_dir.exists() else []
        real_files = list(_iter_video_files(real_dir)) if real_dir.exists() else []

        if fake_files or real_files:
            _cap_per_class([fake_files, real_files], max_per_class)

            for path in fake_files:
                _add_sample(samples, 1, path, "archive/DFDC_Dataset/Fake")
            for path in real_files:
                _add_sample(samples, 0, path, "archive/DFDC_Dataset/Real")

    if include_ffpp:
        # FF++ (FaceForensics++): FF-synthesis clips are fully re-rendered
        # fakes (Deepfakes/Face2Face/FaceShifter/NeuralTextures) where the
        # physiological pulse is genuinely synthesized/altered; FF-real and
        # YouTube-real are pristine real recordings.
        ffpp_root = _repo_root().parent.parent / "FF++"
        if ffpp_root.exists():
            for split in ("train", "val", "test"):
                synth_dir = ffpp_root / split / "FF-synthesis"
                real_dirs = [ffpp_root / split / "FF-real", ffpp_root / split / "YouTube-real"]
                synth_files = sorted(_iter_video_files(synth_dir)) if synth_dir.exists() else []
                real_files = []
                for rd in real_dirs:
                    if rd.exists():
                        real_files.extend(_iter_video_files(rd))
                real_files = sorted(real_files)
                if max_per_class is not None:
                    _cap_per_class([synth_files, real_files], max_per_class)
                for path in synth_files:
                    _add_sample(samples, 1, path, f"FF++/{split}/FF-synthesis")
                for path in real_files:
                    _add_sample(samples, 0, path, f"FF++/{split}/FF-real")

    legacy_root = root / "archive (1)"
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
    parser = argparse.ArgumentParser(description="Extract rPPG features from deepfake and real video datasets.")
    parser.add_argument("--method", default="POS", choices=["POS", "CHROM"], help="rPPG reconstruction method")
    parser.add_argument("--target-fps", type=float, default=None, help="Optional target FPS for sampling")
    parser.add_argument("--blur-threshold", type=float, default=15.0, help="Minimum Laplacian variance to keep a frame")
    parser.add_argument("--min-usable-frames", type=int, default=48, help="Minimum usable frames required per clip")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap for each label when extracting features")
    parser.add_argument("--include-ffpp", action="store_true", help="Also include FaceForensics++ clips (FF-synthesis fakes, FF-real/YouTube-real reals)")
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes (0 = all CPU cores)")
    parser.add_argument("--min-sqi", type=float, default=0.10, help="Drop clips whose signal_quality_index is below this (0 disables)")
    parser.add_argument("--max-nan-features", type=int, default=1, help="Drop clips with more than this many median-filled (raw-NaN) features")
    parser.add_argument("--gpu", action="store_true", help="Use GPU-accelerated face detection (YuNet via ONNX Runtime CUDA) instead of MediaPipe")
    parser.add_argument("--gpu-workers", type=int, default=None, help="Number of GPU worker processes (default: 8 when --gpu is set)")
    args = parser.parse_args()

    samples = collect_samples(max_per_class=args.max_per_class, include_ffpp=args.include_ffpp)
    if not samples:
        print("No dataset videos were found. Check archive/DFDC_Dataset or archive (1).")
        return

    root = _repo_root()
    out_csv_path = Path(args.output) if args.output else _output_dir() / "dataset_features.csv"
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if out_csv_path.exists():
        out_csv_path.unlink()
        print(f"Fresh extraction: removed existing {out_csv_path}")

    use_gpu = args.gpu
    if use_gpu:
        # Validate GPU availability before spawning workers
        try:
            import onnxruntime as _ort
            if "CUDAExecutionProvider" not in _ort.get_available_providers():
                print("WARNING: CUDA provider not available in onnxruntime. Falling back to CPU.")
                use_gpu = False
        except ImportError:
            print("WARNING: onnxruntime-gpu not installed. Falling back to CPU.\n"
                  "  Install with: pip install onnxruntime-gpu>=1.18.1,<1.27.0")
            use_gpu = False

    if use_gpu:
        n_workers = args.gpu_workers if args.gpu_workers else min(8, os.cpu_count() or 8)
        print(f"[gpu] Using GPU face detection (YuNet ONNX Runtime CUDA) with {n_workers} workers")
    elif args.workers == 0:
        n_workers = max(1, os.cpu_count() or 1)
    else:
        n_workers = args.workers

    items = [(label, video_path, source, root) for label, video_path, source in samples]

    features_list = []
    stats = {"processed": 0, "no_features": 0, "failed": 0, "gated": 0}
    t_start = time.time()

    if n_workers <= 1 and not use_gpu:
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
                print(f"  -> Failed to extract features: {video_path.name} {_fail_summary(result)}")
                stats["no_features"] += 1
                continue
            gate = _gate_result(result, args.min_sqi, args.max_nan_features)
            if gate is not None:
                print(f"  -> Gated (low signal quality): {video_path.name} ({gate})")
                stats["gated"] += 1
                continue
            entry = _feature_entry(result, label, video_path, root, source)
            features_list.append(entry)
            _write_features_csv(features_list, out_csv_path)
            stats["processed"] += 1
    else:
        worker_fn = _process_one_gpu if use_gpu else _process_one
        init_fn = _init_worker_gpu if use_gpu else _init_worker
        print(f"Extracting with {n_workers} parallel workers {'(GPU)' if use_gpu else '(CPU)'} ...")
        with mp.Pool(
            n_workers,
            initializer=init_fn,
            initargs=(args.method, args.target_fps, args.blur_threshold, args.min_usable_frames, args.min_sqi, args.max_nan_features),
        ) as pool:
            results = iter(pool.imap_unordered(worker_fn, items, chunksize=1))
            done = 0
            while done < len(items):
                try:
                    entry = results.next(timeout=ITEM_TIMEOUT_S)
                except mp.TimeoutError:
                    pool.terminate()
                    raise SystemExit(
                        f"FATAL: worker hung for > {ITEM_TIMEOUT_S}s on item {done + 1}/{len(items)} "
                        f"(no result received). Pool terminated; re-run with --workers 1 to isolate."
                    ) from None
                except StopIteration:
                    break
                done += 1
                error = entry.pop("error", None)
                no_features = entry.pop("no_features", None)
                gated = entry.pop("gated", None)
                if error:
                    stats["failed"] += 1
                    print(f"  -> Error processing {Path(entry['video_path']).name}: {error}")
                elif no_features:
                    stats["no_features"] += 1
                    n_usable = entry.get("usable_frames", 0)
                    n_total = entry.get("total_frames", 0)
                    n_no_face = entry.get("no_face", 0)
                    n_quality = max(0, n_total - n_usable - n_no_face)
                    print(
                        f"  -> Failed to extract features: {Path(entry['video_path']).name} "
                        f"(usable={n_usable}/{n_total}: no_face={n_no_face}, quality_rejected={n_quality})"
                    )
                elif gated:
                    stats["gated"] += 1
                    print(
                        f"  -> Gated (low signal quality): {Path(entry['video_path']).name} "
                        f"({entry.get('gate_reason', 'unknown')})"
                    )
                else:
                    features_list.append(entry)
                    _write_features_csv(features_list, out_csv_path)
                    stats["processed"] += 1
                if stats["processed"] > 0 and stats["processed"] % 100 == 0:
                    done = stats["processed"] + stats["failed"] + stats["no_features"] + stats["gated"]
                    rate = done / (time.time() - t_start)
                    remain = int((len(items) - done) / rate) if rate > 0 else 0
                    print(f"    ... {stats['processed']} ok / {stats['failed']} err / "
                          f"{stats['no_features']} no-feat / {stats['gated']} gated | "
                          f"{rate:.2f} vid/s | ETA ~{remain/60:.0f} min")

    if not features_list:
        print("No features extracted from any videos.")
        return

    print(f"\nSuccessfully extracted features for {len(features_list)} videos "
          f"({stats['gated']} gated by signal quality, {stats['no_features']} with no features, "
          f"{stats['failed']} errors).")
    print(f"Features saved to {out_csv_path} (created after the first video, updated after each subsequent one).")


if __name__ == "__main__":
    main()
