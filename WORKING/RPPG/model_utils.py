"""
model_utils.py
================
Locates the MediaPipe Face Landmarker model bundle (face_landmarker.task)
and the OpenCV Haar cascade used as fallback face detectors.

All model files are bundled in the repo — no internet download is needed
at runtime. Search order:
  1. Project directory (WORKING/RPPG/rppg/) — bundled models
  2. User cache (~/.cache/rppg_pipeline/) — legacy cached copies
  3. OpenCV data dir (cv2.data) — Haar cascade only

If none of these contain the file, a clear FileNotFoundError is raised.
"""

import hashlib
import os
from pathlib import Path

import cv2

# ── Bundled model locations (inside the repo) ──────────────────────────────
_REPOSITORY_MODELS_DIR = Path(__file__).resolve().parent  # WORKING/RPPG/rppg/

_BUNDLED_FACE_LANDMARKER = _REPOSITORY_MODELS_DIR / "face_landmarker.task"
_BUNDLED_HAAR_CASCADE = _REPOSITORY_MODELS_DIR / "haarcascade_frontalface_default.xml"

# ── Legacy cache locations (pre-bundled) ───────────────────────────────────
_CACHE_DIR = Path.home() / ".cache" / "rppg_pipeline"
_CACHE_FACE_LANDMARKER = _CACHE_DIR / "face_landmarker.task"
_CACHE_HAAR_CASCADE = _CACHE_DIR / "haarcascade_frontalface_default.xml"

# ── SHA-256 checksums (recorded from the bundled copies) ───────────────────
MODEL_SHA256 = "64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF"
HAAR_SHA256 = "0F7D4527844EB514D4A4948E822DA90FBB16A34A0BBBBC6ADC6498747A5AAFB0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _verify(path: Path, expected: str, what: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{what} checksum mismatch at {path}:\n"
            f"  expected SHA-256: {expected}\n"
            f"  actual   SHA-256: {actual}\n"
            "The bundled model file may be corrupted. Re-clone the repository."
        )


def _find_file(candidates: list[Path], label: str, expected_sha: str) -> str:
    """Return the first candidate that exists, passes size check and SHA-256
    verification. Raise FileNotFoundError with a clear message if none found."""
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            _verify(p, expected_sha, label)
            return str(p)
    locations = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"{label} not found.\n"
        "Searched:\n"
        f"  {locations}\n\n"
        f"This file should be bundled in the repository under WORKING/RPPG/rppg/.\n"
        "Re-clone the repository or restore the file from git history."
    )


def ensure_face_landmarker_model(dest_path: str = None) -> str:
    """
    Locate face_landmarker.task — bundled in the repo, no download needed.

    Parameters
    ----------
    dest_path : optional custom path override. If provided and the file
                exists there, it is returned directly (no search).
    """
    if dest_path and Path(dest_path).exists():
        _verify(Path(dest_path), MODEL_SHA256, "face_landmarker.task")
        return dest_path

    return _find_file(
        [_BUNDLED_FACE_LANDMARKER, _CACHE_FACE_LANDMARKER],
        "face_landmarker.task",
        MODEL_SHA256,
    )


def ensure_haar_cascade(dest_path: str = None) -> str:
    """
    Locate haarcascade_frontalface_default.xml — bundled in repo, or
    shipped inside opencv-python (cv2/data). No download needed.
    """
    if dest_path and Path(dest_path).exists():
        _verify(Path(dest_path), HAAR_SHA256, "haarcascade_frontalface_default.xml")
        return dest_path

    cv2_cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    return _find_file(
        [_BUNDLED_HAAR_CASCADE, _CACHE_HAAR_CASCADE, cv2_cascade],
        "haarcascade_frontalface_default.xml",
        HAAR_SHA256,
    )
