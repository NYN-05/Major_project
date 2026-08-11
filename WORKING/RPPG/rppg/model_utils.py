"""
model_utils.py
================
Downloads and caches the MediaPipe Face Landmarker model bundle
(`face_landmarker.task`) used by the new MediaPipe Tasks API, plus the
OpenCV Haar cascade used as a fallback face detector.

Newer MediaPipe releases (0.10.30+) dropped the legacy
`mp.solutions.face_mesh` API in favor of the Tasks API, which
requires an explicit model asset file rather than a bundled one.
This module downloads that file once and caches it locally so
subsequent runs don't re-download it.
"""

import os
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rppg_pipeline"
DEFAULT_MODEL_PATH = DEFAULT_CACHE_DIR / "face_landmarker.task"


def ensure_face_landmarker_model(dest_path: str = None) -> str:
    """
    Ensure the face_landmarker.task model file exists locally,
    downloading it if necessary. Returns the local file path.

    Parameters
    ----------
    dest_path : optional custom path to store/look for the model.
                Defaults to ~/.cache/rppg_pipeline/face_landmarker.task
    """
    dest = Path(dest_path) if dest_path else DEFAULT_MODEL_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)

    print(f"[rppg] Downloading face landmark model to {dest} ...")
    try:
        tmp_path = str(dest) + ".part"
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        os.replace(tmp_path, dest)
        print("[rppg] Model download complete.")
    except Exception as e:
        raise RuntimeError(
            "Failed to auto-download the face landmark model "
            f"from {MODEL_URL}.\n"
            "This usually means no internet access or a firewall block.\n\n"
            "Fix options:\n"
            f"  1) Check your internet connection and retry.\n"
            f"  2) Manually download the file from:\n"
            f"       {MODEL_URL}\n"
            f"     and place it at:\n"
            f"       {dest}\n"
            f"  3) Pass a custom path via "
            f"FaceROIExtractor(model_path='...') pointing to an "
            f"already-downloaded copy.\n\n"
            f"Original error: {e}"
        ) from e

    return str(dest)


HAAR_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)


def ensure_haar_cascade(dest_path: str = None) -> str:
    """
    Ensure a frontal-face Haar cascade XML is available locally,
    downloading it from the OpenCV GitHub repo if the OpenCV wheel did
    not ship with cascade data (opencv-python 5.0.x wheels ship an
    empty cv2/data directory). Returns the local file path.
    """
    if dest_path is None:
        dest = DEFAULT_CACHE_DIR / "haarcascade_frontalface_default.xml"
    else:
        dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)

    print(f"[rppg] Downloading Haar cascade to {dest} ...")
    try:
        tmp_path = str(dest) + ".part"
        urllib.request.urlretrieve(HAAR_CASCADE_URL, tmp_path)
        os.replace(tmp_path, dest)
        print("[rppg] Haar cascade download complete.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to auto-download the Haar cascade from {HAAR_CASCADE_URL}.\n"
            f"Original error: {e}"
        ) from e

    return str(dest)
