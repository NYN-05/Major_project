"""
gpu_face_detector.py
=====================
GPU-accelerated face detection using YuNet via ONNX Runtime with CUDA
execution provider. Replaces the CPU-bound MediaPipe TFLite face
detection for batch dataset extraction where throughput matters.

YuNet is a lightweight (~2 MB) face detector from the OpenCV Zoo that
runs in <10 ms on a laptop GPU.  It provides 5 facial keypoints (eyes,
nose, mouth corners) per face, which the existing bbox-based ROI
extraction path in ``face_roi.py`` can consume directly.

The model outputs raw multi-scale feature maps (strides 8/16/32) that
require anchor decoding + NMS post-processing, implemented here to
avoid a dependency on OpenCV's DNN module (which needs a CUDA build).

Usage::

    from .gpu_face_detector import GPUFaceDetector

    with GPUFaceDetector() as det:
        face = det.detect(frame_bgr, frame_index=0)
        # face is a TrackedFace with bbox set, landmarks_px=None
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .face_roi import TrackedFace

# ---------------------------------------------------------------------------
# Model download / cache
# ---------------------------------------------------------------------------

_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_YUNET_SHA256 = None  # not verified for upstream; file is small and immutable
_CACHE_DIR = Path.home() / ".cache" / "rppg_pipeline"
_MODEL_PATH = _CACHE_DIR / "face_detection_yunet_2023mar.onnx"

_MODEL_INPUT_SIZE = 640  # the 2023mar model expects exactly 640x640


def _ensure_model() -> str:
    """Download the YuNet ONNX model if not already cached."""
    if _MODEL_PATH.exists() and _MODEL_PATH.stat().st_size > 100_000:
        return str(_MODEL_PATH)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[gpu] Downloading YuNet face detector to {_MODEL_PATH} ...")
    tmp = str(_MODEL_PATH) + ".part"
    urllib.request.urlretrieve(_YUNET_URL, tmp)
    os.replace(tmp, _MODEL_PATH)
    print("[gpu] YuNet download complete.")
    return str(_MODEL_PATH)


# ---------------------------------------------------------------------------
# Anchor grid pre-computation (shared across all frames in a session)
# ---------------------------------------------------------------------------

_STRIDES = [8, 16, 32]


def _build_anchors(input_size: int = _MODEL_INPUT_SIZE) -> np.ndarray:
    """Pre-compute anchor grid positions for all three detection scales.

    Returns an array of shape ``(total_anchors, 2)`` with ``(col, row)``
    integer grid coordinates (NOT pixel coords).  Pixel coords are
    computed during decode as ``(col + offset) * stride``.
    """
    anchors: list[np.ndarray] = []
    for s in _STRIDES:
        cols = input_size // s
        rows = input_size // s
        col_idx, row_idx = np.meshgrid(np.arange(cols), np.arange(rows))
        grid = np.stack([col_idx.ravel(), row_idx.ravel()], axis=-1)  # (N, 2)
        anchors.append(grid.astype(np.float32))
    return np.concatenate(anchors, axis=0)


# ---------------------------------------------------------------------------
# Post-processing: anchor decode + score fusion + NMS
# ---------------------------------------------------------------------------

def _decode_outputs(
    outputs: list[np.ndarray],
    conf_thresh: float,
    nms_thresh: float,
    input_size: int,
    anchors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw YuNet multi-scale outputs into final detections.

    Parameters
    ----------
    outputs : list of 12 arrays [cls_8, cls_16, cls_32, obj_8, ...,
              obj_32, bbox_8, ..., bbox_32, kps_8, ..., kps_32]
    Returns
    -------
    boxes  : (N, 4) float32  — x1, y1, x2, y2 in model-input coords
    scores : (N,)   float32  — detection confidence
    kps    : (N, 5, 2) float32 — 5 keypoints (x, y) in model-input coords
    """
    # Concatenate all scales: each is (1, anchors_per_scale, dim).
    # Build per-anchor stride array so we can decode with the correct scale.
    counts = [input_size // s for s in _STRIDES]  # grid dims per scale
    total_per_scale = [c * c for c in counts]     # total anchors per scale
    cls_all = np.concatenate([outputs[i] for i in range(3)], axis=1)[0]      # (A, 1)
    obj_all = np.concatenate([outputs[i + 3] for i in range(3)], axis=1)[0]  # (A, 1)
    bbox_all = np.concatenate([outputs[i + 6] for i in range(3)], axis=1)[0] # (A, 4)
    kps_all = np.concatenate([outputs[i + 9] for i in range(3)], axis=1)[0]  # (A, 10)

    stride_arr = np.concatenate([
        np.full(n, s, dtype=np.float32) for n, s in zip(total_per_scale, _STRIDES)
    ])  # (A,)

    # Score: geometric mean of clamped cls and obj (already in [0,1] range).
    # Matches OpenCV FaceDetectorYN: score = sqrt(clamp(cls) * clamp(obj))
    cls_v = np.clip(cls_all[:, 0], 0.0, 1.0)
    obj_v = np.clip(obj_all[:, 0], 0.0, 1.0)
    score = np.sqrt(cls_v * obj_v)

    # Filter by confidence
    mask = score > conf_thresh
    if not mask.any():
        return (np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32))

    score = score[mask]
    bbox = bbox_all[mask]    # (N, 4): dx, dy, dw, dh
    kps_raw = kps_all[mask]  # (N, 10): 5x(kx, ky) offsets
    grid = anchors[mask]     # (N, 2): (col, row) integer grid positions
    strides = stride_arr[mask]  # (N,)

    # Decode bounding boxes (exact OpenCV FaceDetectorYN convention):
    #   cx = (col + dx) * stride
    #   cy = (row + dy) * stride
    #   w  = exp(dw) * stride
    #   h  = exp(dh) * stride
    cx = (grid[:, 0] + bbox[:, 0]) * strides
    cy = (grid[:, 1] + bbox[:, 1]) * strides
    w = np.exp(bbox[:, 2]) * strides
    h = np.exp(bbox[:, 3]) * strides
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    boxes = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)

    # Decode keypoints (exact OpenCV FaceDetectorYN convention):
    #   kx = (kx_offset + col) * stride
    #   ky = (ky_offset + row) * stride
    kps = np.zeros((len(kps_raw), 5, 2), dtype=np.float32)
    for k in range(5):
        kps[:, k, 0] = (kps_raw[:, k * 2] + grid[:, 0]) * strides
        kps[:, k, 1] = (kps_raw[:, k * 2 + 1] + grid[:, 1]) * strides

    # NMS per-class (YuNet is single-class)
    indices = cv2.dnn.NMSBoxes(
        [[float(boxes[i, 0]), float(boxes[i, 1]),
          float(boxes[i, 2] - boxes[i, 0]),
          float(boxes[i, 3] - boxes[i, 1])] for i in range(len(boxes))],
        [float(s) for s in score],
        conf_thresh,
        nms_thresh,
    )
    if len(indices) == 0:
        return (np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32))

    idx = indices.flatten()
    return boxes[idx], score[idx], kps[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GPUFaceDetector:
    """GPU-accelerated face detection using YuNet + ONNX Runtime CUDA.

    Produces ``TrackedFace`` objects compatible with the existing
    ``FaceROIExtractor.extract_rois`` bbox-fallback path.

    Parameters
    ----------
    conf_threshold : minimum detection confidence to keep a face.
    nms_threshold   : IoU threshold for non-maximum suppression.
    input_size      : model input resolution (must be 640 for the
                      2023mar model).
    """

    def __init__(
        self,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        input_size: int = _MODEL_INPUT_SIZE,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime-gpu is required for GPU face detection.\n"
                "Install with:  pip install onnxruntime-gpu>=1.18.1,<1.27.0"
            ) from exc

        model_path = _ensure_model()
        self._input_size = input_size
        self._conf_thresh = conf_threshold
        self._nms_thresh = nms_threshold
        self._anchors = _build_anchors(input_size)

        # Ensure PyTorch's bundled cuDNN is on the DLL search path
        # (onnxruntime-gpu needs cuDNN 9.x which PyTorch ships).
        torch_lib = self._find_torch_lib()
        if torch_lib:
            os.add_dll_directory(torch_lib)
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            model_path,
            providers=providers,
        )
        active = self._session.get_providers()
        self._gpu_active = "CUDAExecutionProvider" in active
        if not self._gpu_active:
            print("[gpu] WARNING: CUDA provider unavailable; "
                  "falling back to CPU (slow).")

    @staticmethod
    def _find_torch_lib() -> Optional[str]:
        """Locate the ``torch/lib`` directory containing cuDNN DLLs."""
        try:
            import torch
            lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isdir(lib):
                return lib
        except ImportError:
            pass
        return None

    @property
    def gpu_active(self) -> bool:
        return self._gpu_active

    def close(self):
        if hasattr(self, "_session") and self._session is not None:
            del self._session
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # -- preprocessing ---------------------------------------------------

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
        """Resize + pad a BGR frame exactly as OpenCV FaceDetectorYN does.

        Returns (input_tensor, scale_x, scale_y, pad_x, pad_y) where
        scale/pad factors map model-space coords back to original pixels.
        """
        h, w = frame_bgr.shape[:2]
        inp = self._input_size
        # Resize to model input (stretch to square, like OpenCV FaceDetectorYN)
        scale_x = inp / w
        scale_y = inp / h
        resized = cv2.resize(frame_bgr, (inp, inp), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB, uint8 -> float32 [0, 255], HWC -> CHW
        # NOTE: YuNet expects raw pixel values (0-255), NOT /255.
        # OpenCV's FaceDetectorYN uses blobFromImage(scale=1.0) internally.
        blob = resized[:, :, ::-1].astype(np.float32)
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 640, 640)
        return blob, scale_x, scale_y, 0.0, 0.0

    def _postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        kps: np.ndarray,
        orig_h: int,
        orig_w: int,
        scale_x: float,
        scale_y: float,
    ) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray]]:
        """Map detections from model-space back to original frame coords.

        Returns (bbox_or_None, confidence, kps_or_None).
        """
        if len(boxes) == 0:
            return None, 0.0, None

        # Take the highest-confidence face
        best = int(np.argmax(scores))
        x1, y1, x2, y2 = boxes[best]

        # Undo resize: divide by scale (OpenCV stretches, no padding offset)
        x1 = x1 / scale_x
        y1 = y1 / scale_y
        x2 = x2 / scale_x
        y2 = y2 / scale_y

        # Clamp to frame bounds
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

        bw = int(x2 - x1)
        bh = int(y2 - y1)
        if bw < 5 or bh < 5:
            return None, 0.0, None

        bbox = (int(x1), int(y1), bw, bh)

        # Keypoints (not used for ROI extraction but available)
        kp = kps[best]  # (5, 2) in model coords
        kp[:, 0] = kp[:, 0] / scale_x
        kp[:, 1] = kp[:, 1] / scale_y

        return bbox, float(scores[best]), kp

    # -- public detection entry point ------------------------------------

    def detect(self, frame_bgr: np.ndarray, frame_index: int) -> TrackedFace:
        """Detect the most prominent face in a frame.

        Returns a ``TrackedFace`` with ``bbox`` set and
        ``landmarks_px=None`` so that the existing bbox-based ROI
        extraction path in ``FaceROIExtractor`` handles the rest.
        """
        h, w = frame_bgr.shape[:2]
        blob, scale_x, scale_y, _pad_x, _pad_y = self._preprocess(frame_bgr)

        outputs = self._session.run(None, {"input": blob})
        boxes, scores, kps = _decode_outputs(
            outputs, self._conf_thresh, self._nms_thresh,
            self._input_size, self._anchors,
        )
        bbox, conf, _kp = self._postprocess(
            boxes, scores, kps, h, w, scale_x, scale_y,
        )

        if bbox is None:
            return TrackedFace(
                frame_index=frame_index,
                found=False,
                landmarks_px=None,
                bbox=None,
                confidence=0.0,
            )

        return TrackedFace(
            frame_index=frame_index,
            found=True,
            landmarks_px=None,  # triggers bbox-based ROI path
            bbox=bbox,
            confidence=conf,
        )
