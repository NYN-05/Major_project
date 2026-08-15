"""
face_roi.py
===========
Face landmark detection and physiological ROI (Region-of-Interest)
extraction using the MediaPipe Face Landmarker (Tasks API).

Why landmarks instead of a plain bounding box?
-----------------------------------------------
A bounding-box crop drifts with head pose and includes non-skin
pixels (hair, background, glasses) that corrupt the rPPG color
signal. Using the 468-point face mesh topology lets us carve out
precise, skin-rich patches (left cheek, right cheek, forehead) that
stay anatomically consistent even in low-resolution, compressed
video.

Note on MediaPipe API version
-------------------------------
MediaPipe 0.10.30+ dropped the legacy `mp.solutions.face_mesh` API
in favor of the newer Tasks API (`FaceLandmarker`), which this
module uses. It requires a small model file
(`face_landmarker.task`), which is auto-downloaded and cached on
first use via `model_utils.ensure_face_landmarker_model`.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from .model_utils import ensure_haar_cascade
except Exception:  # pragma: no cover - defensive import
    ensure_haar_cascade = None

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MEDIAPIPE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Landmark index groups (MediaPipe Face Mesh, 468-point topology)
# ---------------------------------------------------------------------------
# These indices define small polygons over skin-rich, motion-stable regions.
# Forehead: between eyebrows and hairline.
# Cheeks: below eyes, above jawline, avoiding nose shadow and ears.
LEFT_CHEEK_IDX = [116, 117, 118, 101, 36, 205, 187, 123]
RIGHT_CHEEK_IDX = [345, 346, 347, 330, 266, 425, 411, 352]
FOREHEAD_IDX = [109, 10, 338, 297, 332, 284, 251, 21, 54, 103]

# A minimal landmark set used for pose-stability / ROI smoothing.
STABILIZATION_IDX = [1, 33, 263, 61, 291, 199]  # nose tip, eyes, mouth corners, chin


def _load_haar_cascade() -> Optional[cv2.CascadeClassifier]:
    """
    Load the frontal-face Haar cascade used as a fallback when
    MediaPipe landmarks are unavailable. Some OpenCV distributions
    (notably opencv-python 5.0.x wheels) ship without the cascade XML
    files; in that case download the cascade once via model_utils.
    Returns None if the cascade cannot be made available.
    """
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(path) and ensure_haar_cascade is not None:
        try:
            path = ensure_haar_cascade()
        except Exception as e:
            print(f"[rppg] Could not obtain Haar cascade: {e}")
            return None
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        return None
    return cascade


@dataclass
class TrackedFace:
    """Per-frame face detection result."""
    frame_index: int
    found: bool
    landmarks_px: Optional[np.ndarray] = None  # (468, 2) pixel coords
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    confidence: float = 0.0


@dataclass
class ROISet:
    """Extracted ROI pixel patches for one frame."""
    frame_index: int
    left_cheek: Optional[np.ndarray] = None
    right_cheek: Optional[np.ndarray] = None
    forehead: Optional[np.ndarray] = None
    valid: bool = False


class LandmarkSmoother:
    """
    Exponential-moving-average smoother for facial landmarks.

    Prevents ROI jitter frame-to-frame, which otherwise injects
    high-frequency motion artifacts into the rPPG color signal —
    especially damaging in low-resolution video where detector
    noise is proportionally larger.
    """

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self._prev: Optional[np.ndarray] = None

    def smooth(self, landmarks_px: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = landmarks_px.copy()
            return landmarks_px
        smoothed = self.alpha * landmarks_px + (1 - self.alpha) * self._prev
        self._prev = smoothed
        return smoothed

    def reset(self):
        self._prev = None


if _MEDIAPIPE_AVAILABLE:
    class FaceROIExtractor:
        """
        Detects facial landmarks and extracts skin ROIs (left cheek,
        right cheek, forehead) suitable for rPPG signal extraction.
        """

        def __init__(
            self,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5,
            smoothing_alpha: float = 0.6,
            skin_mask_enabled: bool = True,
            model_path: Optional[str] = None,
        ):
            if not _MEDIAPIPE_AVAILABLE:
                raise ImportError(
                    "mediapipe is required for FaceROIExtractor. "
                    "Install with: pip install mediapipe"
                )

            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )
            from .model_utils import ensure_face_landmarker_model

            resolved_model_path = model_path or ensure_face_landmarker_model()

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=resolved_model_path),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = FaceLandmarker.create_from_options(options)
            self._smoother = LandmarkSmoother(alpha=smoothing_alpha)
            self.skin_mask_enabled = skin_mask_enabled
            self._last_timestamp_ms = -1
            # Haar fallback for environments/models that fail to detect
            self._fallback_cascade = _load_haar_cascade()

        def close(self):
            self._landmarker.close()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()

        # -- detection -----------------------------------------------------

        def detect(self, frame_bgr: np.ndarray, frame_index: int) -> TrackedFace:
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # detect_for_video requires a strictly increasing timestamp (ms).
            timestamp_ms = max(frame_index, self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms

            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.face_landmarks:
                # Try Haar cascade fallback on the RGB frame if MediaPipe
                # didn't find landmarks (useful for synthetic/test videos).
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                if getattr(self, '_fallback_cascade', None) is not None and not self._fallback_cascade.empty():
                    faces = self._fallback_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        return TrackedFace(frame_index=frame_index, found=True, landmarks_px=None, bbox=(x, y, w, h), confidence=0.6)
                # No evidence of a face: reject the frame. A guessed
                # centered bbox would feed non-facial pixels into the rPPG
                # signal and silently produce features (false-accept path).
                return TrackedFace(frame_index=frame_index, found=False, landmarks_px=None, bbox=None, confidence=0.0)

            lm_list = result.face_landmarks[0]
            pts = np.array([[p.x * w, p.y * h] for p in lm_list], dtype=np.float32)
            pts = self._smoother.smooth(pts)

            xs, ys = pts[:, 0], pts[:, 1]
            bbox = (
                int(max(0, xs.min())),
                int(max(0, ys.min())),
                int(min(w, xs.max()) - max(0, xs.min())),
                int(min(h, ys.max()) - max(0, ys.min())),
            )

            return TrackedFace(
                frame_index=frame_index,
                found=True,
                landmarks_px=pts,
                bbox=bbox,
                confidence=1.0,
            )

        def reset_tracking(self):
            self._smoother.reset()

        # -- ROI extraction --------------------------------------------------

        @staticmethod
        def _polygon_mask(shape: Tuple[int, int], points: np.ndarray) -> np.ndarray:
            mask = np.zeros(shape, dtype=np.uint8)
            hull = cv2.convexHull(points.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
            return mask

        @staticmethod
        def _skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
            ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            lower = np.array([0, 133, 77], dtype=np.uint8)
            upper = np.array([255, 173, 127], dtype=np.uint8)
            return cv2.inRange(ycrcb, lower, upper)

        def extract_rois(self, frame_bgr: np.ndarray, face: TrackedFace) -> ROISet:
            if not face.found:
                return ROISet(frame_index=face.frame_index, valid=False)

            h, w = frame_bgr.shape[:2]

            if face.landmarks_px is None:
                # Coarse ROI polygons derived from bounding box when
                # landmarks are unavailable (e.g., Haar fallback).
                if face.bbox is None:
                    return ROISet(frame_index=face.frame_index, valid=False)
                x, y, bw, bh = face.bbox

                def poly_from_bbox(rel_points):
                    pts = np.array([[x + int(px * bw), y + int(py * bh)] for px, py in rel_points], dtype=np.int32)
                    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                    return pts

                left_pts = poly_from_bbox([(0.12, 0.45), (0.32, 0.4), (0.38, 0.6), (0.18, 0.65)])
                right_pts = poly_from_bbox([(0.62, 0.45), (0.82, 0.4), (0.78, 0.6), (0.58, 0.65)])
                forehead_pts = poly_from_bbox([(0.35, 0.12), (0.65, 0.12), (0.65, 0.32), (0.35, 0.32)])

                left_mask = self._polygon_mask((h, w), left_pts)
                right_mask = self._polygon_mask((h, w), right_pts)
                forehead_mask = self._polygon_mask((h, w), forehead_pts)

                if self.skin_mask_enabled:
                    sm = self._skin_mask(frame_bgr)
                    left_and = cv2.bitwise_and(left_mask, sm)
                    right_and = cv2.bitwise_and(right_mask, sm)
                    fore_and = cv2.bitwise_and(forehead_mask, sm)
                    # If skin mask removes too many pixels (e.g., synthetic test
                    # faces), fall back to the original polygon mask.
                    left_mask = left_and if cv2.countNonZero(left_and) >= 25 else left_mask
                    right_mask = right_and if cv2.countNonZero(right_and) >= 25 else right_mask
                    forehead_mask = fore_and if cv2.countNonZero(fore_and) >= 25 else forehead_mask

                valid = any(cv2.countNonZero(m) > 25 for m in (left_mask, right_mask, forehead_mask))

                return ROISet(
                    frame_index=face.frame_index,
                    left_cheek=left_mask if cv2.countNonZero(left_mask) > 0 else None,
                    right_cheek=right_mask if cv2.countNonZero(right_mask) > 0 else None,
                    forehead=forehead_mask if cv2.countNonZero(forehead_mask) > 0 else None,
                    valid=valid,
                )

            pts = face.landmarks_px

            # Skin mask is shared by all three ROIs; compute it once per frame
            # instead of inside the region() closure (3x/frame previously).
            skin = self._skin_mask(frame_bgr) if self.skin_mask_enabled else None

            def region(idx_list) -> Optional[np.ndarray]:
                region_pts = pts[idx_list]
                mask = self._polygon_mask((h, w), region_pts)
                if skin is not None:
                    mask = cv2.bitwise_and(mask, skin)
                if cv2.countNonZero(mask) < 25:
                    return None
                return mask

            left_mask = region(LEFT_CHEEK_IDX)
            right_mask = region(RIGHT_CHEEK_IDX)
            forehead_mask = region(FOREHEAD_IDX)

            valid = any(m is not None for m in (left_mask, right_mask, forehead_mask))

            return ROISet(
                frame_index=face.frame_index,
                left_cheek=left_mask,
                right_cheek=right_mask,
                forehead=forehead_mask,
                valid=valid,
            )

        @staticmethod
        def mean_rgb(frame_bgr: np.ndarray, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if mask is None:
                return None
            if cv2.countNonZero(mask) == 0:
                return None
            # cv2.mean avoids materializing the masked pixel array
            # (frame_bgr[mask > 0]) that the old fancy-index version built
            # per ROI per frame.
            mean_bgr = cv2.mean(frame_bgr, mask)
            return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]])
else:
    class FaceROIExtractor:
        """Fallback lightweight face ROI extractor using OpenCV Haar cascades.

        This is a pragmatic fallback for environments without MediaPipe
        so the pipeline remains runnable for demos/tests. It provides
        coarse ROI masks (cheeks/forehead) derived from the face bbox.
        """

        def __init__(self, skin_mask_enabled: bool = True, **kwargs):
            self._cascade = _load_haar_cascade()
            self.skin_mask_enabled = skin_mask_enabled

        def close(self):
            return

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return

        def detect(self, frame_bgr: np.ndarray, frame_index: int) -> TrackedFace:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = ()
            if self._cascade is not None:
                faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                # No evidence of a face: reject the frame rather than
                # computing features on a guessed centered bbox.
                return TrackedFace(frame_index=frame_index, found=False, landmarks_px=None, bbox=None, confidence=0.0)
            x, y, w, h = faces[0]
            # approximate landmarks as None; bbox provided
            return TrackedFace(frame_index=frame_index, found=True, landmarks_px=None, bbox=(x, y, w, h), confidence=0.8)

        def reset_tracking(self):
            return

        def _polygon_mask(self, shape: Tuple[int, int], points: np.ndarray) -> np.ndarray:
            mask = np.zeros(shape, dtype=np.uint8)
            hull = cv2.convexHull(points.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
            return mask

        def extract_rois(self, frame_bgr: np.ndarray, face: TrackedFace) -> ROISet:
            if not face.found or face.bbox is None:
                return ROISet(frame_index=face.frame_index, valid=False)

            fh, fw = frame_bgr.shape[:2]
            x, y, w, h = face.bbox

            def poly_from_bbox(rel_points):
                pts = np.array([[x + int(px * w), y + int(py * h)] for px, py in rel_points], dtype=np.int32)
                pts[:, 0] = np.clip(pts[:, 0], 0, fw - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, fh - 1)
                return pts

            left_pts = poly_from_bbox([(0.12, 0.45), (0.32, 0.4), (0.38, 0.6), (0.18, 0.65)])
            right_pts = poly_from_bbox([(0.62, 0.45), (0.82, 0.4), (0.78, 0.6), (0.58, 0.65)])
            forehead_pts = poly_from_bbox([(0.35, 0.12), (0.65, 0.12), (0.65, 0.32), (0.35, 0.32)])

            left_mask = self._polygon_mask((fh, fw), left_pts)
            right_mask = self._polygon_mask((fh, fw), right_pts)
            forehead_mask = self._polygon_mask((fh, fw), forehead_pts)

            valid = any(cv2.countNonZero(m) > 25 for m in (left_mask, right_mask, forehead_mask))

            return ROISet(
                frame_index=face.frame_index,
                left_cheek=left_mask if cv2.countNonZero(left_mask) > 0 else None,
                right_cheek=right_mask if cv2.countNonZero(right_mask) > 0 else None,
                forehead=forehead_mask if cv2.countNonZero(forehead_mask) > 0 else None,
                valid=valid,
            )

        @staticmethod
        def mean_rgb(frame_bgr: np.ndarray, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if mask is None:
                return None
            if cv2.countNonZero(mask) == 0:
                return None
            # cv2.mean avoids materializing the masked pixel array
            # (frame_bgr[mask > 0]) that the old fancy-index version built
            # per ROI per frame.
            mean_bgr = cv2.mean(frame_bgr, mask)
            return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]])
