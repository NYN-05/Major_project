"""
pipeline.py
============
End-to-end orchestration: video file -> per-window physiological
feature vectors, ready for a downstream hybrid quantum-classical
classifier.

Stages
------
1. Frame extraction + basic quality assessment (blur, brightness)
2. Face landmark detection & tracking (face_roi.py)
3. ROI extraction: left cheek, right cheek, forehead (face_roi.py)
4. Raw per-ROI mean-RGB trace accumulation
5. Sliding-window rPPG signal reconstruction (signal_extraction.py)
6. Signal cleaning: detrend -> bandpass -> normalize (preprocessing.py)
7. Feature computation (features.py)
"""

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .face_roi import FaceROIExtractor
from .preprocessing import clean_signal
from .signal_extraction import extract_pulse_signal, combine_roi_signals
from .features import compute_features, RPPGFeatures


@dataclass
class FrameQuality:
    frame_index: int
    is_usable: bool
    blur_score: float
    brightness: float
    face_found: bool


@dataclass
class RPPGResult:
    """Full result for one processed video window/clip."""
    fps: float
    n_frames_total: int
    n_frames_usable: int
    features: Optional[RPPGFeatures]
    combined_signal: Optional[np.ndarray]
    left_cheek_signal: Optional[np.ndarray] = None
    right_cheek_signal: Optional[np.ndarray] = None
    forehead_signal: Optional[np.ndarray] = None
    quality_log: List[FrameQuality] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_feature_vector(self) -> Optional[np.ndarray]:
        return self.features.to_vector() if self.features is not None else None


def _blur_score(gray_frame: np.ndarray) -> float:
    """Variance of Laplacian; low values indicate a blurred frame."""
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


def _brightness(gray_frame: np.ndarray) -> float:
    return float(gray_frame.mean())


class RPPGPipeline:
    """
    High-level pipeline: point it at a video file and get back a
    fixed-length physiological feature vector plus intermediate
    signals for inspection/debugging.
    """

    def __init__(
        self,
        method: str = "POS",
        target_fps: Optional[float] = None,
        blur_threshold: float = 15.0,
        brightness_range: tuple = (25, 230),
        low_hz: float = 0.7,
        high_hz: float = 4.0,
        min_usable_frames: int = 48,
        roi_weights: tuple = (0.35, 0.35, 0.30),  # left cheek, right cheek, forehead
    ):
        """
        Parameters
        ----------
        method             : "POS" or "CHROM" rPPG reconstruction method.
        target_fps         : if set, frames are resampled/subsampled to
                              this rate; otherwise the video's native
                              fps is used.
        blur_threshold      : minimum Laplacian variance to keep a frame.
        brightness_range     : (min, max) mean pixel intensity to keep
                              a frame (rejects too-dark/too-bright frames).
        low_hz, high_hz      : physiological frequency band (Hz).
        min_usable_frames    : minimum number of usable frames required
                              to attempt signal extraction (~1.5-2s at
                              25-30fps).
        roi_weights          : weighting for combining left cheek /
                              right cheek / forehead signals.
        """
        self.method = method
        self.target_fps = target_fps
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.min_usable_frames = min_usable_frames
        self.roi_weights = roi_weights

    # -- frame quality ---------------------------------------------------

    def _assess_frame(self, frame_bgr: np.ndarray, idx: int, face_found: bool) -> FrameQuality:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = _blur_score(gray)
        bright = _brightness(gray)
        lo, hi = self.brightness_range
        usable = face_found and (blur >= self.blur_threshold) and (lo <= bright <= hi)
        return FrameQuality(
            frame_index=idx,
            is_usable=usable,
            blur_score=blur,
            brightness=bright,
            face_found=face_found,
        )

    # -- main entry point --------------------------------------------------

    def process_video(self, video_path: str) -> RPPGResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fps = self.target_fps or native_fps
        sample_stride = max(1, round(native_fps / fps)) if self.target_fps else 1

        warnings: List[str] = []
        quality_log: List[FrameQuality] = []

        left_trace, right_trace, forehead_trace = [], [], []

        frame_idx = 0
        kept_idx = 0

        with FaceROIExtractor() as extractor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_stride != 0:
                    frame_idx += 1
                    continue

                face = extractor.detect(frame, frame_idx)
                q = self._assess_frame(frame, frame_idx, face.found)
                quality_log.append(q)

                if q.is_usable:
                    rois = extractor.extract_rois(frame, face)
                    left_trace.append(extractor.mean_rgb(frame, rois.left_cheek))
                    right_trace.append(extractor.mean_rgb(frame, rois.right_cheek))
                    forehead_trace.append(extractor.mean_rgb(frame, rois.forehead))
                else:
                    left_trace.append(None)
                    right_trace.append(None)
                    forehead_trace.append(None)

                frame_idx += 1
                kept_idx += 1

        cap.release()

        n_total = len(quality_log)
        n_usable = sum(1 for q in quality_log if q.is_usable)

        if n_usable < self.min_usable_frames:
            warnings.append(
                f"Only {n_usable}/{n_total} usable frames "
                f"(need >= {self.min_usable_frames}); result unreliable."
            )
            return RPPGResult(
                fps=fps,
                n_frames_total=n_total,
                n_frames_usable=n_usable,
                features=None,
                combined_signal=None,
                quality_log=quality_log,
                warnings=warnings,
            )

        left_arr = self._traces_to_array(left_trace)
        right_arr = self._traces_to_array(right_trace)
        forehead_arr = self._traces_to_array(forehead_trace)

        left_sig = self._roi_to_pulse(left_arr, fps, warnings, "left cheek")
        right_sig = self._roi_to_pulse(right_arr, fps, warnings, "right cheek")
        forehead_sig = self._roi_to_pulse(forehead_arr, fps, warnings, "forehead")

        combined_raw = combine_roi_signals(
            [left_sig, right_sig, forehead_sig], weights=self.roi_weights
        )
        combined_clean = clean_signal(combined_raw, fs=fps, low_hz=self.low_hz, high_hz=self.high_hz)

        left_clean = clean_signal(left_sig, fs=fps, low_hz=self.low_hz, high_hz=self.high_hz) if left_sig is not None else None
        right_clean = clean_signal(right_sig, fs=fps, low_hz=self.low_hz, high_hz=self.high_hz) if right_sig is not None else None
        forehead_clean = clean_signal(forehead_sig, fs=fps, low_hz=self.low_hz, high_hz=self.high_hz) if forehead_sig is not None else None

        feats = compute_features(
            combined_signal=combined_clean,
            fs=fps,
            left_cheek_signal=left_clean,
            right_cheek_signal=right_clean,
            forehead_signal=forehead_clean,
            low_hz=self.low_hz,
            high_hz=self.high_hz,
        )

        return RPPGResult(
            fps=fps,
            n_frames_total=n_total,
            n_frames_usable=n_usable,
            features=feats,
            combined_signal=combined_clean,
            left_cheek_signal=left_clean,
            right_cheek_signal=right_clean,
            forehead_signal=forehead_clean,
            quality_log=quality_log,
            warnings=warnings,
        )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _traces_to_array(trace_list: List[Optional[np.ndarray]]) -> np.ndarray:
        """
        Convert a list of per-frame mean-RGB values (with possible
        None entries for occluded/rejected frames) into an (T, 3)
        array with NaN rows where data is missing, ready for
        interpolation downstream.
        """
        arr = np.full((len(trace_list), 3), np.nan, dtype=np.float64)
        for i, v in enumerate(trace_list):
            if v is not None:
                arr[i] = v
        return arr

    def _roi_to_pulse(
        self,
        rgb_arr: np.ndarray,
        fps: float,
        warnings: List[str],
        roi_name: str,
    ) -> Optional[np.ndarray]:
        valid_ratio = 1.0 - np.isnan(rgb_arr).any(axis=1).mean()
        if valid_ratio < 0.3:
            warnings.append(
                f"ROI '{roi_name}' had insufficient valid samples "
                f"({valid_ratio:.0%}); excluded from combination."
            )
            return None

        # Interpolate small gaps per-channel before signal reconstruction.
        filled = rgb_arr.copy()
        for c in range(3):
            col = filled[:, c]
            nans = np.isnan(col)
            if nans.any() and not nans.all():
                idx = np.arange(len(col))
                col[nans] = np.interp(idx[nans], idx[~nans], col[~nans])
                filled[:, c] = col

        if np.isnan(filled).any():
            return None

        return extract_pulse_signal(filled, fs=fps, method=self.method)
