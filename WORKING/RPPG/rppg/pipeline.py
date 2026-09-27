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

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .face_roi import FaceROIExtractor
from .preprocessing import clean_signal
from .signal_extraction import extract_pulse_signal, combine_roi_signals
from .features import compute_features, RPPGFeatures
from .pqs import compute_pqs, PQSResult

_FRAME_RE = re.compile(r"frame_(\d+)_t(\d+)\.jpg")


def _frame_sort_key(path: Path) -> int:
    match = _FRAME_RE.search(path.name)
    return int(match.group(1)) if match else -1


@dataclass
class FrameQuality:
    frame_index: int
    is_usable: bool
    blur_score: float
    brightness: float
    face_found: bool
    # Phase 2: continuous quality weight [0, 1]
    quality_score: float = 0.0
    # Phase 2: individual quality components for analysis
    face_size_score: float = 0.0
    landmark_stability_score: float = 0.0
    roi_validity_score: float = 0.0
    signal_amplitude_score: float = 0.0


@dataclass
class WindowResult:
    """Result for a single temporal window."""
    window_index: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    n_frames_total: int
    n_frames_usable: int
    features: Optional[RPPGFeatures]
    combined_signal: Optional[np.ndarray]
    quality_score_mean: float
    warnings: List[str] = field(default_factory=list)
    features_valid: bool = True  # False if window failed quality gates

    def to_feature_vector(self) -> Optional[np.ndarray]:
        return self.features.to_vector() if self.features is not None else None


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
    # Phase 3: temporal window analysis
    window_results: List[WindowResult] = field(default_factory=list)
    window_features_aggregated: Optional[RPPGFeatures] = None
    window_feature_stats: dict = field(default_factory=dict)
    # Phase 5: Physiological Quality Score
    pqs: Optional[PQSResult] = None

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
        blur_threshold: float = 5.0,
        brightness_min: int = 20,
        brightness_max: int = 240,
        low_hz: float = 0.7,
        high_hz: float = 4.0,
        min_usable_frames: int = 48,
        min_sqi: float = 0.10,
        roi_weights: tuple = (0.35, 0.35, 0.30),  # left cheek, right cheek, forehead
        # Phase 2: quality-weighted rPPG
        use_quality_weighting: bool = False,
        quality_weight_min: float = 0.05,  # minimum weight for extremely poor frames
        # Phase 3: temporal window analysis
        window_duration_sec: float = 8.0,  # duration of each window in seconds
        window_overlap_sec: float = 4.0,   # overlap between windows in seconds
        min_window_usable_frames: int = 24,  # minimum usable frames per window
        enable_window_analysis: bool = False,  # enable temporal window analysis
    ):
        """
        Parameters
        ----------
        method             : "POS" or "CHROM" rPPG reconstruction method.
        target_fps         : if set, frames are resampled/subsampled to
                              this rate; otherwise the video's native
                              fps is used.
        blur_threshold      : minimum Laplacian variance to keep a frame.
                              Lowered from 15.0 to 5.0 (P7) for low-res compressed video.
        brightness_min       : minimum mean pixel intensity to keep a frame.
                              Lowered from 25 to 20 (P7) for darker videos.
        brightness_max       : maximum mean pixel intensity to keep a frame.
                              Raised from 230 to 240 (P7) for brighter videos.
        low_hz, high_hz      : physiological frequency band (Hz).
        min_usable_frames    : minimum number of usable frames required
                                to attempt signal extraction (~1.5-2s at
                                25-30fps).
        min_sqi              : minimum signal quality index to accept a clip
                                (aligned with training gate in extract_dataset_features.py).
        roi_weights          : weighting for combining left cheek /
                                right cheek / forehead signals.
        use_quality_weighting: if True, use continuous quality weights instead
                                of binary frame rejection (Phase 2).
        quality_weight_min   : minimum weight for extremely poor frames when
                                quality weighting is enabled.
        window_duration_sec  : duration of each temporal window in seconds (Phase 3).
        window_overlap_sec   : overlap between consecutive windows in seconds (Phase 3).
        min_window_usable_frames : minimum usable frames per window (Phase 3).
        enable_window_analysis : enable temporal window analysis (Phase 3).
        """
        self.method = method
        self.target_fps = target_fps
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.min_usable_frames = min_usable_frames
        self.min_sqi = min_sqi
        self.roi_weights = roi_weights
        # Phase 2
        self.use_quality_weighting = use_quality_weighting
        self.quality_weight_min = quality_weight_min
        # Phase 3
        self.window_duration_sec = window_duration_sec
        self.window_overlap_sec = window_overlap_sec
        self.min_window_usable_frames = min_window_usable_frames
        self.enable_window_analysis = enable_window_analysis

        # Quality scoring thresholds (for normalization to [0,1])
        self._blur_max = 500.0  # Laplacian variance above this = perfect blur score
        self._face_size_min_ratio = 0.005  # minimum face area ratio
        self._face_size_max_ratio = 0.5    # maximum face area ratio (full frame)
        self._roi_min_pixels = 10          # minimum ROI pixels for validity
        self._roi_max_pixels = 5000        # maximum ROI pixels for full score
        self._signal_amplitude_min = 1.0   # minimum RGB signal amplitude
        self._signal_amplitude_max = 50.0  # maximum RGB signal amplitude for full score

    # -- frame quality ---------------------------------------------------

    def _compute_quality_components(
        self,
        frame_bgr: np.ndarray,
        face,
        left_roi: Optional[np.ndarray] = None,
        right_roi: Optional[np.ndarray] = None,
        forehead_roi: Optional[np.ndarray] = None,
        left_rgb: Optional[np.ndarray] = None,
        right_rgb: Optional[np.ndarray] = None,
        forehead_rgb: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Compute individual quality components for a frame.

        Returns:
            tuple: (blur_norm, brightness_norm, face_size_score, landmark_stability_score,
                    roi_validity_score, signal_amplitude_score)
        """
        h, w = frame_bgr.shape[:2]

        # 1. Blur score (normalized to [0, 1])
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = _blur_score(gray)
        blur_norm = min(blur / self._blur_max, 1.0)

        # 2. Brightness score (normalized - penalize too dark or too bright)
        bright = _brightness(gray)
        # Physiological range: 50-200 is good, outside is penalized
        if 50 <= bright <= 200:
            brightness_norm = 1.0
        elif bright < 50:
            brightness_norm = bright / 50.0
        else:
            brightness_norm = max(0.0, 1.0 - (bright - 200) / 55.0)

        # 3. Face size score (based on bbox area ratio)
        face_size_score = 0.0
        if face.found and face.bbox is not None:
            x, y, bw, bh = face.bbox
            face_area_ratio = (bw * bh) / (w * h)
            if face_area_ratio >= self._face_size_min_ratio:
                face_size_score = min(face_area_ratio / self._face_size_max_ratio, 1.0)
            else:
                face_size_score = face_area_ratio / self._face_size_min_ratio

        # 4. Landmark stability score (from smoother - only available with MediaPipe)
        landmark_stability_score = 1.0 if face.found and face.landmarks_px is not None else 0.5

        # 5. ROI validity score (based on valid pixels in ROI masks)
        roi_validity_score = 0.0
        roi_masks = []
        if left_roi is not None and cv2.countNonZero(left_roi) > 0:
            roi_masks.append(left_roi)
        if right_roi is not None and cv2.countNonZero(right_roi) > 0:
            roi_masks.append(right_roi)
        if forehead_roi is not None and cv2.countNonZero(forehead_roi) > 0:
            roi_masks.append(forehead_roi)

        if roi_masks:
            valid_roi_count = len(roi_masks)
            total_pixels = sum(cv2.countNonZero(m) for m in roi_masks)
            avg_pixels = total_pixels / max(valid_roi_count, 1)
            roi_validity_score = min(avg_pixels / self._roi_max_pixels, 1.0) if avg_pixels >= self._roi_min_pixels else avg_pixels / self._roi_min_pixels * 0.5
        else:
            roi_validity_score = 0.0

        # 6. Signal amplitude score (based on RGB trace magnitude)
        signal_amplitude_score = 0.0
        rgb_traces = []
        if left_rgb is not None:
            rgb_traces.append(left_rgb)
        if right_rgb is not None:
            rgb_traces.append(right_rgb)
        if forehead_rgb is not None:
            rgb_traces.append(forehead_rgb)

        if rgb_traces:
            amplitudes = [np.std(trace) for trace in rgb_traces if trace is not None]
            if amplitudes:
                avg_amplitude = np.mean(amplitudes)
                if avg_amplitude >= self._signal_amplitude_min:
                    signal_amplitude_score = min(avg_amplitude / self._signal_amplitude_max, 1.0)
                else:
                    signal_amplitude_score = avg_amplitude / self._signal_amplitude_min * 0.5

        return blur_norm, brightness_norm, face_size_score, landmark_stability_score, roi_validity_score, signal_amplitude_score

    def _compute_quality_weight(self, components: tuple) -> float:
        """
        Combine quality components into a single weight in [0, 1].

        Uses weighted geometric mean to ensure that a zero in any critical
        component (face, ROI) strongly reduces the weight.
        """
        blur_norm, brightness_norm, face_size_score, landmark_stability_score, roi_validity_score, signal_amplitude_score = components

        # Critical components (if zero, frame is essentially unusable)
        critical_weights = [face_size_score, roi_validity_score]
        # Quality components (contribute to weight but don't zero it out)
        quality_weights = [blur_norm, brightness_norm, landmark_stability_score, signal_amplitude_score]

        # Geometric mean of critical components (zero if any is zero)
        if any(c <= 0 for c in critical_weights):
            critical_score = 0.0
        else:
            critical_score = np.prod(critical_weights) ** (1.0 / len(critical_weights))

        # Arithmetic mean of quality components
        quality_score = np.mean(quality_weights) if quality_weights else 0.5

        # Combined weight: critical_score gates the quality_score
        weight = critical_score * quality_score

        # Apply minimum weight floor
        weight = max(weight, self.quality_weight_min)

        return float(np.clip(weight, 0.0, 1.0))

    def _assess_frame(
        self,
        frame_bgr: np.ndarray,
        idx: int,
        face_found: bool,
        face=None,
        left_roi: Optional[np.ndarray] = None,
        right_roi: Optional[np.ndarray] = None,
        forehead_roi: Optional[np.ndarray] = None,
        left_rgb: Optional[np.ndarray] = None,
        right_rgb: Optional[np.ndarray] = None,
        forehead_rgb: Optional[np.ndarray] = None,
        compute_quality: bool = True,
    ) -> FrameQuality:
        if not face_found:
            # Without a face the frame can never be usable; blur/brightness
            # cannot change that, so skip the expensive Laplacian entirely.
            return FrameQuality(
                frame_index=idx,
                is_usable=False,
                blur_score=0.0,
                brightness=0.0,
                face_found=False,
                quality_score=0.0,
            )

        if not compute_quality:
            # Caller already quality-gated these frames (stage 1); only the
            # face gate matters here, so skip blur/brightness.
            # Still compute a basic quality score for weighting
            components = self._compute_quality_components(
                frame_bgr, face, left_roi, right_roi, forehead_roi,
                left_rgb, right_rgb, forehead_rgb
            )
            quality_score = self._compute_quality_weight(components)
            return FrameQuality(
                frame_index=idx,
                is_usable=True,
                blur_score=0.0,
                brightness=0.0,
                face_found=True,
                quality_score=quality_score,
                face_size_score=components[2],
                landmark_stability_score=components[3],
                roi_validity_score=components[4],
                signal_amplitude_score=components[5],
            )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = _blur_score(gray)
        bright = _brightness(gray)

        # Binary usability gate (original behavior)
        usable = (blur >= self.blur_threshold) and (self.brightness_min <= bright <= self.brightness_max)

        # Phase 2: compute continuous quality weight
        if self.use_quality_weighting:
            components = self._compute_quality_components(
                frame_bgr, face, left_roi, right_roi, forehead_roi,
                left_rgb, right_rgb, forehead_rgb
            )
            quality_score = self._compute_quality_weight(components)
        else:
            quality_score = 1.0 if usable else 0.0

        return FrameQuality(
            frame_index=idx,
            is_usable=usable,
            blur_score=blur,
            brightness=bright,
            face_found=True,
            quality_score=quality_score,
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

        with FaceROIExtractor() as extractor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_stride != 0:
                    frame_idx += 1
                    continue

                face = extractor.detect(frame, frame_idx)
                
                # First assess frame without ROI data (for binary gate)
                q = self._assess_frame(frame, frame_idx, face.found, face=face)
                quality_log.append(q)

                if q.is_usable:
                    rois = extractor.extract_rois(frame, face)
                    left_rgb = extractor.mean_rgb(frame, rois.left_cheek)
                    right_rgb = extractor.mean_rgb(frame, rois.right_cheek)
                    forehead_rgb = extractor.mean_rgb(frame, rois.forehead)
                    left_trace.append(left_rgb)
                    right_trace.append(right_rgb)
                    forehead_trace.append(forehead_rgb)
                    
                    # Re-assess with ROI and RGB data for quality metrics (always, not just for weighting)
                    q_full = self._assess_frame(
                        frame, frame_idx, face.found, face=face,
                        left_roi=rois.left_cheek, right_roi=rois.right_cheek, forehead_roi=rois.forehead,
                        left_rgb=left_rgb, right_rgb=right_rgb, forehead_rgb=forehead_rgb,
                        compute_quality=self.use_quality_weighting
                    )
                    quality_log[-1] = q_full  # Replace with full assessment
                else:
                    left_trace.append(None)
                    right_trace.append(None)
                    forehead_trace.append(None)

                frame_idx += 1

        cap.release()

        # If window analysis is enabled, process temporal windows
        if self.enable_window_analysis:
            return self._process_windows(
                left_trace, right_trace, forehead_trace, 
                fps, warnings, quality_log
            )
        
        return self._finalize(left_trace, right_trace, forehead_trace, fps, warnings, quality_log)

    def _process_windows(
        self,
        left_trace: List[Optional[np.ndarray]],
        right_trace: List[Optional[np.ndarray]],
        forehead_trace: List[Optional[np.ndarray]],
        fps: float,
        warnings: List[str],
        quality_log: List[FrameQuality],
        # We need the full quality log for window quality scoring
    ) -> RPPGResult:
        """
        Process video in overlapping temporal windows (Phase 3).
        
        Divides the full trace into overlapping windows and extracts
        features for each window, then aggregates window-level features
        into video-level statistics.
        """
        n_total = len(quality_log)
        
        # Calculate window parameters in frames
        window_frames = max(1, int(round(self.window_duration_sec * fps)))
        overlap_frames = max(1, int(round(self.window_overlap_sec * fps)))
        stride_frames = window_frames - overlap_frames
        
        if stride_frames <= 0:
            warnings.append(f"Invalid window config: overlap >= duration. Using whole clip.")
            return self._finalize(left_trace, right_trace, forehead_trace, fps, warnings, quality_log)
        
        # Extract quality weights for window quality assessment
        quality_weights = np.array([q.quality_score for q in quality_log], dtype=np.float64)
        
        # Generate window indices
        window_starts = list(range(0, max(1, n_total - window_frames + 1), stride_frames))
        if not window_starts:
            warnings.append("Video too short for window analysis. Using whole clip.")
            return self._finalize(left_trace, right_trace, forehead_trace, fps, warnings, quality_log)
        
        window_results: List[WindowResult] = []
        failed_windows = 0
        
        for win_idx, start_idx in enumerate(window_starts):
            end_idx = min(start_idx + window_frames, n_total)
            
            # Extract window traces
            win_left = left_trace[start_idx:end_idx]
            win_right = right_trace[start_idx:end_idx]
            win_forehead = forehead_trace[start_idx:end_idx]
            win_quality = quality_log[start_idx:end_idx]
            win_quality_weights = quality_weights[start_idx:end_idx]
            
            n_win_total = len(win_quality)
            n_win_usable = sum(1 for q in win_quality if q.is_usable)
            
            # Check minimum usable frames for this window
            if n_win_usable < self.min_window_usable_frames:
                failed_windows += 1
                win_result = WindowResult(
                    window_index=win_idx,
                    start_frame=start_idx,
                    end_frame=end_idx - 1,
                    start_time_sec=start_idx / fps,
                    end_time_sec=(end_idx - 1) / fps,
                    n_frames_total=n_win_total,
                    n_frames_usable=n_win_usable,
                    features=None,
                    combined_signal=None,
                    quality_score_mean=float(np.mean(win_quality_weights)) if len(win_quality_weights) > 0 else 0.0,
                    warnings=[f"Window {win_idx}: insufficient usable frames ({n_win_usable}/{n_win_total})"],
                    features_valid=False,
                )
                window_results.append(win_result)
                continue
            
            # Process this window (similar to _finalize but for window)
            win_result = self._process_single_window(
                win_left, win_right, win_forehead,
                fps, win_quality, win_quality_weights,
                win_idx, start_idx, end_idx
            )
            window_results.append(win_result)
            if not win_result.features_valid:
                failed_windows += 1
        
        if failed_windows == len(window_results):
            warnings.append(f"All {len(window_results)} windows failed quality gates. Using whole clip.")
            return self._finalize(left_trace, right_trace, forehead_trace, fps, warnings, quality_log)
        
        # Aggregate window features into video-level representation
        aggregated_features, feature_stats = self._aggregate_window_features(window_results, fps)
        
        # Also compute whole-clip features for comparison
        whole_clip_result = self._finalize(left_trace, right_trace, forehead_trace, fps, warnings, quality_log)
        
        # Add window analysis info to warnings
        valid_windows = [w for w in window_results if w.features_valid]
        warnings.append(
            f"Window analysis: {len(valid_windows)}/{len(window_results)} windows valid "
            f"({failed_windows} failed), window={self.window_duration_sec:.1f}s, "
            f"overlap={self.window_overlap_sec:.1f}s, stride={stride_frames/fps:.2f}s"
        )
        
        # Phase 5: Compute PQS for the whole-clip result
        pqs_result = compute_pqs(whole_clip_result)
        whole_clip_warnings = whole_clip_result.warnings + warnings
        whole_clip_warnings.append(
            f"PQS: {pqs_result.pqs:.3f} ({pqs_result.quality_tier}), "
            f"components: SNR={pqs_result.components.snr_quality:.2f}, "
            f"frames={pqs_result.components.frame_utilization:.2f}, "
            f"ROI={pqs_result.components.roi_validity:.2f}, "
            f"crossROI={pqs_result.components.cross_roi_consistency:.2f}, "
            f"freq={pqs_result.components.frequency_stability:.2f}, "
            f"amp={pqs_result.components.signal_amplitude:.2f}, "
            f"temp={pqs_result.components.temporal_consistency:.2f}"
        )

        # Return result with both whole-clip and window-aggregated features
        return RPPGResult(
            fps=whole_clip_result.fps,
            n_frames_total=whole_clip_result.n_frames_total,
            n_frames_usable=whole_clip_result.n_frames_usable,
            features=whole_clip_result.features,  # Keep whole-clip features as primary
            combined_signal=whole_clip_result.combined_signal,
            left_cheek_signal=whole_clip_result.left_cheek_signal,
            right_cheek_signal=whole_clip_result.right_cheek_signal,
            forehead_signal=whole_clip_result.forehead_signal,
            quality_log=whole_clip_result.quality_log,
            warnings=whole_clip_warnings,
            window_results=window_results,
            window_features_aggregated=aggregated_features,
            window_feature_stats=feature_stats,
            pqs=pqs_result,
        )

    def _process_single_window(
        self,
        left_trace: List[Optional[np.ndarray]],
        right_trace: List[Optional[np.ndarray]],
        forehead_trace: List[Optional[np.ndarray]],
        fps: float,
        quality_log: List[FrameQuality],
        quality_weights: np.ndarray,
        window_index: int,
        start_idx: int,
        end_idx: int,
    ) -> WindowResult:
        """Process a single temporal window and extract features."""
        n_win_total = len(quality_log)
        n_win_usable = sum(1 for q in quality_log if q.is_usable)
        quality_score_mean = float(np.mean(quality_weights)) if len(quality_weights) > 0 else 0.0
        
        # Convert traces to arrays (with quality weighting if enabled)
        if self.use_quality_weighting:
            left_arr = self._traces_to_array_weighted(left_trace, quality_weights)
            right_arr = self._traces_to_array_weighted(right_trace, quality_weights)
            forehead_arr = self._traces_to_array_weighted(forehead_trace, quality_weights)
        else:
            left_arr = self._traces_to_array(left_trace)
            right_arr = self._traces_to_array(right_trace)
            forehead_arr = self._traces_to_array(forehead_trace)
        
        # Extract pulse signals
        left_sig = self._roi_to_pulse(left_arr, fps, [], "left cheek")
        right_sig = self._roi_to_pulse(right_arr, fps, [], "right cheek")
        forehead_sig = self._roi_to_pulse(forehead_arr, fps, [], "forehead")
        
        # Combine ROI signals
        try:
            combined_raw = combine_roi_signals(
                [left_sig, right_sig, forehead_sig], weights=self.roi_weights
            )
        except ValueError:
            return WindowResult(
                window_index=window_index,
                start_frame=start_idx,
                end_frame=end_idx - 1,
                start_time_sec=start_idx / fps,
                end_time_sec=(end_idx - 1) / fps,
                n_frames_total=n_win_total,
                n_frames_usable=n_win_usable,
                features=None,
                combined_signal=None,
                quality_score_mean=quality_score_mean,
                warnings=["No valid ROI signals to combine"],
                features_valid=False,
            )
        
        # Clean signal
        combined_clean = clean_signal(combined_raw, fs=fps, low_hz=self.low_hz, high_hz=self.high_hz)
        
        left_clean = clean_signal(left_sig, fs=fps, low_hz=self.low_hz, high_hz=self.high_hz) if left_sig is not None else None
        right_clean = clean_signal(right_sig, fs=fps, low_hz=self.low_hz, high_hz=self.high_hz) if right_sig is not None else None
        forehead_clean = clean_signal(forehead_sig, fs=fps, low_hz=self.low_hz, high_hz=self.high_hz) if forehead_sig is not None else None
        
        # Compute features
        try:
            feats = compute_features(
                combined_signal=combined_clean,
                fs=fps,
                left_cheek_signal=left_clean,
                right_cheek_signal=right_clean,
                forehead_signal=forehead_clean,
                low_hz=self.low_hz,
                high_hz=self.high_hz,
            )
        except Exception as e:
            return WindowResult(
                window_index=window_index,
                start_frame=start_idx,
                end_frame=end_idx - 1,
                start_time_sec=start_idx / fps,
                end_time_sec=(end_idx - 1) / fps,
                n_frames_total=n_win_total,
                n_frames_usable=n_win_usable,
                features=None,
                combined_signal=combined_clean,
                quality_score_mean=quality_score_mean,
                warnings=[f"Feature computation failed: {e}"],
                features_valid=False,
            )
        
        # Check feature quality
        raw_nan_count = getattr(feats, "_raw_nan_count", 0)
        if raw_nan_count >= 2 or feats.signal_quality_index < self.min_sqi:
            return WindowResult(
                window_index=window_index,
                start_frame=start_idx,
                end_frame=end_idx - 1,
                start_time_sec=start_idx / fps,
                end_time_sec=(end_idx - 1) / fps,
                n_frames_total=n_win_total,
                n_frames_usable=n_win_usable,
                features=None,
                combined_signal=combined_clean,
                quality_score_mean=quality_score_mean,
                warnings=[f"Degenerate signal: nan={raw_nan_count}, SQI={feats.signal_quality_index:.3f}"],
                features_valid=False,
            )
        
        return WindowResult(
            window_index=window_index,
            start_frame=start_idx,
            end_frame=end_idx - 1,
            start_time_sec=start_idx / fps,
            end_time_sec=(end_idx - 1) / fps,
            n_frames_total=n_win_total,
            n_frames_usable=n_win_usable,
            features=feats,
            combined_signal=combined_clean,
            quality_score_mean=quality_score_mean,
            warnings=[],
            features_valid=True,
        )

    def _aggregate_window_features(
        self,
        window_results: List[WindowResult],
        fps: float,
    ) -> tuple[Optional[RPPGFeatures], dict]:
        """
        Aggregate window-level features into video-level statistics.
        
        Computes mean, std, min, max, range, and coefficient of variation
        for each feature across all valid windows.
        """
        valid_windows = [w for w in window_results if w.features_valid and w.features is not None]
        if not valid_windows:
            return None, {}
        
        # Collect feature vectors
        feature_vectors = np.array([w.to_feature_vector() for w in valid_windows])
        feature_names = RPPGFeatures.feature_names()
        
        # Compute statistics
        stats = {}
        for i, name in enumerate(feature_names):
            vals = feature_vectors[:, i]
            valid_vals = vals[np.isfinite(vals)]
            if len(valid_vals) > 0:
                stats[name] = {
                    "mean": float(np.mean(valid_vals)),
                    "std": float(np.std(valid_vals)),
                    "min": float(np.min(valid_vals)),
                    "max": float(np.max(valid_vals)),
                    "range": float(np.max(valid_vals) - np.min(valid_vals)),
                    "cv": float(np.std(valid_vals) / np.mean(valid_vals)) if np.mean(valid_vals) != 0 else float('inf'),
                    "n_valid": len(valid_vals),
                    "n_total": len(vals),
                }
        
        # Create aggregated features using mean of each feature
        mean_vector = np.mean(feature_vectors, axis=0)
        # Replace NaN with feature means where possible
        for i in range(len(mean_vector)):
            if not np.isfinite(mean_vector[i]):
                valid_vals = feature_vectors[:, i][np.isfinite(feature_vectors[:, i])]
                if len(valid_vals) > 0:
                    mean_vector[i] = np.mean(valid_vals)
        
        # Create RPPGFeatures from mean vector
        try:
            aggregated = RPPGFeatures(*mean_vector)
        except Exception:
            # If construction fails, create with defaults
            aggregated = None
        
        return aggregated, stats

    def process_frames(
        self,
        frames_dir: str,
        metadata_path: Optional[str] = None,
        fps: float = 10.0,
    ) -> RPPGResult:
        """
        Process accepted frames produced by the stage-1 frame layer.

        The frames already passed the frame stage's quality gate (YOLO face
        detection + blur/brightness checks), so no quality re-gating happens
        here; MediaPipe still runs per frame to build the ROI traces, and
        frames where MediaPipe finds no face are treated as unusable
        (interpolated over downstream).
        """
        frames_root = Path(frames_dir)
        frame_paths = sorted(frames_root.glob("*.jpg"), key=_frame_sort_key)
        if not frame_paths:
            raise IOError(f"No stage-1 frames found in: {frames_dir}")

        source_ids: dict[int, int] = {}
        if metadata_path is not None and Path(metadata_path).exists():
            for line in Path(metadata_path).read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                frame_id = record.get("frame_id", -1)
                source_id = record.get("source_frame_id", -1)
                if isinstance(frame_id, int) and isinstance(source_id, int):
                    source_ids[frame_id] = source_id

        warnings: List[str] = []
        quality_log: List[FrameQuality] = []

        left_trace, right_trace, forehead_trace = [], [], []

        with FaceROIExtractor() as extractor:
            for path in frame_paths:
                frame = cv2.imread(str(path))
                if frame is None:
                    warnings.append(f"Unreadable stage-1 frame skipped: {path.name}")
                    continue
                frame_id = _frame_sort_key(path)
                source_idx = source_ids.get(frame_id, frame_id)
                face = extractor.detect(frame, source_idx)
                # Stage 1 already gated blur/brightness, so skip that compute here.
                q = self._assess_frame(frame, source_idx, face.found, face=face, compute_quality=False)
                quality_log.append(q)

                if face.found:
                    rois = extractor.extract_rois(frame, face)
                    left_rgb = extractor.mean_rgb(frame, rois.left_cheek)
                    right_rgb = extractor.mean_rgb(frame, rois.right_cheek)
                    forehead_rgb = extractor.mean_rgb(frame, rois.forehead)
                    left_trace.append(left_rgb)
                    right_trace.append(right_rgb)
                    forehead_trace.append(forehead_rgb)
                    
                    # Phase 2: Re-assess with ROI and RGB data for quality weighting
                    if self.use_quality_weighting:
                        q_full = self._assess_frame(
                            frame, source_idx, face.found, face=face,
                            left_roi=rois.left_cheek, right_roi=rois.right_cheek, forehead_roi=rois.forehead,
                            left_rgb=left_rgb, right_rgb=right_rgb, forehead_rgb=forehead_rgb,
                            compute_quality=False
                        )
                        quality_log[-1] = q_full  # Replace with full assessment
                else:
                    left_trace.append(None)
                    right_trace.append(None)
                    forehead_trace.append(None)

        return self._finalize(left_trace, right_trace, forehead_trace, fps, warnings, quality_log)

    # -- shared signal/feature tail ---------------------------------------

    def _finalize(
        self,
        left_trace: List[Optional[np.ndarray]],
        right_trace: List[Optional[np.ndarray]],
        forehead_trace: List[Optional[np.ndarray]],
        fps: float,
        warnings: List[str],
        quality_log: List[FrameQuality],
    ) -> RPPGResult:
        n_total = len(quality_log)
        n_usable = sum(1 for q in quality_log if q.is_usable)

        # Phase 2: collect quality weights for each frame
        quality_weights = np.array([q.quality_score for q in quality_log], dtype=np.float64)
        
        # Frame utilization metrics
        if self.use_quality_weighting:
            avg_quality = quality_weights.mean() if len(quality_weights) > 0 else 0.0
            effective_frames = quality_weights.sum()
            warnings.append(
                f"Frame utilization: total={n_total}, binary_usable={n_usable}, "
                f"avg_quality_weight={avg_quality:.3f}, effective_frames={effective_frames:.1f}"
            )

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

        # Phase 2: use weighted trace array conversion
        if self.use_quality_weighting:
            left_arr = self._traces_to_array_weighted(left_trace, quality_weights)
            right_arr = self._traces_to_array_weighted(right_trace, quality_weights)
            forehead_arr = self._traces_to_array_weighted(forehead_trace, quality_weights)
        else:
            left_arr = self._traces_to_array(left_trace)
            right_arr = self._traces_to_array(right_trace)
            forehead_arr = self._traces_to_array(forehead_trace)

        left_sig = self._roi_to_pulse(left_arr, fps, warnings, "left cheek")
        right_sig = self._roi_to_pulse(right_arr, fps, warnings, "right cheek")
        forehead_sig = self._roi_to_pulse(forehead_arr, fps, warnings, "forehead")

        try:
            combined_raw = combine_roi_signals(
                [left_sig, right_sig, forehead_sig], weights=self.roi_weights
            )
        except ValueError:
            # Every ROI trace collapsed (e.g. sparse usable frames in a long
            # clip pushed each ROI below its 30% valid-sample floor). Treat
            # as no-features (INCONCLUSIVE) rather than crashing inference.
            warnings.append(
                "No valid ROI signals to combine; treating as no-features (INCONCLUSIVE)."
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
        combined_clean = clean_signal(combined_raw, fs=fps, low_hz=self.low_hz, high_hz=self.high_hz)

        left_clean = clean_signal(left_sig, fs=fps, low_hz=self.low_hz, high_hz=self.high_hz) if left_sig is not None else None
        right_clean = clean_signal(right_sig, fs=fps, low_hz=self.low_hz, high_hz=self.high_hz) if right_sig is not None else None
        forehead_clean = clean_signal(forehead_sig, fs=fps, low_hz=self.low_hz, high_hz=self.high_hz) if forehead_sig is not None else None

        # Diagnostic logging: signal quality statistics
        if combined_clean is not None:
            # combined_raw is 1D pulse signal, not 2D RGB - check for NaN differently
            if combined_raw is not None and combined_raw.ndim == 1:
                valid_ratio = 1.0 - np.isnan(combined_raw).mean()
            elif combined_raw is not None and combined_raw.ndim == 2:
                valid_ratio = 1.0 - np.isnan(combined_raw).any(axis=1).mean()
            else:
                valid_ratio = 0.0
            warnings.append(
                f"Signal diagnostics: fps={fps:.1f}, n_total={n_total}, n_usable={n_usable}, "
                f"valid_ratio={valid_ratio:.2f}, combined_signal_len={len(combined_clean)}, "
                f"left_valid={left_sig is not None}, right_valid={right_sig is not None}, "
                f"forehead_valid={forehead_sig is not None}"
            )

        feats = compute_features(
            combined_signal=combined_clean,
            fs=fps,
            left_cheek_signal=left_clean,
            right_cheek_signal=right_clean,
            forehead_signal=forehead_clean,
            low_hz=self.low_hz,
            high_hz=self.high_hz,
        )

        # Diagnostic logging: feature quality
        raw_nan_count = getattr(feats, "_raw_nan_count", 0)
        warnings.append(
            f"Feature diagnostics: raw_nan_count={raw_nan_count}, "
            f"SQI={feats.signal_quality_index:.3f}, HR={feats.heart_rate_bpm:.1f}BPM, "
            f"SNR={feats.snr_db:.1f}dB, PRV={feats.prv_std_ms:.1f}ms, "
            f"Entropy={feats.spectral_entropy:.3f}, MAD={feats.mad:.3f}"
        )

        raw_nan_count = getattr(feats, "_raw_nan_count", 0)
        if raw_nan_count >= 2 or feats.signal_quality_index < self.min_sqi:
            warnings.append(
                f"Degenerate rPPG signal (non-finite features: {raw_nan_count}, "
                f"SQI: {feats.signal_quality_index:.2f} < {self.min_sqi:.2f}); "
                "treating as no-features (INCONCLUSIVE)."
            )
            return RPPGResult(
                fps=fps,
                n_frames_total=n_total,
                n_frames_usable=n_usable,
                features=None,
                combined_signal=combined_clean,
                left_cheek_signal=left_clean,
                right_cheek_signal=right_clean,
                forehead_signal=forehead_clean,
                quality_log=quality_log,
                warnings=warnings,
            )

        raw_nan_count = getattr(feats, "_raw_nan_count", 0)
        if raw_nan_count >= 2 or feats.signal_quality_index < self.min_sqi:
            warnings.append(
                f"Degenerate rPPG signal (non-finite features: {raw_nan_count}, "
                f"SQI: {feats.signal_quality_index:.2f} < {self.min_sqi:.2f}); "
                "treating as no-features (INCONCLUSIVE)."
            )
            return RPPGResult(
                fps=fps,
                n_frames_total=n_total,
                n_frames_usable=n_usable,
                features=None,
                combined_signal=combined_clean,
                left_cheek_signal=left_clean,
                right_cheek_signal=right_clean,
                forehead_signal=forehead_clean,
                quality_log=quality_log,
                warnings=warnings,
            )

        # Phase 5: Compute Physiological Quality Score (PQS)
        pqs_result = compute_pqs(
            RPPGResult(
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
        )
        warnings.append(
            f"PQS: {pqs_result.pqs:.3f} ({pqs_result.quality_tier}), "
            f"components: SNR={pqs_result.components.snr_quality:.2f}, "
            f"frames={pqs_result.components.frame_utilization:.2f}, "
            f"ROI={pqs_result.components.roi_validity:.2f}, "
            f"crossROI={pqs_result.components.cross_roi_consistency:.2f}, "
            f"freq={pqs_result.components.frequency_stability:.2f}, "
            f"amp={pqs_result.components.signal_amplitude:.2f}, "
            f"temp={pqs_result.components.temporal_consistency:.2f}"
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
            pqs=pqs_result,
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

    def _traces_to_array_weighted(
        self,
        trace_list: List[Optional[np.ndarray]],
        quality_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Convert traces to array with quality-weighted interpolation.

        Instead of simple NaN filling, this method uses quality weights
        to perform weighted interpolation of missing frames. Frames with
        higher quality weights contribute more to the interpolated values.
        """
        n_frames = len(trace_list)
        arr = np.full((n_frames, 3), np.nan, dtype=np.float64)
        
        # Fill in available traces
        valid_indices = []
        valid_values = []
        for i, v in enumerate(trace_list):
            if v is not None:
                arr[i] = v
                valid_indices.append(i)
                valid_values.append(v)
        
        if len(valid_indices) < 2:
            # Not enough valid frames for interpolation
            return arr
        
        valid_indices = np.array(valid_indices)
        valid_values = np.array(valid_values)  # (n_valid, 3)
        valid_weights = quality_weights[valid_indices]
        
        # Weighted interpolation per channel
        for c in range(3):
            col = arr[:, c]
            nans = np.isnan(col)
            if nans.any() and not nans.all():
                # For each NaN position, do weighted interpolation
                nan_indices = np.where(nans)[0]
                for nan_idx in nan_indices:
                    # Find valid frames before and after
                    before = valid_indices[valid_indices < nan_idx]
                    after = valid_indices[valid_indices > nan_idx]
                    
                    if len(before) > 0 and len(after) > 0:
                        # Linear interpolation with weights
                        idx_before = before[-1]
                        idx_after = after[0]
                        val_before = valid_values[np.where(valid_indices == idx_before)[0][0], c]
                        val_after = valid_values[np.where(valid_indices == idx_after)[0][0], c]
                        w_before = valid_weights[np.where(valid_indices == idx_before)[0][0]]
                        w_after = valid_weights[np.where(valid_indices == idx_after)[0][0]]
                        
                        # Weighted linear interpolation
                        t = (nan_idx - idx_before) / (idx_after - idx_before)
                        col[nan_idx] = (1 - t) * val_before * w_before + t * val_after * w_after
                        col[nan_idx] /= (w_before + w_after)
                    elif len(before) > 0:
                        # Extrapolate from before
                        idx_before = before[-1]
                        val_before = valid_values[np.where(valid_indices == idx_before)[0][0], c]
                        col[nan_idx] = val_before
                    elif len(after) > 0:
                        # Extrapolate from after
                        idx_after = after[0]
                        val_after = valid_values[np.where(valid_indices == idx_after)[0][0], c]
                        col[nan_idx] = val_after
        
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
