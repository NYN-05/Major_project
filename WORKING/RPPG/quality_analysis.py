"""
quality_analysis.py
===================
Quality analysis module for compression and quality-robustness analysis (Phase 10).

Extracts quality indicators from video frames and rPPG signals,
performs quality grouping, and evaluates classification performance
across quality groups.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import json

import cv2
import numpy as np
from scipy import signal
from scipy.stats import kurtosis as sp_kurtosis


# ---------------------------------------------------------------------------
# Quality Indicators Data Classes
# ---------------------------------------------------------------------------

@dataclass
class FrameQualityIndicators:
    """Quality indicators for a single frame."""
    frame_index: int
    face_resolution: Tuple[int, int]  # (width, height) of face bbox
    roi_area: Dict[str, int]  # area in pixels for each ROI
    roi_valid: Dict[str, bool]  # whether each ROI is valid
    blur_score: float  # Laplacian variance
    brightness: float  # mean pixel intensity
    face_confidence: float  # detection confidence
    landmark_stability: float = 0.0  # inter-frame landmark displacement
    roi_pixels: Dict[str, int] = field(default_factory=dict)  # valid pixel count per ROI


@dataclass
class VideoQualityIndicators:
    """Aggregated quality indicators for a video."""
    video_path: str
    n_frames_total: int
    n_frames_with_face: int
    n_frames_usable: int
    frame_rejection_rate: float
    
    # Face/ROI metrics
    mean_face_area: float  # mean face bbox area in pixels
    mean_face_area_ratio: float  # mean face area / frame area
    mean_roi_area: Dict[str, float]  # mean ROI area per region
    roi_validity_rate: Dict[str, float]  # fraction of frames with valid ROI
    
    # Signal quality metrics
    mean_blur_score: float
    mean_brightness: float
    blur_rejection_rate: float
    brightness_rejection_rate: float
    face_missing_rate: float
    
    # Signal-level metrics (computed from rPPG traces)
    mean_snr_db: float = 0.0
    mean_signal_quality_index: float = 0.0
    mean_hr_bpm: float = 0.0
    snr_distribution: List[float] = field(default_factory=list)
    sqi_distribution: List[float] = field(default_factory=list)
    
    # Compression indicators (if available)
    estimated_compression_quality: Optional[float] = None
    estimated_bitrate: Optional[float] = None
    
    # Derived quality group
    quality_group: str = "unknown"  # "high", "medium", "low", "unknown"


@dataclass
class QualityGroupMetrics:
    """Metrics for a quality group."""
    group_name: str
    n_videos: int
    n_samples: int
    
    # rPPG quality metrics
    mean_snr_db: float
    std_snr_db: float
    mean_sqi: float
    std_sqi: float
    mean_hr_bpm: float
    usable_frame_pct: float
    
    # Classification performance
    accuracy: float
    auc_roc: float
    balanced_accuracy: float
    f1: float
    precision: float
    recall: float
    specificity: float
    
    # Coverage
    n_classified: int
    n_total: int
    coverage: float


# ---------------------------------------------------------------------------
# Quality Indicator Extraction
# ---------------------------------------------------------------------------

def compute_frame_quality_indicators(
    frame_bgr: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]],
    roi_masks: Dict[str, Optional[np.ndarray]],
    frame_index: int,
    prev_landmarks: Optional[np.ndarray] = None,
    curr_landmarks: Optional[np.ndarray] = None,
) -> FrameQualityIndicators:
    """
    Compute quality indicators for a single frame.
    
    Args:
        frame_bgr: Input frame in BGR format
        face_bbox: Face bounding box (x, y, w, h)
        roi_masks: Dict mapping ROI name to mask array
        frame_index: Frame index
        prev_landmarks: Previous frame landmarks for stability computation
        curr_landmarks: Current frame landmarks
    
    Returns:
        FrameQualityIndicators object
    """
    h, w = frame_bgr.shape[:2]
    
    # Face resolution
    if face_bbox:
        x, y, w, h = face_bbox
        face_area = w * h
        face_area_ratio = face_area / (w * h)
        face_resolution = (w, h)
    else:
        face_area = 0
        face_area_ratio = 0.0
        face_resolution = (0, 0)
    
    # Blur score (Laplacian variance)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Brightness
    brightness = float(gray.mean())
    
    # ROI areas and validity
    roi_area = {}
    roi_valid = {}
    roi_pixels = {}
    for roi_name, mask in roi_masks.items():
        if mask is not None:
            area = int(cv2.countNonZero(mask))
            roi_area[roi_name] = area
            roi_valid[roi_name] = area > 0
            roi_pixels[roi_name] = area
        else:
            roi_area[roi_name] = 0
            roi_valid[roi_name] = False
            roi_pixels[roi_name] = 0
    
    # Landmark stability (inter-frame displacement)
    landmark_stability = 0.0
    if prev_landmarks is not None and curr_landmarks is not None:
        if len(prev_landmarks) == len(curr_landmarks):
            displacement = np.mean(np.linalg.norm(curr_landmarks - prev_landmarks, axis=1))
            # Normalize by face size if available
            landmark_stability = float(np.exp(-displacement / 10.0))  # exponential decay
    
    return FrameQualityIndicators(
        frame_index=frame_index,
        face_resolution=face_resolution,
        roi_area=roi_area,
        roi_valid=roi_valid,
        blur_score=blur_score,
        brightness=brightness,
        face_confidence=1.0,  # placeholder
        landmark_stability=landmark_stability,
        roi_pixels=roi_pixels,
    )


def aggregate_video_quality(
    frame_indicators: List[FrameQualityIndicators],
    n_frames_total: int,
    rppg_features: Optional[object] = None,
) -> VideoQualityIndicators:
    """
    Aggregate frame-level quality indicators to video-level.
    
    Args:
        frame_indicators: List of FrameQualityIndicators
        n_frames_total: Total number of frames in video
        rppg_features: Optional RPPGFeatures object for signal-level metrics
    
    Returns:
        VideoQualityIndicators object
    """
    if not frame_indicators:
        return VideoQualityIndicators(
            video_path="",
            n_frames_total=n_frames_total,
            n_frames_with_face=0,
            n_frames_usable=0,
            frame_rejection_rate=1.0,
            mean_face_area=0.0,
            mean_face_area_ratio=0.0,
            mean_roi_area={},
            roi_validity_rate={},
            mean_blur_score=0.0,
            mean_brightness=0.0,
            blur_rejection_rate=1.0,
            brightness_rejection_rate=1.0,
            face_missing_rate=1.0,
        )
    
    n_frames_with_face = sum(1 for fi in frame_indicators if any(fi.roi_valid.values()))
    n_frames_usable = sum(1 for fi in frame_indicators if all(fi.roi_valid.values()))
    
    # Face area metrics
    face_areas = [fi.roi_area.get('left_cheek', 0) + fi.roi_area.get('right_cheek', 0) + fi.roi_area.get('forehead', 0) 
                  for fi in frame_indicators]
    mean_face_area = float(np.mean(face_areas)) if face_areas else 0.0
    
    # ROI areas
    roi_names = ['left_cheek', 'right_cheek', 'forehead']
    mean_roi_area = {}
    roi_validity_rate = {}
    for roi_name in roi_names:
        areas = [fi.roi_area.get(roi_name, 0) for fi in frame_indicators if fi.roi_area.get(roi_name, 0) > 0]
        valid_counts = sum(1 for fi in frame_indicators if fi.roi_valid.get(roi_name, False))
        mean_roi_area[roi_name] = float(np.mean(areas)) if areas else 0.0
        roi_validity_rate[roi_name] = valid_counts / len(frame_indicators) if frame_indicators else 0.0
    
    # Blur and brightness
    blur_scores = [fi.blur_score for fi in frame_indicators]
    brightness_scores = [fi.brightness for fi in frame_indicators]
    mean_blur = float(np.mean(blur_scores)) if blur_scores else 0.0
    mean_brightness = float(np.mean(brightness_scores)) if brightness_scores else 0.0
    
    # Rejection rates (using typical thresholds)
    blur_threshold = 5.0
    brightness_min, brightness_max = 20, 240
    blur_rejected = sum(1 for b in blur_scores if b < blur_threshold)
    brightness_rejected = sum(1 for b in brightness_scores if b < brightness_min or b > brightness_max)
    face_missing = sum(1 for fi in frame_indicators if not any(fi.roi_valid.values()))
    
    # Signal-level metrics from rPPG features
    mean_snr_db = 0.0
    mean_sqi = 0.0
    mean_hr = 0.0
    snr_dist = []
    sqi_dist = []
    
    if rppg_features is not None:
        mean_snr_db = getattr(rppg_features, 'snr_db', 0.0)
        mean_sqi = getattr(rppg_features, 'signal_quality_index', 0.0)
        mean_hr = getattr(rppg_features, 'heart_rate_bpm', 0.0)
        if hasattr(rppg_features, 'snr_distribution'):
            snr_dist = rppg_features.snr_distribution
        if hasattr(rppg_features, 'sqi_distribution'):
            sqi_dist = rppg_features.sqi_distribution
    
    # Face area ratio
    frame_area = 1.0  # placeholder - would need frame dimensions
    mean_face_area_ratio = mean_face_area / frame_area if frame_area > 0 else 0.0
    
    return VideoQualityIndicators(
        video_path="",
        n_frames_total=n_frames_total,
        n_frames_with_face=n_frames_with_face,
        n_frames_usable=n_frames_usable,
        frame_rejection_rate=1.0 - (n_frames_usable / n_frames_total) if n_frames_total > 0 else 1.0,
        mean_face_area=mean_face_area,
        mean_face_area_ratio=mean_face_area_ratio,
        mean_roi_area=mean_roi_area,
        roi_validity_rate=roi_validity_rate,
        mean_blur_score=mean_blur,
        mean_brightness=mean_brightness,
        blur_rejection_rate=blur_rejected / len(frame_indicators) if frame_indicators else 0.0,
        brightness_rejection_rate=brightness_rejected / len(frame_indicators) if frame_indicators else 0.0,
        face_missing_rate=face_missing / len(frame_indicators) if frame_indicators else 0.0,
        mean_snr_db=mean_snr_db,
        mean_signal_quality_index=mean_sqi,
        mean_hr_bpm=mean_hr,
        snr_distribution=snr_dist,
        sqi_distribution=sqi_dist,
    )


def assign_quality_group(
    video_quality: VideoQualityIndicators,
    snr_thresholds: Tuple[float, float] = (-5.0, 5.0),
    sqi_thresholds: Tuple[float, float] = (0.2, 0.5),
    usable_frame_thresholds: Tuple[float, float] = (0.3, 0.6),
) -> str:
    """
    Assign quality group based on video quality indicators.
    
    Args:
        video_quality: VideoQualityIndicators object
        snr_thresholds: (low, high) SNR thresholds in dB
        sqi_thresholds: (low, high) SQI thresholds
        usable_frame_thresholds: (low, high) usable frame percentage thresholds
    
    Returns:
        Quality group: "high", "medium", "low", or "unknown"
    """
    if video_quality.n_frames_total == 0:
        return "unknown"
    
    # Score based on multiple criteria
    score = 0
    max_score = 0
    
    # SNR criterion
    if video_quality.mean_snr_db > snr_thresholds[1]:
        score += 3
    elif video_quality.mean_snr_db > snr_thresholds[0]:
        score += 2
    else:
        score += 1
    max_score += 3
    
    # SQI criterion
    if video_quality.mean_signal_quality_index > sqi_thresholds[1]:
        score += 3
    elif video_quality.mean_signal_quality_index > sqi_thresholds[0]:
        score += 2
    else:
        score += 1
    max_score += 3
    
    # Usable frame criterion
    usable_pct = 1.0 - video_quality.frame_rejection_rate
    if usable_pct > usable_frame_thresholds[1]:
        score += 2
    elif usable_pct > usable_frame_thresholds[0]:
        score += 1
    max_score += 2
    
    # ROI validity criterion
    roi_validity = np.mean(list(video_quality.roi_validity_rate.values())) if video_quality.roi_validity_rate else 0
    if roi_validity > 0.8:
        score += 2
    elif roi_validity > 0.5:
        score += 1
    max_score += 2
    
    # Normalize and assign group
    normalized = score / max_score if max_score > 0 else 0
    
    if normalized >= 0.75:
        return "high"
    elif normalized >= 0.5:
        return "medium"
    elif normalized >= 0.25:
        return "low"
    else:
        return "unknown"


# ---------------------------------------------------------------------------
# Quality Grouping and Analysis
# ---------------------------------------------------------------------------

def group_videos_by_quality(
    video_qualities: List[VideoQualityIndicators],
    group_method: str = "score_based",
) -> Dict[str, List[VideoQualityIndicators]]:
    """
    Group videos by quality.
    
    Args:
        video_qualities: List of VideoQualityIndicators
        group_method: "score_based" or "percentile" or "fixed_thresholds"
    
    Returns:
        Dict mapping group name to list of VideoQualityIndicators
    """
    groups = {"high": [], "medium": [], "low": [], "unknown": []}
    
    for vq in video_qualities:
        group = assign_quality_group(vq)
        vq.quality_group = group
        groups[group].append(vq)
    
    return groups


def compute_group_metrics(
    group_videos: List[VideoQualityIndicators],
    feature_set: str = "rppg",
    classification_results: Optional[Dict] = None,
) -> QualityGroupMetrics:
    """
    Compute aggregate metrics for a quality group.
    
    Args:
        group_videos: List of VideoQualityIndicators in the group
        feature_set: Feature set used for classification
        classification_results: Optional dict with classification results per video
    
    Returns:
        QualityGroupMetrics object
    """
    if not group_videos:
        return QualityGroupMetrics(
            group_name="empty",
            n_videos=0,
            n_samples=0,
            mean_snr_db=0.0, std_snr_db=0.0,
            mean_sqi=0.0, std_sqi=0.0,
            mean_hr_bpm=0.0,
            usable_frame_pct=0.0,
            accuracy=0.0, auc_roc=0.0, balanced_accuracy=0.0,
            f1=0.0, precision=0.0, recall=0.0, specificity=0.0,
            n_classified=0, n_total=0, coverage=0.0,
        )
    
    # rPPG quality metrics
    snr_values = [v.mean_snr_db for v in group_videos if np.isfinite(v.mean_snr_db)]
    sqi_values = [v.mean_signal_quality_index for v in group_videos if np.isfinite(v.mean_signal_quality_index)]
    hr_values = [v.mean_hr_bpm for v in group_videos if np.isfinite(v.mean_hr_bpm)]
    usable_pcts = [1.0 - v.frame_rejection_rate for v in group_videos]
    
    # Classification metrics (if available)
    accuracy = 0.0
    auc_roc = 0.0
    balanced_acc = 0.0
    f1 = 0.0
    precision = 0.0
    recall = 0.0
    specificity = 0.0
    n_classified = 0
    n_total = 0
    coverage = 0.0
    
    if classification_results:
        # Aggregate classification results
        # This would require the actual classification results per video
        pass
    
    return QualityGroupMetrics(
        group_name="group",
        n_videos=len(group_videos),
        n_samples=len(group_videos),
        mean_snr_db=float(np.mean(snr_values)) if snr_values else 0.0,
        std_snr_db=float(np.std(snr_values)) if snr_values else 0.0,
        mean_sqi=float(np.mean(sqi_values)) if sqi_values else 0.0,
        std_sqi=float(np.std(sqi_values)) if sqi_values else 0.0,
        mean_hr_bpm=float(np.mean(hr_values)) if hr_values else 0.0,
        usable_frame_pct=float(np.mean(usable_pcts)) if usable_pcts else 0.0,
        accuracy=accuracy,
        auc_roc=auc_roc,
        balanced_accuracy=balanced_acc,
        f1=f1,
        precision=precision,
        recall=recall,
        specificity=specificity,
        n_classified=0,
        n_total=0,
        coverage=0.0,
    )


# ---------------------------------------------------------------------------
# Quality Robustness Analysis
# ---------------------------------------------------------------------------

def analyze_quality_robustness(
    video_qualities: List[VideoQualityIndicators],
    classification_results: Dict[str, Dict],
    feature_set: str = "rppg",
) -> Dict:
    """
    Perform comprehensive quality robustness analysis.
    
    Args:
        video_qualities: List of VideoQualityIndicators for all videos
        classification_results: Dict mapping video_path to classification results
        feature_set: Feature set used for classification
    
    Returns:
        Dict with analysis results
    """
    # Group by quality
    groups = group_videos_by_quality(video_qualities)
    
    # Compute metrics per group
    group_metrics = {}
    for group_name, group_videos in groups.items():
        if group_videos:
            metrics = compute_group_metrics(group_videos, feature_set)
            metrics.group_name = group_name
            group_metrics[group_name] = metrics
    
    # Analyze quality trends
    # Correlation between quality indicators and classification performance
    quality_indicators = []
    performances = []
    
    for vq in video_qualities:
        vid = vq.video_path
        if vid in classification_results:
            quality_indicators.append({
                'snr_db': vq.mean_snr_db,
                'sqi': vq.mean_signal_quality_index,
                'usable_frames': 1.0 - vq.frame_rejection_rate,
                'snr': vq.mean_snr_db,
                'sqi': vq.mean_signal_quality_index,
                'hr': vq.mean_hr_bpm,
                'blur': vq.mean_blur_score,
                'brightness': vq.mean_brightness,
                'face_area_ratio': vq.mean_face_area_ratio,
                'roi_validity': np.mean(list(vq.roi_validity_rate.values())) if vq.roi_validity_rate else 0,
            })
            performances.append({
                'correct': classification_results[vid].get('correct', False),
                'prob_real': classification_results[vid].get('prob_real', 0.5),
                'verdict': classification_results[vid].get('verdict', 'UNCERTAIN'),
            })
    
    # Compute correlations
    correlations = {}
    if quality_indicators and performances:
        qi_array = np.array(quality_indicators)
        perf_array = np.array([p['correct'] for p in performances])
        
        for i, key in enumerate(quality_indicators[0].keys()):
            qi_vals = np.array([q[key] for q in quality_indicators])
            if np.all(np.isfinite(qi_vals)):
                corr = np.corrcoef(qi_vals, perf_array)[0, 1] if len(qi_vals) > 1 else 0
                correlations[key] = float(corr) if np.isfinite(corr) else 0.0
    
    # Identify failure regions
    failure_videos = [
        vq.video_path for vq in video_qualities
        if vq.video_path in classification_results and not classification_results[vq.video_path].get('correct', True)
    ]
    
    failure_quality = [vq for vq in video_qualities if vq.video_path in failure_videos]
    failure_analysis = {}
    if failure_quality:
        failure_analysis = {
            'n_failures': len(failure_quality),
            'mean_snr': float(np.mean([v.mean_snr_db for v in failure_quality])),
            'mean_sqi': float(np.mean([v.mean_signal_quality_index for v in failure_quality])),
            'mean_usable_frames': float(np.mean([1.0 - v.frame_rejection_rate for v in failure_quality])),
            'quality_groups': {},
        }
        for vq in failure_quality:
            failure_analysis['quality_groups'][vq.quality_group] = failure_analysis['quality_groups'].get(vq.quality_group, 0) + 1
    
    return {
        'group_metrics': {k: v.__dict__ for k, v in group_metrics.items()},
        'correlations': correlations,
        'failure_analysis': failure_analysis,
        'n_videos_total': len(video_qualities),
        'n_groups': len([g for g in groups.values() if g]),
    }


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_quality_report(
    analysis_results: Dict,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a human-readable quality robustness report.
    
    Args:
        analysis_results: Results from analyze_quality_robustness
        output_path: Optional path to save report
    
    Returns:
        Report as string
    """
    lines = [
        "=" * 70,
        "PHASE 10: COMPRESSION AND QUALITY-ROBUSTNESS ANALYSIS REPORT",
        "=" * 70,
        "",
        "Total videos analyzed: {}".format(analysis_results.get('n_videos_total', 0)),
        "Quality groups identified: {}".format(analysis_results.get('n_groups', 0)),
        "",
        "GROUP METRICS:",
        "-" * 40,
    ]
    
    for group_name, metrics in analysis_results.get('group_metrics', {}).items():
        n_videos = metrics.get('n_videos', 0)
        mean_snr = metrics.get('mean_snr_db', 0)
        std_snr = metrics.get('std_snr_db', 0)
        mean_sqi = metrics.get('mean_sqi', 0)
        std_sqi = metrics.get('std_sqi', 0)
        mean_hr = metrics.get('mean_hr_bpm', 0)
        usable_pct = metrics.get('usable_frame_pct', 0)
        accuracy = metrics.get('accuracy', 0)
        auc_roc = metrics.get('auc_roc', 0)
        balanced_acc = metrics.get('balanced_accuracy', 0)
        f1 = metrics.get('f1', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        specificity = metrics.get('specificity', 0)
        coverage = metrics.get('coverage', 0)
        n_classified = metrics.get('n_classified', 0)
        n_total = metrics.get('n_total', 0)
        
        lines.extend([
            "\n{} QUALITY GROUP ({} videos):".format(group_name.upper(), n_videos),
            "  rPPG Quality:",
            "    Mean SNR: {:.2f} dB (std: {:.2f})".format(mean_snr, std_snr),
            "    Mean SQI: {:.3f} (std: {:.3f})".format(mean_sqi, std_sqi),
            "    Mean HR: {:.1f} BPM".format(mean_hr),
            "    Usable frames: {:.1f}%".format(usable_pct * 100),
            "  Classification Performance:",
            "    Accuracy: {:.4f}".format(accuracy),
            "    AUC-ROC: {:.4f}".format(auc_roc),
            "    Balanced Accuracy: {:.4f}".format(balanced_acc),
            "    F1: {:.4f}".format(f1),
            "    Precision: {:.4f}".format(precision),
            "    Recall: {:.4f}".format(recall),
            "    Specificity: {:.4f}".format(specificity),
            "  Coverage: {:.2%} ({}/{})".format(coverage, n_classified, n_total),
        ])
    
    # Correlations
    lines.extend([
        "",
        "QUALITY-PERFORMANCE CORRELATIONS:",
        "-" * 40,
    ])
    for indicator, corr in analysis_results.get('correlations', {}).items():
        lines.append("  {}: r = {:.4f}".format(indicator, corr))
    
    # Failure analysis
    fa = analysis_results.get('failure_analysis', {})
    if fa:
        n_failures = fa.get('n_failures', 0)
        mean_snr = fa.get('mean_snr', 0)
        mean_sqi = fa.get('mean_sqi', 0)
        mean_usable = fa.get('mean_usable_frames', 0)
        
        lines.extend([
            "",
            "FAILURE ANALYSIS:",
            "-" * 40,
            "  Failed videos: {}".format(n_failures),
            "  Mean SNR: {:.2f} dB".format(mean_snr),
            "  Mean SQI: {:.3f}".format(mean_sqi),
            "  Mean usable frames: {:.1%}".format(mean_usable),
            "  Failures by quality group:",
        ])
        for group, count in fa.get('quality_groups', {}).items():
            lines.append("    {}: {}".format(group, count))
    
    lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ])
    
    report = "\n".join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
    
    return report


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    "FrameQualityIndicators",
    "VideoQualityIndicators",
    "QualityGroupMetrics",
    "FrameQualityIndicators",
    "VideoQualityIndicators",
    "QualityGroupMetrics",
    "compute_frame_quality_indicators",
    "aggregate_video_quality",
    "assign_quality_group",
    "group_videos_by_quality",
    "compute_group_metrics",
    "analyze_quality_robustness",
    "generate_quality_report",
]