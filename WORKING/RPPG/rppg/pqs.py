"""
pqs.py
=======
Physiological Quality Score (PQS) computation for rPPG signals.

Estimates the reliability of physiological evidence from rPPG analysis,
independent of the deepfake classification decision.

The PQS combines multiple quality indicators into a single score in [0, 1],
where higher values indicate more reliable physiological evidence.

Key principle: PQS measures evidence reliability, NOT deepfake probability.
A genuine video can have low PQS due to compression, lighting, motion, etc.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np


@dataclass
class PQSComponents:
    """Individual quality components before aggregation."""
    snr_quality: float = 0.0           # Based on signal-to-noise ratio
    frame_utilization: float = 0.0     # Based on usable frames / total frames
    roi_validity: float = 0.0          # Based on ROI validity scores
    cross_roi_consistency: float = 0.0 # Based on cross-ROI correlation/coherence
    frequency_stability: float = 0.0   # Based on frequency agreement / HR stability
    signal_amplitude: float = 0.0      # Based on signal amplitude / MAD
    temporal_consistency: float = 0.0  # Based on windowed analysis (Phase 3)

    def to_dict(self) -> Dict[str, float]:
        return {
            "snr_quality": self.snr_quality,
            "frame_utilization": self.frame_utilization,
            "roi_validity": self.roi_validity,
            "cross_roi_consistency": self.cross_roi_consistency,
            "frequency_stability": self.frequency_stability,
            "signal_amplitude": self.signal_amplitude,
            "temporal_consistency": self.temporal_consistency,
        }

    def to_array(self) -> np.ndarray:
        return np.array([
            self.snr_quality,
            self.frame_utilization,
            self.roi_validity,
            self.cross_roi_consistency,
            self.frequency_stability,
            self.signal_amplitude,
            self.temporal_consistency,
        ], dtype=np.float64)


@dataclass
class PQSResult:
    """Complete PQS result with components and final score."""
    pqs: float                          # Final PQS in [0, 1]
    components: PQSComponents           # Individual quality components
    weights: np.ndarray                 # Weights used for aggregation
    quality_tier: str                   # "HIGH", "MEDIUM", "LOW"

    def to_dict(self) -> Dict:
        return {
            "pqs": self.pqs,
            "components": self.components.to_dict(),
            "weights": self.weights.tolist(),
            "quality_tier": self.quality_tier,
        }


# Default weights for PQS components (can be tuned via validation)
DEFAULT_PQS_WEIGHTS = np.array([
    0.20,  # snr_quality
    0.15,  # frame_utilization
    0.15,  # roi_validity
    0.20,  # cross_roi_consistency
    0.10,  # frequency_stability
    0.10,  # signal_amplitude
    0.10,  # temporal_consistency
], dtype=np.float64)

# Quality tier thresholds
PQS_HIGH_THRESHOLD = 0.7
PQS_LOW_THRESHOLD = 0.3


def _normalize_snr(snr_db: float) -> float:
    """
    Normalize SNR to [0, 1] using sigmoid-like mapping.
    
    SNR < 0 dB (noise dominates) -> low quality
    SNR 0-10 dB -> medium quality  
    SNR > 10 dB -> high quality
    """
    if not np.isfinite(snr_db):
        return 0.0
    # Sigmoid centered at 5 dB with slope 0.4
    return float(1.0 / (1.0 + np.exp(-0.4 * (snr_db - 5.0))))


def _normalize_frame_utilization(n_usable: int, n_total: int) -> float:
    """Normalize frame utilization ratio to [0, 1]."""
    if n_total <= 0:
        return 0.0
    ratio = n_usable / n_total
    # Linear scaling with minimum at 0.2 (below which quality is poor)
    return float(np.clip((ratio - 0.2) / 0.8, 0.0, 1.0))


def _normalize_roi_validity(quality_log) -> float:
    """
    Normalize ROI validity based on per-frame ROI validity scores.
    """
    if not quality_log:
        return 0.0
    
    roi_scores = [getattr(q, 'roi_validity_score', 0.0) for q in quality_log]
    if not roi_scores:
        # Fallback: use face_found as proxy
        face_found = sum(1 for q in quality_log if getattr(q, 'face_found', False))
        return float(face_found / len(quality_log))
    
    return float(np.mean(roi_scores))


def _normalize_cross_roi_consistency(features) -> float:
    """
    Normalize cross-ROI consistency based on aggregated correlation/coherence.
    """
    if features is None:
        return 0.0
    
    # Use aggregated cross-ROI correlation mean (already in [0, 1] for correlation)
    if hasattr(features, 'cross_roi_corr_mean') and np.isfinite(features.cross_roi_corr_mean):
        # Correlation is already in [-1, 1], map to [0, 1]
        return float(np.clip((features.cross_roi_corr_mean + 1.0) / 2.0, 0.0, 1.0))
    
    # Fallback: use individual correlations
    corrs = [
        getattr(features, 'cheek_forehead_correlation', np.nan),
        getattr(features, 'left_right_cheek_correlation', np.nan),
    ]
    valid_corrs = [c for c in corrs if np.isfinite(c)]
    if valid_corrs:
        mean_corr = np.mean(valid_corrs)
        return float(np.clip((mean_corr + 1.0) / 2.0, 0.0, 1.0))
    
    return 0.0


def _normalize_frequency_stability(features) -> float:
    """
    Normalize frequency stability based on HR half-difference and frequency agreement.
    """
    if features is None:
        return 0.0
    
    # Use frequency agreement if available (Phase 4 feature, already in [0, 1])
    if hasattr(features, 'freq_agreement_lr') and np.isfinite(features.freq_agreement_lr):
        return float(features.freq_agreement_lr)
    
    # Fallback: use HR half-difference
    # HR half-diff of 0 -> perfect stability, >20 BPM -> poor
    if hasattr(features, 'hr_half_diff') and np.isfinite(features.hr_half_diff):
        return float(np.clip(1.0 - features.hr_half_diff / 20.0, 0.0, 1.0))
    
    return 0.5  # Neutral if no info


def _normalize_signal_amplitude(features) -> float:
    """
    Normalize signal amplitude quality based on MAD and signal-to-motion ratio.
    """
    if features is None:
        return 0.0
    
    # Use signal-to-motion ratio (already in dB, normalize like SNR)
    if hasattr(features, 'signal_to_motion_ratio') and np.isfinite(features.signal_to_motion_ratio):
        return _normalize_snr(features.signal_to_motion_ratio)
    
    # Fallback: use MAD (higher MAD = more signal variation = better, up to a point)
    if hasattr(features, 'mad') and np.isfinite(features.mad) and features.mad > 0:
        # Normalize MAD: typical range 0.1-1.0 for physiological signals
        return float(np.clip(features.mad / 1.0, 0.0, 1.0))
    
    return 0.5


def _normalize_temporal_consistency(result) -> float:
    """
    Normalize temporal consistency based on Phase 3 windowed analysis.
    """
    if not hasattr(result, 'window_feature_stats') or not result.window_feature_stats:
        # No window analysis available
        return 0.5  # Neutral
    
    # Check coefficient of variation for key features across windows
    cv_values = []
    for feat_name, stats in result.window_feature_stats.items():
        if 'cv' in stats and np.isfinite(stats['cv']):
            cv_values.append(stats['cv'])
    
    if not cv_values:
        return 0.5
    
    # Lower CV = higher temporal consistency
    mean_cv = np.mean(cv_values)
    return float(np.clip(1.0 - mean_cv / 0.5, 0.0, 1.0))  # CV > 0.5 is poor


def compute_pqs(
    result,
    weights: Optional[np.ndarray] = None,
) -> 'PQSResult':
    """
    Compute Physiological Quality Score (PQS) for an rPPG analysis result.
    
    Args:
        result: RPPGResult containing features, quality_log, and metadata
        weights: Optional custom weights for the 7 components (default: equal-ish)
        
    Returns:
        PQSResult with final PQS, component scores, weights, and quality tier
    """
    if weights is None:
        weights = DEFAULT_PQS_WEIGHTS.copy()
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != 7:
            raise ValueError("weights must have length 7")
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    # Extract components
    features = getattr(result, 'features', None)
    
    snr_q = _normalize_snr(features.snr_db) if features and np.isfinite(features.snr_db) else 0.0
    frame_q = _normalize_frame_utilization(getattr(result, 'n_frames_usable', 0), getattr(result, 'n_frames_total', 0))
    roi_q = _normalize_roi_validity(getattr(result, 'quality_log', []))
    cross_roi_q = _normalize_cross_roi_consistency(features)
    freq_q = _normalize_frequency_stability(features)
    amp_q = _normalize_signal_amplitude(features)
    temp_q = _normalize_temporal_consistency(result)
    
    components = PQSComponents(
        snr_quality=snr_q,
        frame_utilization=frame_q,
        roi_validity=roi_q,
        cross_roi_consistency=cross_roi_q,
        frequency_stability=freq_q,
        signal_amplitude=amp_q,
        temporal_consistency=temp_q,
    )
    
    # Weighted aggregation
    component_array = components.to_array()
    pqs = float(np.dot(component_array, weights))
    pqs = np.clip(pqs, 0.0, 1.0)
    
    # Determine quality tier
    if pqs >= PQS_HIGH_THRESHOLD:
        tier = "HIGH"
    elif pqs <= PQS_LOW_THRESHOLD:
        tier = "LOW"
    else:
        tier = "MEDIUM"
    
    return PQSResult(
        pqs=pqs,
        components=components,
        weights=weights,
        quality_tier=tier,
    )


def compute_pqs_simple(
    snr_db: float,
    n_usable: int,
    n_total: int,
    cross_roi_corr_mean: float = 0.5,
    freq_agreement_lr: float = 0.5,
    has_window_analysis: bool = False,
) -> float:
    """
    Simplified PQS computation using only core metrics.
    
    Useful when full RPPGResult is not available (e.g., in quantum pipeline).
    """
    snr_q = _normalize_snr(snr_db)
    frame_q = _normalize_frame_utilization(n_usable, n_total)
    roi_q = 0.5  # Default neutral
    cross_roi_q = np.clip((cross_roi_corr_mean + 1.0) / 2.0, 0.0, 1.0) if np.isfinite(cross_roi_corr_mean) else 0.5
    freq_q = freq_agreement_lr if np.isfinite(freq_agreement_lr) else 0.5
    amp_q = 0.5  # Default
    temp_q = 0.5 if has_window_analysis else 0.5
    
    weights = DEFAULT_PQS_WEIGHTS
    component_array = np.array([snr_q, frame_q, roi_q, cross_roi_q, freq_q, amp_q, temp_q])
    pqs = float(np.dot(component_array, weights))
    return np.clip(pqs, 0.0, 1.0)


def get_quality_tier(pqs: float) -> str:
    """Get quality tier label from PQS value."""
    if pqs >= PQS_HIGH_THRESHOLD:
        return "HIGH"
    elif pqs <= PQS_LOW_THRESHOLD:
        return "LOW"
    return "MEDIUM"