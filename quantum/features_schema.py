from dataclasses import dataclass, field

LABEL_REAL = 1
LABEL_FAKE = 0

FEATURE_NAMES = [
    "temporal_consistency",
    "inter_region_agreement",
    "signal_stability",
    "amplitude_reliability",
    "rhythm_quality",
    "sync_behavior",
    "frame_quality_score",
    "roi_stability",
    "temporal_coverage_ratio",
]

PRIMARY_FEATURES = FEATURE_NAMES[:6]
AUXILIARY_FEATURES = FEATURE_NAMES[6:]

PRIMARY_INDEX = list(range(6))
AUXILIARY_INDEX = list(range(6, len(FEATURE_NAMES)))

MIN_VALUE = 0.0
MAX_VALUE = 1.0

FEATURE_MEANINGS = {
    "temporal_consistency": "Coherence of the recovered pulse signal across the sampled sequence",
    "inter_region_agreement": "Agreement between left cheek, right cheek, and forehead signals",
    "signal_stability": "Waveform stability of the rPPG signal over time",
    "amplitude_reliability": "Strength and reliability of the pulse amplitude",
    "rhythm_quality": "Periodicity and biological plausibility of the rhythm",
    "sync_behavior": "Synchronization behavior across facial regions",
    "frame_quality_score": "Mean frame quality score from the extraction pipeline",
    "roi_stability": "ROI consistency across the frame sequence",
    "temporal_coverage_ratio": "Fraction of sampled frames accepted after quality filtering",
}


@dataclass(frozen=True)
class FeatureContract:
    names: tuple = field(default_factory=lambda: FEATURE_NAMES)
    dim: int = len(FEATURE_NAMES)
    min_value: float = MIN_VALUE
    max_value: float = MAX_VALUE
    real_label: int = LABEL_REAL
    fake_label: int = LABEL_FAKE
