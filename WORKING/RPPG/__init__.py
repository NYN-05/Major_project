"""
rPPG Pipeline for Low-Resolution KYC Deepfake Detection
==========================================================

A production-grade remote photoplethysmography (rPPG) signal extraction
and feature generation pipeline, designed to feed physiological
liveness features into a downstream (hybrid quantum-classical)
classifier.

Modules
-------
face_roi            : Face landmark detection, ROI (cheeks/forehead)
                         extraction, skin masking, landmark smoothing.
preprocessing        : Detrending, bandpass filtering, normalization.
signal_extraction    : CHROM and POS rPPG signal reconstruction methods.
features             : Heart-rate, SNR, PRV, inter-region correlation,
                        spectral entropy, MAD, and other physiological
                        features.
pqs                  : Physiological Quality Score (PQS) computation.
quality_analysis     : Quality indicators and compression/quality-robustness analysis.
pipeline             : End-to-end orchestration: video -> feature vector.
"""

from .pipeline import RPPGPipeline, RPPGResult
from .pqs import compute_pqs, compute_pqs_simple, PQSResult, PQSComponents, get_quality_tier
from .quality_analysis import (
    FrameQualityIndicators,
    VideoQualityIndicators,
    QualityGroupMetrics,
    compute_frame_quality_indicators,
    aggregate_video_quality,
    assign_quality_group,
    group_videos_by_quality,
    compute_group_metrics,
    analyze_quality_robustness,
    generate_quality_report,
)

__all__ = [
    "RPPGPipeline", 
    "RPPGResult", 
    "compute_pqs", 
    "compute_pqs_simple", 
    "PQSResult", 
    "PQSComponents", 
    "get_quality_tier",
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
__version__ = "1.0.0"
