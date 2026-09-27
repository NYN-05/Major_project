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
pipeline             : End-to-end orchestration: video -> feature vector.
"""

from .pipeline import RPPGPipeline, RPPGResult
from .pqs import compute_pqs, compute_pqs_simple, PQSResult, PQSComponents, get_quality_tier

__all__ = ["RPPGPipeline", "RPPGResult", "compute_pqs", "compute_pqs_simple", "PQSResult", "PQSComponents", "get_quality_tier"]
__version__ = "1.0.0"
