"""
explain.py
==========
Explainable output module for Phase 11: Separate Physiological Signal Quality From Deepfake Evidence.

Produces human-readable, structured explanations of the classification decision
by separating signal reliability from classification evidence.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path
import json

from quantum.config import DecisionConfig

# Conditional import for PQS (optional dependency)
try:
    from rppg.pqs import PQSResult, PQSComponents
except ImportError:
    # For environments where rppg is not in path
    PQSResult = None
    PQSComponents = None


class Verdict(Enum):
    """Classification verdict."""
    REAL = "REAL"
    FAKE = "FAKE"


class ReliabilityLevel(Enum):
    """Reliability level for evidence."""
    HIGH = "High reliability"
    MODERATE = "Moderate reliability"
    LOW = "Low reliability"
    UNKNOWN = "Unknown reliability"


@dataclass
class PhysiologicalEvidence:
    """Physiological evidence from rPPG analysis."""
    verdict: Verdict
    probability_real: float
    confidence: float
    pqs_result: Optional['PQSResult'] = None
    signal_quality_index: float = 0.0
    heart_rate_bpm: float = 0.0
    snr_db: float = 0.0
    cross_roi_consistency: float = 0.0
    temporal_consistency: float = 0.0
    reliability_level: ReliabilityLevel = ReliabilityLevel.UNKNOWN
    insufficient_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "verdict": self.verdict.value,
            "probability_real": self.probability_real,
            "confidence": self.confidence,
            "reliability": self.reliability_level.value,
        }
        if self.pqs_result:
            result["pqs"] = {
                "score": self.pqs_result.pqs,
                "tier": self.pqs_result.quality_tier,
                "components": self.pqs_result.components.to_dict(),
            }
        if self.insufficient_reason:
            result["insufficient_reason"] = self.insufficient_reason
        return result


@dataclass
class VisualEvidence:
    """Visual evidence from visual analysis."""
    verdict: Optional[Verdict] = None
    probability_fake: float = 0.0
    confidence: float = 0.0
    reliability_level: ReliabilityLevel = ReliabilityLevel.UNKNOWN
    features_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value if self.verdict else None,
            "probability_fake": self.probability_fake,
            "confidence": self.confidence,
            "reliability": self.reliability_level.value,
            "features_used": self.features_used,
        }


@dataclass
class DecisionExplanation:
    """Complete explainable decision output."""
    final_verdict: Verdict
    physiological_evidence: PhysiologicalEvidence
    visual_evidence: Optional[VisualEvidence] = None
    decision_logic: str = ""
    requires_verification: bool = False
    coverage: float = 0.0
    overall_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_verdict": self.final_verdict.value,
            "physiological_evidence": self.physiological_evidence.to_dict(),
            "visual_evidence": self.visual_evidence.to_dict() if self.visual_evidence else None,
            "decision_logic": self.decision_logic,
            "requires_verification": self.requires_verification,
            "coverage": self.coverage,
            "overall_confidence": self.overall_confidence,
        }

    def to_text(self) -> str:
        """Generate human-readable text explanation."""
        lines = [
            f"Prediction: {self.final_verdict.value}",
            f"Confidence: {self.overall_confidence:.2f}",
            "",
            "Physiological Evidence:",
            f"  Reliability: {self.physiological_evidence.reliability_level.value}",
            f"  P(Real): {self.physiological_evidence.probability_real:.2f}",
            f"  Confidence: {self.physiological_evidence.confidence:.2f}",
        ]
        if self.physiological_evidence.pqs_result:
            pqs = self.physiological_evidence.pqs_result
            lines.append(f"  PQS: {pqs.pqs:.3f} ({pqs.quality_tier})")
            comp = pqs.components
            lines.append(f"  SNR Quality: {comp.snr_quality:.2f}")
            lines.append(f"  Frame Utilization: {comp.frame_utilization:.2f}")
            lines.append(f"  ROI Validity: {comp.roi_validity:.2f}")
            lines.append(f"  Cross-ROI Consistency: {comp.cross_roi_consistency:.2f}")
            lines.append(f"  Frequency Stability: {comp.frequency_stability:.2f}")
            lines.append(f"  Signal Amplitude: {comp.signal_amplitude:.2f}")
            lines.append(f"  Temporal Consistency: {comp.temporal_consistency:.2f}")
        
        if self.physiological_evidence.insufficient_reason:
            lines.append(f"  Insufficient Reason: {self.physiological_evidence.insufficient_reason}")

        if self.visual_evidence:
            lines.append("")
            lines.append("Visual Evidence:")
            lines.append(f"  Reliability: {self.visual_evidence.reliability_level.value}")
            lines.append(f"  P(Fake): {self.visual_evidence.probability_fake:.2f}")
            lines.append(f"  Confidence: {self.visual_evidence.confidence:.2f}")
            lines.append(f"  Features Used: {', '.join(self.visual_evidence.features_used) if self.visual_evidence.features_used else 'N/A'}")

        lines.extend([
            "",
            f"Decision Logic: {self.decision_logic}",
            f"Requires Verification: {'Yes' if self.requires_verification else 'No'}",
            f"Coverage: {self.coverage:.1%}",
            f"Overall Confidence: {self.overall_confidence:.2f}",
        ])
        if self.requires_verification:
            lines.append("")
            lines.append("DECISION: Requires additional verification")
        return "\n".join(lines)

    def to_json(self, output_path: Optional[Path] = None) -> str:
        """Serialize to JSON."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(self.to_dict(), indent=2))
        return json_str


def build_explanation(
    prob_real: float,
    decision_cfg,
    pqs_result: Optional[PQSResult] = None,
    visual_result: Optional[Dict] = None,
    features: Optional[Dict] = None,
) -> DecisionExplanation:
    """
    Build an explainable decision output from model outputs.
    
    Args:
        prob_real: Probability of REAL class from quantum model
        decision_cfg: DecisionConfig with thresholds
        pqs_result: Optional PQS result for physiological quality
        visual_result: Optional dict with visual model results
        features: Optional dict with feature values for explanation
    
    Returns:
        DecisionExplanation with full explanation
    """
    cfg = decision_cfg
    
    # --- Physiological Evidence ---
    phys_verdict = Verdict.REAL if prob_real >= cfg.decision_threshold else Verdict.FAKE
    insufficient_reason = None
    
    # Check PQS quality threshold (for reliability info only, doesn't change verdict)
    reliability = ReliabilityLevel.UNKNOWN
    if pqs_result:
        pqs_score = pqs_result.pqs
        if pqs_score >= 0.7:
            reliability = ReliabilityLevel.HIGH
        elif pqs_result.pqs >= 0.4:
            reliability = ReliabilityLevel.MODERATE
        else:
            reliability = ReliabilityLevel.LOW
    else:
        reliability = ReliabilityLevel.UNKNOWN
    
    phys_evidence = PhysiologicalEvidence(
        verdict=phys_verdict,
        probability_real=prob_real,
        confidence=abs(prob_real - 0.5) * 2,  # distance from 0.5
        pqs_result=pqs_result,
        signal_quality_index=0.0,
        heart_rate_bpm=0.0,
        snr_db=0.0,
        cross_roi_consistency=0.0,
        temporal_consistency=0.0,
        reliability_level=reliability,
        insufficient_reason=None,
    )
    
    if pqs_result:
        comp = pqs_result.components
        phys_evidence.signal_quality_index = comp.signal_quality_index if hasattr(comp, 'signal_quality_index') else 0
        phys_evidence.heart_rate_bpm = getattr(comp, 'frame_utilization', 0)
        phys_evidence.snr_db = comp.snr_quality if hasattr(comp, 'snr_quality') else 0
        phys_evidence.cross_roi_consistency = comp.cross_roi_consistency if hasattr(comp, 'cross_roi_consistency') else 0
        phys_evidence.temporal_consistency = comp.temporal_consistency if hasattr(comp, 'temporal_consistency') else 0
    
    # --- Visual Evidence ---
    visual_evidence = None
    if visual_result:
        vis_verdict = Verdict.FAKE if visual_result.get('probability_fake', 0) > 0.5 else Verdict.REAL
        visual_evidence = VisualEvidence(
            verdict=vis_verdict,
            probability_fake=visual_result.get('probability_fake', 0.0),
            confidence=visual_result.get('confidence', 0.0),
            reliability_level=ReliabilityLevel.MODERATE,
            features_used=visual_result.get('features_used', []),
        )
    
    # --- Decision Logic ---
    decision_parts = []
    if prob_real >= cfg.decision_threshold:
        decision_parts.append(f"P(Real)={prob_real:.3f} >= {cfg.decision_threshold} → REAL")
    else:
        decision_parts.append(f"P(Real)={prob_real:.3f} < {cfg.decision_threshold} → FAKE")
    
    decision_logic = "; ".join(decision_parts)
    
    # Final verdict
    final_verdict = phys_verdict
    
    # Coverage calculation (always 1.0 for binary)
    coverage = 1.0
    
    # Overall confidence
    overall_confidence = abs(prob_real - 0.5) * 2
    if pqs_result:
        overall_confidence = (overall_confidence + pqs_result.pqs) / 2
    
    # Requires verification based on low confidence
    requires_verification = overall_confidence < 0.3
    
    return DecisionExplanation(
        final_verdict=final_verdict,
        physiological_evidence=phys_evidence,
        visual_evidence=visual_evidence,
        decision_logic=decision_logic,
        requires_verification=requires_verification,
        coverage=coverage,
        overall_confidence=overall_confidence,
    )


def create_explanation_from_pipeline_result(
    prob_real: float,
    pqs_result: Optional['PQSResult'] = None,
    visual_result: Optional[Dict] = None,
    decision_cfg = None,
    features: Optional[Dict] = None,
) -> DecisionExplanation:
    """Convenience function to create explanation from pipeline results."""
    from quantum.config import DecisionConfig
    cfg = decision_cfg or DecisionConfig()
    return build_explanation(prob_real, cfg, pqs_result, visual_result)


if __name__ == "__main__":
    # Quick test
    from rppg.pqs import PQSResult, PQSComponents
    
    # Test with mock PQS
    comp = PQSComponents(
        snr_quality=0.8, frame_utilization=0.9, roi_validity=0.9,
        cross_roi_consistency=0.8, frequency_stability=0.7,
        signal_amplitude=0.6, temporal_consistency=0.5
    )
    pqs = PQSResult(pqs=0.75, quality_tier="HIGH", components=comp)
    
    expl = build_explanation(0.8, decision_cfg=None, pqs_result=pqs)
    print(expl.to_text())
    print()
    print(expl.to_json())