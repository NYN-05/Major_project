from dataclasses import dataclass

import cv2

from app.detector import FaceDetection


@dataclass(frozen=True)
class QualityResult:
    blur_score: float
    brightness_score: float
    face_confidence: float
    max_face_area_ratio: float
    face_visible: bool
    extreme_pose: bool
    face_too_small: bool
    accepted: bool
    quality_flag: str
    rejection_reasons: list[str]
    quality_score: float


class FrameQualityAssessor:
    def __init__(
        self,
        blur_threshold: float,
        dark_threshold: float,
        bright_threshold: float,
        min_face_area_ratio: float = 0.02,
        edge_margin_ratio: float = 0.03,
        score_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    ):
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.min_face_area_ratio = min_face_area_ratio
        self.edge_margin_ratio = edge_margin_ratio
        self.score_weights = score_weights

    @staticmethod
    def _normalize_brightness(score: float) -> float:
        # Center quality around mid-tones and degrade towards very dark/bright values.
        return max(0.0, min(1.0, 1.0 - (abs(score - 128.0) / 128.0)))

    def evaluate(self, frame, detections: list[FaceDetection]) -> QualityResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_score = float(gray.mean())
        face_visible = len(detections) > 0
        face_confidence = max((float(det.confidence) for det in detections), default=0.0)
        max_face_area_ratio = 0.0

        extreme_pose = False
        face_too_small = False
        if face_visible:
            h, w = frame.shape[:2]
            frame_area = max(1, h * w)
            margin_x = int(w * self.edge_margin_ratio)
            margin_y = int(h * self.edge_margin_ratio)
            for det in detections:
                bw = max(1, det.x2 - det.x1)
                bh = max(1, det.y2 - det.y1)
                area_ratio = (bw * bh) / frame_area
                max_face_area_ratio = max(max_face_area_ratio, area_ratio)
                aspect_ratio = bw / bh
                touches_edge = (
                    det.x1 <= margin_x
                    or det.y1 <= margin_y
                    or det.x2 >= w - margin_x
                    or det.y2 >= h - margin_y
                )
                unusual_shape = aspect_ratio < 0.65 or aspect_ratio > 1.7
                if touches_edge or unusual_shape:
                    extreme_pose = True

            if max_face_area_ratio < self.min_face_area_ratio:
                face_too_small = True

        reasons: list[str] = []
        if blur_score < self.blur_threshold:
            reasons.append("blur")
        if brightness_score < self.dark_threshold:
            reasons.append("dark_frame")
        if brightness_score > self.bright_threshold:
            reasons.append("overexposed_frame")
        if not face_visible:
            reasons.append("no_face")
        if face_too_small:
            reasons.append("face_too_small")
        if extreme_pose:
            reasons.append("extreme_pose")

        accepted = len(reasons) == 0

        blur_norm = max(0.0, min(1.0, blur_score / max(self.blur_threshold, 1e-6)))
        brightness_norm = self._normalize_brightness(brightness_score)
        face_norm = max(0.0, min(1.0, face_confidence))
        w1, w2, w3 = self.score_weights
        score = (w1 * blur_norm) + (w2 * brightness_norm) + (w3 * face_norm)
        score = max(0.0, min(1.0, score))
        quality_flag = "accept" if accepted else "reject"

        return QualityResult(
            blur_score=blur_score,
            brightness_score=brightness_score,
            face_confidence=face_confidence,
            max_face_area_ratio=max_face_area_ratio,
            face_visible=face_visible,
            extreme_pose=extreme_pose,
            face_too_small=face_too_small,
            accepted=accepted,
            quality_flag=quality_flag,
            rejection_reasons=reasons,
            quality_score=score,
        )
