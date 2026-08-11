import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from app.detector import FaceDetection

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    frame: Any
    timestamp_ms: int
    source_frame_id: int


class FrameIngestor:
    def __init__(self, source: str):
        self.source = source
        self.capture = None
        self._image_mode = False
        self._opened_backend = None

    def _resolve_source(self):
        source_text = str(self.source)
        source_path = Path(source_text)

        if source_path.exists() and source_path.is_file() and source_path.suffix.lower() in IMAGE_EXTENSIONS:
            self._image_mode = True
            return source_path

        if source_text.isdigit():
            return int(source_text)

        return source_text

    def open(self):
        resolved = self._resolve_source()
        if self._image_mode:
            return resolved

        candidate_backends = [None]
        if isinstance(resolved, int):
            candidate_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]

        for backend in candidate_backends:
            if backend is None:
                self.capture = cv2.VideoCapture(resolved)
            else:
                self.capture = cv2.VideoCapture(resolved, backend)

            if self.capture.isOpened():
                self._opened_backend = backend
                ok, frame = self.capture.read()
                if ok and frame is not None:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return resolved

            self.close()

        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError(f"Could not open source: {self.source}")
        return resolved

    def _sample_step(self, sample_fps: float | None) -> int:
        if sample_fps is None or sample_fps <= 0:
            return 1

        if self.capture is None:
            return 1

        source_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            return 1

        return max(1, int(round(source_fps / sample_fps)))

    def frames(self, sample_fps: float | None = None):
        resolved = self.open()

        if self._image_mode:
            image = cv2.imread(str(resolved))
            if image is None:
                raise RuntimeError(f"Could not read image: {resolved}")
            yield FramePacket(frame_id=1, frame=image, timestamp_ms=1, source_frame_id=1)
            self.close()
            return

        frame_id = 0
        source_frame_id = 0
        sample_step = self._sample_step(sample_fps)
        sample_interval_ms = (1000.0 / sample_fps) if (sample_fps is not None and sample_fps > 0) else None
        next_target_ms = 0.0

        while True:
            ok, frame = self.capture.read()
            if not ok:
                break

            source_frame_id += 1
            timestamp_ms = int(self.capture.get(cv2.CAP_PROP_POS_MSEC))

            if sample_interval_ms is None:
                if (source_frame_id - 1) % sample_step != 0:
                    continue
            else:
                if timestamp_ms < next_target_ms:
                    continue

            frame_id += 1
            yield FramePacket(
                frame_id=frame_id,
                frame=frame,
                timestamp_ms=timestamp_ms,
                source_frame_id=source_frame_id,
            )

            if sample_interval_ms is not None:
                while next_target_ms <= timestamp_ms:
                    next_target_ms += sample_interval_ms

        self.close()

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


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


class StorageManager:
    def __init__(
        self,
        full_frames_dir: Path,
        cropped_faces_dir: Path,
        metadata_file: Path,
        save_metadata: bool,
        max_workers: int = 4,
    ):
        self.full_frames_dir = full_frames_dir
        self.cropped_faces_dir = cropped_faces_dir
        self.metadata_file = metadata_file
        self.save_metadata = save_metadata
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending: list[Future] = []

        self.full_frames_dir.mkdir(parents=True, exist_ok=True)
        self.cropped_faces_dir.mkdir(parents=True, exist_ok=True)
        if self.save_metadata:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

    def _submit(self, fn, *args):
        future = self.executor.submit(fn, *args)
        self.pending.append(future)

    def save_annotated_frame(self, timestamp_id: str, annotated_frame):
        path = self.full_frames_dir / f"frame_{timestamp_id}.jpg"
        self._submit(cv2.imwrite, str(path), annotated_frame)

    def save_face_crop(self, frame_id: int, face_id: int, crop):
        path = self.cropped_faces_dir / f"face_{frame_id:06d}_{face_id:02d}.jpg"
        self._submit(cv2.imwrite, str(path), crop)

    def save_metadata_record(self, frame_id: int, timestamp_id: str, detections: list[FaceDetection]):
        if not self.save_metadata:
            return

        record: dict[str, Any] = {
            "frame_id": frame_id,
            "timestamp": timestamp_id,
            "faces": [
                {
                    "x": detection.x1,
                    "y": detection.y1,
                    "w": detection.x2 - detection.x1,
                    "h": detection.y2 - detection.y1,
                    "confidence": round(detection.confidence, 4),
                }
                for detection in detections
            ],
        }

        self._submit(self._append_jsonl, self.metadata_file, record)

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]):
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\n")

    def flush(self):
        for future in self.pending:
            future.result()
        self.pending.clear()

    def close(self):
        self.flush()
        self.executor.shutdown(wait=True)