from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

DEFAULT_WEIGHTS_FILENAME = "yolov8n-face-lindevs.pt"


@dataclass(frozen=True)
class FaceDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


def resolve_device(device_text: str):
    if device_text in {"gpu", "cuda"}:
        device_text = "0"

    if device_text != "auto":
        if device_text == "0" and not torch.cuda.is_available():
            return "cpu"
        return int(device_text) if device_text.isdigit() else device_text

    return 0 if torch.cuda.is_available() else "cpu"


class FaceDetector:
    def __init__(
        self,
        weights_path: Path,
        confidence_threshold: float,
        image_size: int,
        device: str,
        use_half: bool,
    ):
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = resolve_device(device)
        self.use_half = use_half and self.device != "cpu"
        self.model = self._load_model(weights_path)

    def _load_model(self, weights_path: Path) -> YOLO:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO face weights not found at {weights_path}.\n"
                "The weights file must be bundled in the repo (WORKING/frame/weights/).\n"
                "If missing, download manually from:\n"
                "  https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt\n"
                "and place it at:\n"
                f"  {weights_path}"
            )
        return YOLO(str(weights_path))

    def detect(self, frame) -> list[FaceDetection]:
        results = self.model.predict(
            frame,
            verbose=False,
            device=self.device,
            imgsz=self.image_size,
            half=self.use_half,
        )

        detections: list[FaceDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(1, min(x2, w))
                y2 = max(1, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(FaceDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence))

        return detections


def annotate_frame(frame, detections: list[FaceDetection]):
    annotated = frame.copy()

    for detection in detections:
        cv2.rectangle(
            annotated,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"{detection.confidence:.2f}",
            (detection.x1, max(detection.y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        annotated,
        f"Faces: {len(detections)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return annotated


def crop_faces(frame, detections: list[FaceDetection]):
    crops = []
    for face_id, detection in enumerate(detections, start=1):
        crop = frame[detection.y1:detection.y2, detection.x1:detection.x2]
        if crop.size == 0:
            continue
        crops.append((face_id, crop))
    return crops
