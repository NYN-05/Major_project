import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2

from app.detector import FaceDetection


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
