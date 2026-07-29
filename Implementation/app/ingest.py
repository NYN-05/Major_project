from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

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
