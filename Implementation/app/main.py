import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import MODEL_WEIGHTS, build_config
from app.detector import FaceDetector
from app.extract_frames import run_frame_sampling_quality_layer
from app.ingest import FrameIngestor
from app.logger import setup_logger
from app.processor import annotate_frame, crop_faces
from app.storage import StorageManager


def parse_args():
    parser = argparse.ArgumentParser(description="Production-grade face pipeline for image, video, and webcam sources.")
    parser.add_argument("--source", default="1", help="Image path, video path, stream URL, or webcam index")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_WEIGHTS.keys()),
        default="yolov8n",
        help="Predefined YOLO face model preset",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Custom YOLO face model weights path (overrides --model)",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Minimum face confidence threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="Model inference image size")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, 0, 1, gpu, cuda")
    parser.add_argument("--half", action="store_true", help="Use half precision on CUDA")
    parser.add_argument("--output-root", default="output", help="Root output folder")
    parser.add_argument("--display", action="store_true", help="Display annotated frames in a window")
    parser.add_argument("--save-metadata", action="store_true", help="Write JSONL metadata per frame")
    parser.add_argument("--max-io-workers", type=int, default=4, help="Background workers for storage I/O")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger()
    source = "1" if str(args.source).strip() == "0" else args.source
    if source != args.source:
        logger.warning("Remapping webcam source 0 to camera index 1")
    selected_weights = args.weights if args.weights else MODEL_WEIGHTS[args.model]

    config = build_config(
        source=source,
        weights_path=selected_weights,
        confidence_threshold=args.conf,
        image_size=args.imgsz,
        device=args.device,
        use_half=args.half,
        display=args.display,
        output_root=args.output_root,
        save_metadata=args.save_metadata,
        max_io_workers=args.max_io_workers,
    )

    detector = FaceDetector(
        weights_path=config.weights_path,
        confidence_threshold=config.confidence_threshold,
        image_size=config.image_size,
        device=config.device,
        use_half=config.use_half,
    )
    ingestor = FrameIngestor(config.source)
    storage = StorageManager(
        full_frames_dir=config.full_frames_dir,
        cropped_faces_dir=config.cropped_faces_dir,
        metadata_file=config.metadata_file,
        save_metadata=config.save_metadata,
        max_workers=config.max_io_workers,
    )

    logger.info("Pipeline started")
    start = time.time()
    processed_frames = 0

    try:
        for packet in ingestor.frames():
            detections = detector.detect(packet.frame)
            annotated = annotate_frame(packet.frame, detections)

            timestamp_id = str(time.time_ns())
            storage.save_annotated_frame(timestamp_id=timestamp_id, annotated_frame=annotated)

            for face_id, crop in crop_faces(packet.frame, detections):
                storage.save_face_crop(frame_id=packet.frame_id, face_id=face_id, crop=crop)

            storage.save_metadata_record(
                frame_id=packet.frame_id,
                timestamp_id=timestamp_id,
                detections=detections,
            )

            processed_frames += 1

            if config.display:
                cv2.imshow("Face Pipeline", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        ingestor.close()
        storage.close()
        cv2.destroyAllWindows()

    elapsed = max(time.time() - start, 1e-6)
    fps = processed_frames / elapsed
    logger.info("Pipeline finished | frames=%d | fps=%.2f", processed_frames, fps)
    logger.info("Annotated frames: %s", config.full_frames_dir)
    logger.info("Cropped faces: %s", config.cropped_faces_dir)
    if config.save_metadata:
        logger.info("Metadata file: %s", config.metadata_file)

    layer_result = run_frame_sampling_quality_layer(
        source=config.source,
        weights_path=str(config.weights_path),
        confidence_threshold=config.confidence_threshold,
        image_size=config.image_size,
        device=config.device,
        use_half=config.use_half,
    )

    if layer_result.get("status") == "skipped":
        logger.info("Sampling/quality layer skipped: %s", layer_result.get("reason", "n/a"))
    else:
        logger.info("Sampling/quality layer status: %s", layer_result.get("status", "unknown"))
        logger.info("Sampling/quality summary: %s", layer_result.get("summary_file", "n/a"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
