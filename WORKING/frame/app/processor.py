import cv2

from app.detector import FaceDetection


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
