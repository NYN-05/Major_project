from dataclasses import dataclass
from pathlib import Path

MODEL_WEIGHTS = {
    "yolov8n": "weights/yolov8n-face-lindevs.pt",
    "yolov8x": "weights/yolov8x-face-lindevs.pt",
    "yolov9e": "weights/yolov9e-face-lindevs.pt",
    "yolov9t": "weights/yolov9t-face-lindevs.pt",
}


@dataclass(frozen=True)
class PipelineConfig:
    source: str
    weights_path: Path
    confidence_threshold: float
    image_size: int
    device: str
    use_half: bool
    display: bool
    output_root: Path
    full_frames_dir: Path
    cropped_faces_dir: Path
    metadata_file: Path
    save_metadata: bool
    max_io_workers: int


def build_config(
    source: str,
    weights_path: str,
    confidence_threshold: float,
    image_size: int,
    device: str,
    use_half: bool,
    display: bool,
    output_root: str,
    save_metadata: bool,
    max_io_workers: int,
) -> PipelineConfig:
    output_root_path = Path(output_root)
    full_frames_dir = output_root_path / "full_frames"
    cropped_faces_dir = output_root_path / "cropped_faces"
    metadata_file = output_root_path / "metadata.jsonl"

    return PipelineConfig(
        source=source,
        weights_path=Path(weights_path),
        confidence_threshold=confidence_threshold,
        image_size=image_size,
        device=device,
        use_half=use_half,
        display=display,
        output_root=output_root_path,
        full_frames_dir=full_frames_dir,
        cropped_faces_dir=cropped_faces_dir,
        metadata_file=metadata_file,
        save_metadata=save_metadata,
        max_io_workers=max(1, max_io_workers),
    )
