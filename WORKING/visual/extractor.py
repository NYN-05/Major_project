"""
visual/extractor.py
===================
Visual feature extraction from face crops using a pre-trained CNN backbone.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from dataclasses import dataclass, asdict
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from quantum.operations import resolve_torch_device

RESNET_FEATURE_DIM = 2048
PCA_PREPROCESSING_VERSION = "resnet50_imagenet_v1_bgr_to_rgb_224_normalized"


@dataclass
class VisualFeatures:
    """
    Compact visual feature representation extracted from face crops.

    Uses the penultimate layer (global average pooling) of a pre-trained
    ResNet50 (2048-dim) or similar backbone. For the quantum pipeline,
    we project to a lower dimension (e.g., 16 features) via PCA or
    use the raw 2048-dim vector with a classical baseline.
    """
    # Deep CNN features (ResNet50 GAP output = 2048 dim)
    # We store a reduced set for practical use with the quantum pipeline
    deep_feat_0: float = 0.0
    deep_feat_1: float = 0.0
    deep_feat_2: float = 0.0
    deep_feat_3: float = 0.0
    deep_feat_4: float = 0.0
    deep_feat_5: float = 0.0
    deep_feat_6: float = 0.0
    deep_feat_7: float = 0.0
    deep_feat_8: float = 0.0
    deep_feat_9: float = 0.0
    deep_feat_10: float = 0.0
    deep_feat_11: float = 0.0
    deep_feat_12: float = 0.0
    deep_feat_13: float = 0.0
    deep_feat_14: float = 0.0
    deep_feat_15: float = 0.0

    # Handcrafted complementary features
    lbp_uniform_hist_0: float = 0.0
    lbp_uniform_hist_1: float = 0.0
    lbp_uniform_hist_2: float = 0.0
    lbp_uniform_hist_3: float = 0.0
    lbp_uniform_hist_4: float = 0.0
    lbp_uniform_hist_5: float = 0.0
    lbp_uniform_hist_6: float = 0.0
    lbp_uniform_hist_7: float = 0.0
    lbp_uniform_hist_8: float = 0.0
    lbp_uniform_hist_9: float = 0.0

    # Texture statistics (computed on grayscale face crop)
    texture_contrast: float = 0.0
    texture_energy: float = 0.0
    texture_homogeneity: float = 0.0
    texture_correlation: float = 0.0

    # Color statistics (mean/std per channel on face crop)
    color_mean_r: float = 0.0
    color_mean_g: float = 0.0
    color_mean_b: float = 0.0
    color_std_r: float = 0.0
    color_std_g: float = 0.0
    color_std_b: float = 0.0

    # Frequency domain (DCT/FFT energy in bands)
    freq_low_energy: float = 0.0
    freq_mid_energy: float = 0.0
    freq_high_energy: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Fixed-order numeric feature vector."""
        return np.array([
            # Deep features (16 dims)
            self.deep_feat_0, self.deep_feat_1, self.deep_feat_2, self.deep_feat_3,
            self.deep_feat_4, self.deep_feat_5, self.deep_feat_6, self.deep_feat_7,
            self.deep_feat_8, self.deep_feat_9, self.deep_feat_10, self.deep_feat_11,
            self.deep_feat_12, self.deep_feat_13, self.deep_feat_14, self.deep_feat_15,
            # LBP histogram (10 dims)
            self.lbp_uniform_hist_0, self.lbp_uniform_hist_1, self.lbp_uniform_hist_2,
            self.lbp_uniform_hist_3, self.lbp_uniform_hist_4, self.lbp_uniform_hist_5,
            self.lbp_uniform_hist_6, self.lbp_uniform_hist_7, self.lbp_uniform_hist_8,
            self.lbp_uniform_hist_9,
            # Texture (4 dims)
            self.texture_contrast, self.texture_energy, self.texture_homogeneity,
            self.texture_correlation,
            # Color (6 dims)
            self.color_mean_r, self.color_mean_g, self.color_mean_b,
            self.color_std_r, self.color_std_g, self.color_std_b,
            # Frequency (3 dims)
            self.freq_low_energy, self.freq_mid_energy, self.freq_high_energy,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            # Deep features (16 dims)
            "deep_feat_0", "deep_feat_1", "deep_feat_2", "deep_feat_3",
            "deep_feat_4", "deep_feat_5", "deep_feat_6", "deep_feat_7",
            "deep_feat_8", "deep_feat_9", "deep_feat_10", "deep_feat_11",
            "deep_feat_12", "deep_feat_13", "deep_feat_14", "deep_feat_15",
            # LBP histogram (10 dims)
            "lbp_uniform_hist_0", "lbp_uniform_hist_1", "lbp_uniform_hist_2",
            "lbp_uniform_hist_3", "lbp_uniform_hist_4", "lbp_uniform_hist_5",
            "lbp_uniform_hist_6", "lbp_uniform_hist_7", "lbp_uniform_hist_8",
            "lbp_uniform_hist_9",
            # Texture (4 dims)
            "texture_contrast", "texture_energy", "texture_homogeneity",
            "texture_correlation",
            # Color (6 dims)
            "color_mean_r", "color_mean_g", "color_mean_b",
            "color_std_r", "color_std_g", "color_std_b",
            # Frequency (3 dims)
            "freq_low_energy", "freq_mid_energy", "freq_high_energy",
        ]

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# Full feature list (39 features total); must match quantum.config.VISUAL_FEATURE_NAMES.
VISUAL_FEATURE_NAMES = VisualFeatures.feature_names()


class VisualFeatureExtractor:
    """
    Extracts visual features from face crops using a pre-trained ResNet50
    plus complementary handcrafted features (LBP, texture, color, frequency).
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        device: str = "auto",
        deep_feature_dim: int = 16,
        use_pca: bool = True,
        pca_path: Optional[str] = None,
        allow_pca_fit: bool = False,
        batch_size: int = 32,
    ):
        self.backbone_name = backbone
        self.deep_feature_dim = deep_feature_dim
        self.use_pca = use_pca
        self.allow_pca_fit = allow_pca_fit
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size

        if device == "auto":
            self.device = resolve_torch_device("auto")
        else:
            self.device = resolve_torch_device(device)

        # Initialize backbone
        if backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.model = resnet50(weights=weights)
            # Remove final FC layer, keep up to global avg pool (2048-d)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.model.eval()
        self.model.to(self.device)

        # Preprocessing transform for ResNet
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # PCA is always loaded from, or explicitly fitted into, one artifact.
        # It must never be fitted implicitly while processing a video.
        self.pca = None
        self.pca_fitted = False
        self.actual_pca_dim = None
        self.pca_artifact_path = None
        if self.use_pca and pca_path is not None:
            self.load_pca(pca_path)

    def extract_deep_features(self, face_crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract deep CNN features from a list of face crops.

        Args:
            face_crops: List of BGR face crop images (H, W, 3)

        Returns:
            Array of shape (N, 2048) - ResNet50 GAP features per frame
        """
        if not face_crops:
            return np.empty((0, 2048), dtype=np.float32)

        chunks = []
        with torch.inference_mode():
            for start in range(0, len(face_crops), self.batch_size):
                batch = [
                    self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    for crop in face_crops[start:start + self.batch_size]
                ]
                batch_tensor = torch.stack(batch).to(self.device, non_blocking=True)
                output = self.model(batch_tensor).flatten(1).cpu().numpy()
                chunks.append(output)
                del batch_tensor, output
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        features = np.concatenate(chunks, axis=0)
        if features.shape[1] != RESNET_FEATURE_DIM:
            raise ValueError(
                f"Expected ResNet50 GAP dimension {RESNET_FEATURE_DIM}, "
                f"got {features.shape[1]}"
            )
        return features

    def fit_pca(self, features: np.ndarray, artifact_path: Optional[str] = None) -> None:
        """Fit the training PCA once and optionally persist it.

        This method is intentionally explicit. Production extraction must pass
        a fitted artifact and must not call this method per video.
        """
        from sklearn.decomposition import PCA

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != RESNET_FEATURE_DIM:
            raise ValueError(
                f"PCA training data must have shape (n, {RESNET_FEATURE_DIM}), "
                f"got {features.shape}"
            )
        if features.shape[0] < self.deep_feature_dim + 1:
            raise ValueError(
                f"At least {self.deep_feature_dim + 1} training frame features are "
                f"required to fit {self.deep_feature_dim}-component PCA; "
                f"got {features.shape[0]}"
            )
        n_components = self.deep_feature_dim
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(features)
        self.pca_fitted = True
        self.actual_pca_dim = n_components
        if artifact_path is not None:
            self.save_pca(artifact_path)

    def _apply_pca(self, features: np.ndarray) -> np.ndarray:
        """Apply fitted PCA to reduce deep features."""
        if not self.pca_fitted:
            raise RuntimeError(
                "A training-fitted visual PCA artifact is required. "
                "Fit PCA on the training partition and pass pca_path."
            )
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != RESNET_FEATURE_DIM:
            raise ValueError(
                f"PCA input must have shape (n, {RESNET_FEATURE_DIM}), got {features.shape}"
            )
        reduced = self.pca.transform(features)
        if reduced.shape[1] != self.deep_feature_dim:
            raise ValueError(
                f"PCA artifact produced {reduced.shape[1]} components; "
                f"expected {self.deep_feature_dim}"
            )
        return reduced

    def save_pca(self, artifact_path: str) -> None:
        """Persist PCA parameters in a non-pickle, validated NumPy artifact."""
        if not self.pca_fitted or self.pca is None:
            raise RuntimeError("Cannot save PCA before fitting it")
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            components=np.asarray(self.pca.components_, dtype=np.float64),
            mean=np.asarray(self.pca.mean_, dtype=np.float64),
            explained_variance=np.asarray(self.pca.explained_variance_, dtype=np.float64),
            explained_variance_ratio=np.asarray(self.pca.explained_variance_ratio_, dtype=np.float64),
            singular_values=np.asarray(self.pca.singular_values_, dtype=np.float64),
            n_features_in=np.asarray(self.pca.n_features_in_, dtype=np.int64),
            n_components=np.asarray(self.pca.n_components_, dtype=np.int64),
            deep_feature_dim=np.asarray(self.deep_feature_dim, dtype=np.int64),
            preprocessing_version=np.asarray(PCA_PREPROCESSING_VERSION),
            backbone=np.asarray(self.backbone_name),
        )
        self.pca_artifact_path = path

    def load_pca(self, artifact_path: str) -> None:
        """Load and validate a trusted training PCA artifact."""
        path = Path(artifact_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Training-fitted visual PCA artifact not found: {path}. "
                "Run the visual training extraction first."
            )
        with np.load(path, allow_pickle=False) as artifact:
            required = {
                "components", "mean", "explained_variance",
                "explained_variance_ratio", "singular_values",
                "n_features_in", "n_components", "deep_feature_dim",
                "preprocessing_version", "backbone",
            }
            missing = required.difference(artifact.files)
            if missing:
                raise ValueError(f"Visual PCA artifact is missing fields: {sorted(missing)}")
            if int(artifact["n_features_in"]) != RESNET_FEATURE_DIM:
                raise ValueError(
                    f"Visual PCA expects {int(artifact['n_features_in'])} input features; "
                    f"expected {RESNET_FEATURE_DIM}"
                )
            if int(artifact["n_components"]) != self.deep_feature_dim:
                raise ValueError(
                    f"Visual PCA has {int(artifact['n_components'])} components; "
                    f"requested {self.deep_feature_dim}"
                )
            if str(artifact["preprocessing_version"]) != PCA_PREPROCESSING_VERSION:
                raise ValueError("Visual PCA preprocessing version does not match the extractor")
            if str(artifact["backbone"]) != self.backbone_name:
                raise ValueError("Visual PCA backbone does not match the extractor")

            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.deep_feature_dim)
            pca.components_ = np.asarray(artifact["components"], dtype=np.float64)
            pca.mean_ = np.asarray(artifact["mean"], dtype=np.float64)
            pca.explained_variance_ = np.asarray(artifact["explained_variance"], dtype=np.float64)
            pca.explained_variance_ratio_ = np.asarray(
                artifact["explained_variance_ratio"], dtype=np.float64
            )
            pca.singular_values_ = np.asarray(artifact["singular_values"], dtype=np.float64)
            pca.n_features_in_ = RESNET_FEATURE_DIM
            pca.n_samples_ = 2
            pca.n_components_ = self.deep_feature_dim
            self.pca = pca
        self.pca_fitted = True
        self.actual_pca_dim = self.deep_feature_dim
        self.pca_artifact_path = path

    def compute_lbp_histogram(self, gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Compute uniform LBP histogram (10 bins for uniform patterns)."""
        from skimage.feature import local_binary_pattern

        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        n_bins = n_points + 2  # uniform patterns + 1 for non-uniform
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist.astype(np.float64)

    def compute_texture_features(self, gray: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute GLCM texture features (contrast, energy, homogeneity, correlation)."""
        from skimage.feature import graycomatrix, graycoprops

        # Quantize to 32 levels for GLCM
        gray_q = (gray / 8).astype(np.uint8)
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_q, distances, angles, levels=32, symmetric=True, normed=True)

        contrast = float(np.mean(graycoprops(glcm, 'contrast')))
        energy = float(np.mean(graycoprops(glcm, 'energy')))
        homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
        correlation = float(np.mean(graycoprops(glcm, 'correlation')))

        return contrast, energy, homogeneity, correlation

    def compute_color_stats(self, crop: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """Compute mean and std per color channel."""
        means = crop.mean(axis=(0, 1)).astype(np.float64)  # BGR
        stds = crop.std(axis=(0, 1)).astype(np.float64)
        # Return as RGB order
        return float(means[2]), float(means[1]), float(means[0]), float(stds[2]), float(stds[1]), float(stds[0])

    def compute_frequency_features(self, gray: np.ndarray) -> Tuple[float, float, float]:
        """Compute DCT energy in low/mid/high frequency bands."""
        h, w = gray.shape
        # DCT on 64x64 normalized patch
        patch = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
        dct = cv2.dct(patch)

        # Energy in frequency bands
        low = dct[:8, :8]
        mid = dct[8:24, 8:24]
        high = dct[24:, 24:]

        low_energy = float(np.sum(low ** 2))
        mid_energy = float(np.sum(mid ** 2))
        high_energy = float(np.sum(high ** 2))

        total = low_energy + mid_energy + high_energy + 1e-12
        return low_energy / total, mid_energy / total, high_energy / total

    def extract_from_crops(
        self,
        face_crops: List[np.ndarray],
        deep_features: Optional[np.ndarray] = None,
    ) -> VisualFeatures:
        """
        Extract full visual feature vector from a sequence of face crops.

        Aggregates per-frame features by averaging across frames.
        """
        if not face_crops:
            return VisualFeatures()

        # Transform each frame with the shared training PCA, then aggregate.
        deep_feats = (
            self.extract_deep_features(face_crops)
            if deep_features is None
            else np.asarray(deep_features, dtype=np.float32)
        )
        if deep_feats.shape != (len(face_crops), RESNET_FEATURE_DIM):
            raise ValueError(
                f"Expected deep features with shape ({len(face_crops)}, "
                f"{RESNET_FEATURE_DIM}), got {deep_feats.shape}"
            )
        deep_agg = deep_feats.mean(axis=0)  # (2048,)
        deep_reduced = self._apply_pca(deep_agg.reshape(1, -1)).flatten()

        # Handcrafted features - compute on each frame and average
        lbp_hists = []
        texture_feats = []
        color_feats = []
        freq_feats = []

        for crop in face_crops:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            lbp_hist = self.compute_lbp_histogram(gray)
            lbp_hists.append(lbp_hist)

            contrast, energy, homogeneity, correlation = self.compute_texture_features(gray)
            texture_feats.append([contrast, energy, homogeneity, correlation])

            color_mean_r, color_mean_g, color_mean_b, color_std_r, color_std_g, color_std_b = self.compute_color_stats(crop)
            color_feats.append([color_mean_r, color_mean_g, color_mean_b, color_std_r, color_std_g, color_std_b])

            freq_low, freq_mid, freq_high = self.compute_frequency_features(gray)
            freq_feats.append([freq_low, freq_mid, freq_high])

        lbp_agg = np.mean(lbp_hists, axis=0)  # (10,)
        texture_agg = np.mean(texture_feats, axis=0)  # (4,)
        color_agg = np.mean(color_feats, axis=0)  # (6,)
        freq_agg = np.mean(freq_feats, axis=0)  # (3,)

        # Build VisualFeatures object
        features = VisualFeatures()

        # Deep features (16)
        for i in range(16):
            setattr(features, f"deep_feat_{i}", float(deep_reduced[i]))

        # LBP (10)
        for i in range(10):
            setattr(features, f"lbp_uniform_hist_{i}", float(lbp_agg[i]))

        # Texture (4)
        features.texture_contrast = float(texture_agg[0])
        features.texture_energy = float(texture_agg[1])
        features.texture_homogeneity = float(texture_agg[2])
        features.texture_correlation = float(texture_agg[3])

        # Color (6)
        features.color_mean_r = float(color_agg[0])
        features.color_mean_g = float(color_agg[1])
        features.color_mean_b = float(color_agg[2])
        features.color_std_r = float(color_agg[3])
        features.color_std_g = float(color_agg[4])
        features.color_std_b = float(color_agg[5])

        # Frequency (3)
        features.freq_low_energy = float(freq_agg[0])
        features.freq_mid_energy = float(freq_agg[1])
        features.freq_high_energy = float(freq_agg[2])

        return features


def compute_visual_features(
    frames_dir: str,
    metadata_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    backbone: str = "resnet50",
    device: str = "auto",
    deep_feature_dim: int = 16,
    pca_path: Optional[str] = None,
) -> VisualFeatures:
    """
    Convenience function to extract visual features from stage-1 frames.

    Reads accepted frames from the frame stage output directory and
    extracts visual features using the face crops saved by stage 1.
    """
    frames_root = Path(frames_dir)
    frame_paths = sorted(frames_root.glob("*.jpg"))
    if not frame_paths:
        raise IOError(f"No frames found in: {frames_dir}")

    face_crops = load_face_crops_from_frames(frames_dir, max_frames=max_frames)
    extractor = VisualFeatureExtractor(
        backbone=backbone,
        device=device,
        deep_feature_dim=deep_feature_dim,
        pca_path=pca_path or _default_pca_path(),
    )
    return extractor.extract_from_crops(face_crops)


def load_face_crops_from_frames(
    frames_dir: str,
    max_frames: Optional[int] = None,
) -> List[np.ndarray]:
    """Load the accepted stage-1 face crop for each frame."""
    frames_root = Path(frames_dir)
    frame_paths = sorted(frames_root.glob("*.jpg"))
    if not frame_paths:
        raise IOError(f"No frames found in: {frames_dir}")
    crops_dir = frames_root.parent / "cropped_faces"
    if not crops_dir.exists():
        crops_dir = frames_root

    face_crops = []
    for frame_path in frame_paths:
        frame_id = int(frame_path.stem.split("_")[1])  # frame_000001_t...
        # Stage 1 saves: frame_000001_face_0.jpg
        crop_files = list(crops_dir.glob(f"frame_{frame_id:06d}_face_*.jpg"))
        if crop_files:
            # Use the first (largest) face crop
            crop = cv2.imread(str(crop_files[0]))
            if crop is not None and crop.size > 0:
                face_crops.append(crop)
        if max_frames and len(face_crops) >= max_frames:
            break

    if not face_crops:
        for frame_path in frame_paths[:max_frames] if max_frames else frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                h, w = frame.shape[:2]
                # Center crop assuming face is in center
                crop = frame[h//4:3*h//4, w//4:3*w//4]
                if crop.size > 0:
                    face_crops.append(crop)

    return face_crops


def iter_video_center_crops(
    video_path,
    sample_fps: float = 10.0,
    max_frames: Optional[int] = None,
) -> List[np.ndarray]:
    """Yield center-cropped frames sampled at ``sample_fps`` from a video.

    Shared by the frames-dir fallback and the batch video-root extraction
    in visual/pipeline.py (single source for the crop/stride logic).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(native_fps / sample_fps))

    crops: List[np.ndarray] = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                h, w = frame.shape[:2]
                crop = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
                if crop.size > 0:
                    crops.append(crop)
                if max_frames and len(crops) >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()
    return crops


def _default_pca_path() -> str:
    output_root = Path(os.environ.get("MAJ_OUTPUT_ROOT", Path(__file__).resolve().parents[2] / "Scrape" / "output"))
    return str(output_root / "visual" / "visual_pca.npz")