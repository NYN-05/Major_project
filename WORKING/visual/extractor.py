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
from torchvision.models import resnet50, ResNet50_Weights
from typing import List, Optional, Tuple
from pathlib import Path

from .features import VisualFeatures, VISUAL_FEATURE_NAMES


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
    ):
        self.backbone_name = backbone
        self.deep_feature_dim = deep_feature_dim
        self.use_pca = use_pca

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

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

        # PCA projection for deep features (fitted on first batch)
        self.pca_components = None
        self.pca_mean = None
        self.pca_fitted = False

    @torch.no_grad()
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

        batch = []
        for crop in face_crops:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb)
            batch.append(tensor)

        batch_tensor = torch.stack(batch).to(self.device)
        features = self.model(batch_tensor)  # (N, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1).cpu().numpy()  # (N, 2048)
        return features

    def _fit_pca(self, features: np.ndarray) -> None:
        """Fit PCA on deep features to reduce to target dimension."""
        from sklearn.decomposition import PCA
        n_samples = features.shape[0]
        n_components = min(self.deep_feature_dim, n_samples - 1, features.shape[1])
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(features)
        self.pca_fitted = True
        self.actual_pca_dim = n_components

    def _apply_pca(self, features: np.ndarray) -> np.ndarray:
        """Apply fitted PCA to reduce deep features."""
        if not self.pca_fitted:
            self._fit_pca(features)
        reduced = self.pca.transform(features)
        # Pad with zeros if actual PCA dim < target dim
        if reduced.shape[1] < self.deep_feature_dim:
            padding = np.zeros((reduced.shape[0], self.deep_feature_dim - reduced.shape[1]), dtype=reduced.dtype)
            reduced = np.hstack([reduced, padding])
        return reduced

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

    def extract_from_crops(self, face_crops: List[np.ndarray]) -> VisualFeatures:
        """
        Extract full visual feature vector from a sequence of face crops.

        Aggregates per-frame features by averaging across frames.
        """
        if not face_crops:
            return VisualFeatures()

        # Deep features - fit PCA on per-frame features, then transform aggregated
        deep_feats = self.extract_deep_features(face_crops)  # (N, 2048)
        if not self.pca_fitted and len(deep_feats) >= 2:
            self._fit_pca(deep_feats)
        deep_agg = deep_feats.mean(axis=0)  # (2048,)
        if self.pca_fitted:
            deep_reduced = self._apply_pca(deep_agg.reshape(1, -1)).flatten()  # (16,)
        else:
            # Fallback: just take first 16 components if not enough samples for PCA
            deep_reduced = deep_agg[:16]

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

    # Load face crops from stage 1's cropped_faces directory
    # Stage 1 saves crops at: output/frames/frame_sequences/<video>/cropped_faces/
    video_name = frames_root.parent.name
    crops_dir = frames_root.parent / "cropped_faces"
    if not crops_dir.exists():
        # Fallback: use full frames and detect faces again
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
        # Fallback: read full frames and use center crop as face region
        for frame_path in frame_paths[:max_frames] if max_frames else frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                h, w = frame.shape[:2]
                # Center crop assuming face is in center
                crop = frame[h//4:3*h//4, w//4:3*w//4]
                if crop.size > 0:
                    face_crops.append(crop)

    extractor = VisualFeatureExtractor(
        backbone=backbone,
        device=device,
        deep_feature_dim=deep_feature_dim,
    )
    return extractor.extract_from_crops(face_crops)


def compute_visual_features_from_video(
    video_path: str,
    max_frames: Optional[int] = None,
    sample_fps: float = 10.0,
    backbone: str = "resnet50",
    device: str = "auto",
    deep_feature_dim: int = 16,
) -> VisualFeatures:
    """
    Extract visual features directly from a video file (fallback when
    stage-1 frames are not available).
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_stride = max(1, round(native_fps / sample_fps))

    face_crops = []
    frame_idx = 0

    # Use a simple face detector for cropping (could use MediaPipe/YuNet)
    # For now, use center crop as fallback
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_stride != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        crop = frame[h//4:3*h//4, w//4:3*w//4]
        if crop.size > 0:
            face_crops.append(crop)

        if max_frames and len(face_crops) >= max_frames:
            break
        frame_idx += 1

    cap.release()

    if not face_crops:
        return VisualFeatures()

    extractor = VisualFeatureExtractor(
        backbone=backbone,
        device=device,
        deep_feature_dim=deep_feature_dim,
    )
    return extractor.extract_from_crops(face_crops)