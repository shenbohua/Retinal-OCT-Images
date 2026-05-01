from __future__ import annotations

"""Traditional feature extraction: HOG, LBP, and SIFT-BoVW."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

from src.preprocess import preprocess_for_features


def extract_hog_vector(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (16, 16),
    cells_per_block: tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Extract a single HOG feature vector."""
    return hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def extract_lbp_histogram(
    image: np.ndarray,
    radius: int = 2,
    points: int = 16,
) -> np.ndarray:
    """Extract normalized uniform-LBP histogram."""
    lbp = local_binary_pattern(image, P=points, R=radius, method="uniform")
    bins = np.arange(0, points + 3)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, points + 2), density=True)
    return hist.astype(np.float32)


def compute_hog_matrix(
    image_paths: Sequence[str | Path],
    image_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Compute HOG matrix for a list of image paths."""
    vectors = []
    for path in tqdm(image_paths, desc="HOG", leave=False):
        image = preprocess_for_features(path, size=image_size, normalize=False)
        vectors.append(extract_hog_vector(image))
    return np.vstack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)


def compute_lbp_matrix(
    image_paths: Sequence[str | Path],
    image_size: tuple[int, int] = (224, 224),
    radius: int = 2,
    points: int = 16,
) -> np.ndarray:
    """Compute LBP histogram matrix for a list of image paths."""
    vectors = []
    for path in tqdm(image_paths, desc="LBP", leave=False):
        image = preprocess_for_features(path, size=image_size, normalize=False)
        vectors.append(extract_lbp_histogram(image, radius=radius, points=points))
    return np.vstack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)


@dataclass(frozen=True)
class SIFTBOVWConfig:
    vocab_size: int = 128
    max_images_for_codebook: int = 5000
    max_descriptors: int = 250000
    image_size: tuple[int, int] = (224, 224)
    random_state: int = 42


class SIFTBOVWExtractor:
    """Two-stage SIFT + Bag-of-Visual-Words extractor."""

    def __init__(self, config: SIFTBOVWConfig | None = None) -> None:
        self.config = config or SIFTBOVWConfig()
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "OpenCV SIFT is not available. Install `opencv-contrib-python`."
            )
        self.sift = cv2.SIFT_create()
        self.kmeans: MiniBatchKMeans | None = None

    def _descriptors_for_image(self, path: str | Path) -> np.ndarray | None:
        """Return SIFT descriptors for one image; `None` when no keypoints found."""
        image = preprocess_for_features(path, size=self.config.image_size, normalize=False)
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        if keypoints is None or descriptors is None or len(descriptors) == 0:
            return None
        return descriptors.astype(np.float32)

    def fit(self, image_paths: Sequence[str | Path]) -> "SIFTBOVWExtractor":
        """Build visual vocabulary (k-means codebook) from training descriptors."""
        sampled_paths = list(image_paths)[: self.config.max_images_for_codebook]
        descriptor_chunks: list[np.ndarray] = []
        total_desc = 0

        for path in tqdm(sampled_paths, desc="SIFT-fit", leave=False):
            descriptors = self._descriptors_for_image(path)
            if descriptors is None:
                continue
            descriptor_chunks.append(descriptors)
            total_desc += descriptors.shape[0]
            if total_desc >= self.config.max_descriptors:
                break

        if not descriptor_chunks:
            raise ValueError("No SIFT descriptors collected. Check image quality/preprocessing.")

        all_desc = np.vstack(descriptor_chunks)
        if len(all_desc) > self.config.max_descriptors:
            all_desc = all_desc[: self.config.max_descriptors]

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.config.vocab_size,
            random_state=self.config.random_state,
            batch_size=4096,
            n_init="auto",
        )
        self.kmeans.fit(all_desc)
        return self

    def transform(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        """Map each image to normalized BoVW histogram using learned codebook."""
        if self.kmeans is None:
            raise RuntimeError("SIFTBOVWExtractor must be fitted before transform.")

        features = []
        k = self.config.vocab_size
        for path in tqdm(image_paths, desc="SIFT-transform", leave=False):
            descriptors = self._descriptors_for_image(path)
            hist = np.zeros(k, dtype=np.float32)
            if descriptors is not None:
                words = self.kmeans.predict(descriptors)
                hist = np.bincount(words, minlength=k).astype(np.float32)
                norm = np.linalg.norm(hist, ord=1)
                if norm > 0:
                    hist = hist / norm
            features.append(hist)

        return np.vstack(features) if features else np.empty((0, k), dtype=np.float32)

    def fit_transform(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        """Convenience wrapper to fit codebook and transform in one call."""
        return self.fit(image_paths).transform(image_paths)

    def save(self, path: str | Path) -> None:
        """Persist codebook and config."""
        if self.kmeans is None:
            raise RuntimeError("No codebook to save. Call fit first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": self.config, "kmeans": self.kmeans}, path)

    @classmethod
    def load(cls, path: str | Path) -> "SIFTBOVWExtractor":
        """Load a persisted BoVW extractor."""
        data = joblib.load(path)
        extractor = cls(config=data["config"])
        extractor.kmeans = data["kmeans"]
        return extractor


def save_feature_cache(path: str | Path, X: np.ndarray, y: np.ndarray, filepaths: Sequence[str]) -> None:
    """Save extracted features and labels to compressed NPZ cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, filepaths=np.array(filepaths))


def load_feature_cache(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load features, labels, and filepaths from NPZ cache."""
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["filepaths"]
