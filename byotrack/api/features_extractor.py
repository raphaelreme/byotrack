from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import numba  # type: ignore
import torch

import byotrack  # pylint: disable=cyclic-import


class FeaturesExtractor(ABC):
    """Abstract class for features extraction from an image for each detections"""

    @abstractmethod
    def __call__(self, frame: np.ndarray, detections: byotrack.Detections) -> torch.Tensor:
        """Extract the features for each detection given from the current frame

        Args:
            frame (np.ndarray): The video frame
                Shape: ([D, ]H, W, C)
            detections (byotrack.Detections): Detected objects on the frame, for which features should be computed

        Returns:
            torch.Tensor: d features for each detection.
                Shape: (n_dets, d), dtype: float32

        """

    def register(self, frame: np.ndarray, detections: byotrack.Detections) -> None:
        """Register the features inside the detections data.

        It is up to the caller to unregister the features if needed. (It may increase memory usage)

        Args:
            frame (np.ndarray): The video frame
                Shape: ([D, ]H, W, C)
            detections (byotrack.Detections): Detected objects on the frame, for which features should be computed

        """
        detections.data["features"] = self(frame, detections)


class MultiFeaturesExtractor(FeaturesExtractor):
    """Merge multiple features extractors by concatenating the features

    It is useful to test several features together quickly. But this will probably lead
    to sub-optimal computational performances when computing the features.
    """

    def __init__(self, features_extractors: Iterable[FeaturesExtractor]) -> None:
        self.features_extractors = features_extractors

    def __call__(self, frame: np.ndarray, detections: byotrack.Detections) -> torch.Tensor:
        features = [extractor(frame, detections) for extractor in self.features_extractors]
        return torch.cat(features, dim=-1)


class MassExtractor(FeaturesExtractor):
    """Extract the mass of each detection (number of pixels)"""

    def __call__(self, frame: np.ndarray, detections: byotrack.Detections):
        return detections.mass


class IntensityExtractor(FeaturesExtractor):
    """Extract the sum of intensities of each detection"""

    def __call__(self, frame: np.ndarray, detections: byotrack.Detections):
        torch.tensor(compute_intensity(detections.segmentation.numpy(), frame.sum(axis=-1)), dtype=torch.float32)


@numba.njit(cache=byotrack.NUMBA_CACHE)
def compute_intensity(segmentation: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Extract the cumulated intensity of each detection

    Args:
        segmentation (np.ndarray): Segmentation mask
        frame (np.ndarray): Video frame (should have the same number of pixels than segmentation)

    Returns:
        np.ndarray: Sum of intensity for each object

    """
    n = segmentation.max()
    intensity = np.zeros(n, dtype=frame.dtype)

    # Ravel in 1D
    segmentation = segmentation.reshape(-1)
    frame = frame.reshape(-1)

    for i in range(segmentation.shape[0]):
        instance = segmentation[i] - 1
        if instance != -1:
            intensity[instance] += frame[i]

    return intensity
