from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
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


# TODO: Add some examples
