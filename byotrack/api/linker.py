from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection, Iterable

import numpy as np

import byotrack  # pylint: disable=cyclic-import
from .parameters import ParametrizedObjectMixin


class Linker(ABC, ParametrizedObjectMixin):  # pylint: disable=too-few-public-methods
    """Base class for linkers in videos

    Link detections through time to build tracks.

    Each linker can define a set of parameters (See `ParametrizedObjectMixin`)

    """

    @abstractmethod
    def run(
        self, video: Iterable[np.ndarray], detections_sequence: Collection[byotrack.Detections]
    ) -> Collection[byotrack.Track]:
        """Run the linker on a whole video

        Args:
            video (Iterable[np.ndarray]): Sequence of frames (video)
                Each array is expected to have a shape (H, W, C)
            detections_sequence (Collection[byotrack.Detections]): Detections for each frame

        Returns:
            Collection[byotrack.Track]: Tracks of particles

        """
