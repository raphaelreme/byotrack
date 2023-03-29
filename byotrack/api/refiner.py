from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection, Iterable

import numpy as np

import byotrack  # pylint: disable=cyclic-import
from .parameters import ParametrizedObjectMixin


class Refiner(ABC, ParametrizedObjectMixin):  # pylint: disable=too-few-public-methods
    """Base class for tracks refiners

    Improve a set of tracks

    Each refiner can define a set of parameters (See `ParametrizedObjectMixin`)

    """

    @abstractmethod
    def run(self, video: Iterable[np.ndarray], tracks: Collection[byotrack.Track]) -> Collection[byotrack.Track]:
        """Run the refiner on a whole video

        Args:
            video (Iterable[np.ndarray]): Sequence of frames (video)
                Each array is expected to have a shape (H, W, C)
            tracks (Collection[byotrack.Track]): Tracks of particles

        Returns:
            Collection[byotrack.Track]: Refined tracks of particles

        """
