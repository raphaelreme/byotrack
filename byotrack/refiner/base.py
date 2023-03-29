from abc import ABC, abstractmethod
from typing import Collection, Iterable

import numpy as np

from ..parameters import ParametrizedObjectMixin
from ..tracks import Track


class Refiner(ABC, ParametrizedObjectMixin):  # pylint: disable=too-few-public-methods
    """Base class for tracks refiners

    Improve a set of tracks

    Each refiner can define a set of parameters (See `ParametrizedObjectMixin`)

    """

    @abstractmethod
    def run(self, video: Iterable[np.ndarray], tracks: Collection[Track]) -> Collection[Track]:
        """Run the refiner on a whole video

        Args:
            video (Iterable[np.ndarray]): Sequence of frames (video)
                Each array is expected to have a shape (H, W, C)
            tracks (Collection[Track]): Tracks of particles

        Returns:
            Collection[Track]: Refined tracks of particles

        """


class MultiRefiner(Refiner):  # pylint: disable=too-few-public-methods
    """Refine tracks with multiple refiners

    Attributes:
        refiners (Iterable[Refiner]): Refiners to run
    """

    def __init__(self, refiners: Iterable[Refiner]) -> None:
        super().__init__()
        self.refiners = refiners

    def run(self, video: Iterable[np.ndarray], tracks: Collection[Track]) -> Collection[Track]:
        for refiner in self.refiners:
            tracks = refiner.run(video, tracks)

        return tracks
