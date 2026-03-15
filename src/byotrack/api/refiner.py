from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import numpy as np

    import byotrack


class Refiner(ABC):
    """Base class for tracks refiners.

    Improve a set of tracks

    Each refiner can define a set of parameters (See `ParametrizedObjectMixin`)

    """

    @abstractmethod
    def run(
        self, video: Sequence[np.ndarray] | np.ndarray, tracks: Collection[byotrack.Track]
    ) -> Collection[byotrack.Track]:
        """Run the refiner on a whole video.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): Sequence of T frames (array).
                Each array is expected to have a shape ([D, ]H, W, C)
            tracks (Collection[byotrack.Track]): Tracks of particles

        Returns:
            Collection[byotrack.Track]: Refined tracks of particles

        """
