from abc import ABC, abstractmethod
from typing import Collection

from ..parameters import ParametrizedObjectMixin
from ..tracks import Track
from ..video.reader import VideoReader


class Refiner(ABC, ParametrizedObjectMixin):  # pylint: disable=too-few-public-methods
    """Base class for tracks refiners

    Improve a set of tracks

    Each refiner can define a set of parameters (See `ParametrizedObjectMixin`)
    """

    @abstractmethod
    def run(self, video: VideoReader, tracks: Collection[Track]) -> Collection[Track]:
        """Run the refiner on a whole video

        Args:
            video (byotrack.VideoReader): Input video
            tracks (Collection[Track]): Tracks of particles

        Returns:
            Collection[Track]: Refined tracks of particles
        """
