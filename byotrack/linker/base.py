from abc import ABC, abstractmethod
from typing import Collection

from ..detector.detections import Detections
from ..parameters import ParametrizedObjectMixin
from ..tracks import Track
from ..video.reader import VideoReader


class Linker(ABC, ParametrizedObjectMixin):  # pylint: disable=too-few-public-methods
    """Base class for linkers in videos

    Link detections through time to build tracks.

    Each linker can define a set of parameters (See `ParametrizedObjectMixin`)
    """

    @abstractmethod
    def run(self, video: VideoReader, detections_sequence: Collection[Detections]) -> Collection[Track]:
        """Run the linker on a whole video

        Args:
            video (byotrack.VideoReader): Input video
            detections_sequence (Collection[Detections]): Detections for each frame

        Returns:
            Collection[byotrack.Track]: Tracks of particles
        """
