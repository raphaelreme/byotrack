from abc import ABC, abstractmethod
from typing import Collection, Optional

from .detector.base import Detector
from .linker.base import Linker
from .parameters import ParametrizedObjectMixin
from .refiner.base import Refiner
from .tracks import Track
from .video.reader import VideoReader


class Tracker(ABC, ParametrizedObjectMixin):  # pylint: disable=too-few-public-methods
    """Base abstract tracker class

    A tracker can be run on a whole video and produces tracks
    """

    @abstractmethod
    def run(self, video: VideoReader, tracks: Collection[Track]) -> Collection[Track]:
        """Run the tracker on a whole video

        Args:
            video (byotrack.VideoReader): Input video

        Returns:
            Collection[Track]: Tracks of particles
        """


class MultiStepTracker(Tracker):  # pylint: disable=too-few-public-methods
    """Multi step tracker: split the tracking task into Detect / Link / Refine

    Attrs:
        detector (byotrack.Detector): Performs the detection on the video
        linker (byotrack.Linker): Links detections through time
        refiner (Optional[byotrack.Refiner]): Refines tracks
            No refining if non-given.
    """

    def __init__(self, detector: Detector, linker: Linker, refiner: Optional[Refiner] = None) -> None:
        super().__init__()

        self.detector = detector
        self.linker = linker
        self.refiner = refiner

    def run(self, video: VideoReader, tracks: Collection[Track]) -> Collection[Track]:
        detections = self.detector.run(video)
        tracks = self.linker.run(video, detections)
        if self.refiner is not None:
            tracks = self.refiner.run(video, tracks)

        return tracks
