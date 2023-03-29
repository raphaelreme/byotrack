from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection, Iterable

import numpy as np

import byotrack  # pylint: disable=cyclic-import
from .parameters import ParametrizedObjectMixin


class Tracker(ABC, ParametrizedObjectMixin):  # pylint: disable=too-few-public-methods
    """Base abstract tracker class

    A tracker can be run on a whole video and produces tracks

    """

    @abstractmethod
    def run(self, video: Iterable[np.ndarray]) -> Collection[byotrack.Track]:
        """Run the tracker on a whole video

        Args:
            video (Iterable[np.ndarray]): Sequence of frames (video)
                Each array is expected to have a shape (H, W, C)

        Returns:
            Collection[byotrack.Track]: Tracks of particles

        """


class MultiStepTracker(Tracker):  # pylint: disable=too-few-public-methods
    """Multi step tracker: split the tracking task into Detect / Link / Refine

    Attributes:
        detector (byotrack.Detector): Performs the detection on the video
        linker (byotrack.Linker): Links detections through time
        refiners (Iterable[byotrack.Refiner]): Optional refinement steps
            Empty by default

    """

    def __init__(
        self, detector: byotrack.Detector, linker: byotrack.Linker, refiners: Iterable[byotrack.Refiner] = ()
    ) -> None:
        super().__init__()

        self.detector = detector
        self.linker = linker
        self.refiners = refiners

    def run(self, video: Iterable[np.ndarray]) -> Collection[byotrack.Track]:
        detections = self.detector.run(video)
        tracks = self.linker.run(video, detections)
        for refiner in self.refiners:
            tracks = refiner.run(video, tracks)

        return tracks
