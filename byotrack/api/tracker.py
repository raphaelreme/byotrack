from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection, Iterable, List, Sequence, Union

import numpy as np
import tqdm.auto as tqdm

import byotrack  # pylint: disable=cyclic-import
from .parameters import ParametrizedObjectMixin


class Tracker(ABC, ParametrizedObjectMixin):  # pylint: disable=too-few-public-methods
    """Base abstract tracker class

    A tracker can be run on a whole video and produces tracks

    """

    @abstractmethod
    def run(self, video: Union[Sequence[np.ndarray], np.ndarray]) -> Collection[byotrack.Track]:
        """Run the tracker on a whole video

        Args:
            video (Sequence[np.ndarray] | np.ndarray): Sequence of T frames (array).
                Each array is expected to have a shape ([D, ]H, W, C)

        Returns:
            Collection[byotrack.Track]: Tracks of particles

        """


class MultiStepTracker(Tracker):  # pylint: disable=too-few-public-methods
    """Multi step tracker: split the tracking task into Detect / Link / Refine

    Attributes:
        detector (byotrack.Detector): Performs the detection on the video
        linker (byotrack.Linker): Links detections through time
        refiners (Sequence[byotrack.Refiner]): Optional refinement steps
            Empty by default

    """

    def __init__(
        self, detector: byotrack.Detector, linker: byotrack.Linker, refiners: Iterable[byotrack.Refiner] = ()
    ) -> None:
        super().__init__()

        self.detector = detector
        self.linker = linker
        self.refiners = refiners

    def run(self, video: Union[Sequence[np.ndarray], np.ndarray]) -> Collection[byotrack.Track]:
        detections = self.detector.run(video)
        tracks = self.linker.run(video, detections)
        for refiner in self.refiners:
            tracks = refiner.run(video, tracks)

        return tracks


class BatchMultiStepTracker(MultiStepTracker):  # pylint: disable=too-few-public-methods
    """Online Tracking with detect/link/refine framework

    It only works with BatchDetector and OnlineLinker: it reduces RAM usage and computations
    by loading frames only by batch. Each batch is processed by the detector followed
    by the linker. Once a batch is processed, a new one is loaded. The detections are never fully stored,
    neither the video.

    Tracks refining is done offline.

    Attributes:
        detector (byotrack.BatchDetector): Performs the detection on the video by batch
            It defines the batch size to use
        linker (byotrack.OnlineLinker): Links detections one frame at time
        refiners (Sequence[byotrack.Refiner]): Optional refinement steps
            Empty by default

    """

    def __init__(
        self, detector: byotrack.BatchDetector, linker: byotrack.OnlineLinker, refiners: Iterable[byotrack.Refiner] = ()
    ) -> None:
        super().__init__(detector, linker, refiners)
        self.detector: byotrack.BatchDetector
        self.linker: byotrack.OnlineLinker

    def run(self, video: Union[Sequence[np.ndarray], np.ndarray]) -> Collection[byotrack.Track]:
        if len(video) == 0:
            return []

        reader = None
        if self.detector.add_true_frames and isinstance(video, byotrack.Video):
            reader = video.reader

        self.linker.reset()

        detections_sequence: Sequence[byotrack.Detections] = []
        frame_ids: List[int] = []

        detect_bar = tqdm.tqdm(desc=self.detector.progress_bar_description, total=len(video))
        link_bar = tqdm.tqdm(desc=self.linker.progress_bar_description, total=len(video))

        first = video[0]
        batch = np.zeros((self.detector.batch_size, *first.shape), dtype=first.dtype)
        batch[0] = first
        n = 1
        frame_ids.append(reader.tell() if reader else 0)

        for frame in video[1:]:
            if n >= self.detector.batch_size:
                detections_sequence = self.detector.detect(batch)
                detect_bar.update(n)

                for i in range(n):
                    detections_sequence[i].frame_id = frame_ids[i]
                    self.linker.update(batch[i], detections_sequence[i])
                    link_bar.update()

                link_bar.refresh()

                frame_ids = []
                n = 0

            batch[n] = frame
            frame_ids.append(reader.tell() if reader else len(frame_ids))
            n += 1

        detections_sequence = self.detector.detect(batch[:n])
        detect_bar.update(n)

        for i in range(n):
            detections_sequence[i].frame_id = frame_ids[i]
            self.linker.update(batch[i], detections_sequence[i])
            link_bar.update()

        detect_bar.close()
        link_bar.close()

        tracks = self.linker.collect()
        for refiner in self.refiners:
            tracks = refiner.run(video, tracks)

        return tracks
