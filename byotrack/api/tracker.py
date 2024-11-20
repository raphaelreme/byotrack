from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection, Iterable, List, Sequence, Union
import warnings

import numpy as np
import tqdm.auto as tqdm

import byotrack  # pylint: disable=cyclic-import
from .parameters import ParametrizedObjectMixin


class PauseableTQDM(tqdm.tqdm):  # pylint: disable=inconsistent-mro
    """Tqdm with a real pause mechanism"""

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        self.last_pause_t = 0.0

    def update(self, n: float | None = 1) -> bool | None:
        if self.last_pause_t != 0.0:
            warnings.warn("Progress bar is paused. You should explicitly call unpause before updating again")
            self.unpause()

        return super().update(n)

    def pause(self, refresh=True):
        if self.disable:
            return

        if self.last_pause_t != 0.0:
            warnings.warn("The progress bar was already paused")
            return

        if refresh:  # By default refresh before doing a pause
            self.refresh()

        self.last_pause_t = self._time()  # type: ignore

    def unpause(self):
        if self.disable:
            return

        if self.last_pause_t == 0.0:
            warnings.warn("The progress bar was not paused")
            return

        delta_t = self._time() - self.last_pause_t  # type: ignore
        self.last_pause_t = 0.0
        if delta_t < 0.0:
            warnings.warn("Found negative pause time, ignoring...")
            return

        self.start_t += delta_t
        self.last_print_t += delta_t


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

        # Check produced tracks
        byotrack.Track.check_tracks(tracks, warn=True)

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

        detect_bar = PauseableTQDM(desc=self.detector.progress_bar_description, total=len(video))
        link_bar = PauseableTQDM(desc=self.linker.progress_bar_description, total=len(video))
        link_bar.pause()  # Starting with detections

        first = video[0]
        batch = np.zeros((self.detector.batch_size, *first.shape), dtype=first.dtype)
        batch[0] = first
        n = 1
        frame_ids.append(reader.tell() if reader else 0)

        for frame in video[1:]:
            if n >= self.detector.batch_size:
                detections_sequence = self.detector.detect(batch)
                detect_bar.update(n)

                # Switch from detection to linking
                detect_bar.pause()
                link_bar.unpause()

                for i in range(n):
                    detections_sequence[i].frame_id = frame_ids[i]
                    self.linker.update(batch[i], detections_sequence[i])
                    link_bar.update()

                # Switch from linking to detection
                link_bar.pause()
                detect_bar.unpause()

                frame_ids = []
                n = 0

            batch[n] = frame
            frame_ids.append(reader.tell() if reader else len(frame_ids))
            n += 1

        detections_sequence = self.detector.detect(batch[:n])
        detect_bar.update(n)
        detect_bar.close()

        link_bar.unpause()

        for i in range(n):
            detections_sequence[i].frame_id = frame_ids[i]
            self.linker.update(batch[i], detections_sequence[i])
            link_bar.update()

        link_bar.close()

        tracks = self.linker.collect()
        for refiner in self.refiners:
            tracks = refiner.run(video, tracks)

        # Check produced tracks
        byotrack.Track.check_tracks(tracks, warn=True)

        return tracks
