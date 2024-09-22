from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection, Sequence, Union
import warnings

import numpy as np
import tqdm.auto as tqdm

import byotrack  # pylint: disable=cyclic-import
from .parameters import ParametrizedObjectMixin


class Linker(ABC, ParametrizedObjectMixin):  # pylint: disable=too-few-public-methods
    """Base class for linkers in videos

    Link detections through time to build tracks.

    Each linker can define a set of parameters (See `ParametrizedObjectMixin`)

    """

    @abstractmethod
    def run(
        self, video: Union[Sequence[np.ndarray], np.ndarray], detections_sequence: Sequence[byotrack.Detections]
    ) -> Collection[byotrack.Track]:
        """Run the linker on a whole video

        Args:
            video (Sequence[np.ndarray] | np.ndarray): Sequence of T frames (array).
                Each array is expected to have a shape ([D, ]H, W, C)
            detections_sequence (Sequence[byotrack.Detections]): Detections for each frame
                Detections is expected for each frame of the video, in the same order.
                (Note that for a given frame, the Detections can be empty)

        Returns:
            Collection[byotrack.Track]: Tracks of particles

        """


class OnlineLinker(Linker):
    """Online linker, it tracks particles one frame at a time

    The linking algorithm is dividing into three steps:

    * reset: Reset the algorithm
    * update: Update the algorithm with a new frame and detections
    * collect: Collect the constructed tracks up to the last frame

    """

    progress_bar_description = "Linking"

    def reset(self) -> None:
        """Reset the linking algorithm

        Flush all data stored from a previous linking and prepare a new linking
        """

    @abstractmethod
    def update(self, frame: np.ndarray, detections: byotrack.Detections) -> None:
        """Progress in the linking step by one frame

        Will update the internal algorithm by a single frame and its detections.

        Args:
            frame (np.ndarray): Frame of the video
                Shape: ([D, ]H, W, C), dtype: float
            detections (byotrack.Detections): Detections for the given frame

        """

    @abstractmethod
    def collect(self) -> Collection[byotrack.Track]:
        """Collect and build all the tracks up to the last given frame

        Returns:
            Collection[byotrack.Track]: Tracks from the last reset to the last given frame.

        """

    def run(
        self, video: Union[Sequence[np.ndarray], np.ndarray], detections_sequence: Sequence[byotrack.Detections]
    ) -> Collection[byotrack.Track]:
        if len(video) != len(detections_sequence):
            warnings.warn(
                f"""Expected to have one Detections for each frame of the video.

            There are {len(detections_sequence)} Detections for {len(video)} frames.
            This can lead to unexpected behavior. By default we assume that the first Detections
            is aligned with the first frame and stop when the end of shortest sequence is reached.
            """
            )
        self.reset()

        progress_bar = tqdm.tqdm(desc=self.progress_bar_description, total=min(len(video), len(detections_sequence)))

        for frame, detections in zip(video, detections_sequence):
            self.update(frame, detections)
            progress_bar.update()

        progress_bar.close()

        return self.collect()
