from __future__ import annotations

import sys
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import tqdm.auto as tqdm

import byotrack

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import numpy as np

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class Linker(ABC):
    """Base class for linkers in videos.

    A linker solves the tracking as the optimal links of detections through time.
    """

    @abstractmethod
    def run(
        self,
        video: Sequence[np.ndarray] | np.ndarray | None,
        detections_sequence: Sequence[byotrack.DetectionsLike],
    ) -> Collection[byotrack.Track]:
        """Run the linker on a whole sequence.

        Args:
            video (Sequence[np.ndarray] | np.ndarray | None): Optional video sequence of T frames.
                Each frame (array) is expected to have a shape ([D, ]H, W, C). Some linkers may
                not require any video. In that case, you may provide explicitly None.
            detections_sequence (Sequence[byotrack.DetectionsLike]): Detections are expected for
                for each frame of the video (sorted in time). Note that for a given frame, the
                Detections can be empty (i.e. Detections.length = 0).

        Returns:
            Collection[byotrack.Track]: Tracks for the given detections.

        """


class OnlineLinker(Linker):
    """Online linker. Build tracks one frame at a time.

    The linking algorithm is dividing into three steps:

    * :meth:`reset`: Reset the algorithm.
    * :meth:`update`: Update the algorithm with a new frame and its detections.
    * :meth:`collect`: Collect the constructed tracks up to the last frame.

    """

    progress_bar_description = "Linking"

    def reset(self, dim=2) -> None:
        """Reset the linking algorithm.

        Flush all data stored from a previous linking and prepare a new linking.

        Args:
            dim (int): The dimension of the data.
                Default: 2

        """

    @abstractmethod
    def update(self, frame: np.ndarray | None, detections: byotrack.Detections) -> None:
        """Progress in the linking step by one frame.

        Will update the internal algorithm by a single frame and its detections.

        Args:
            frame (np.ndarray | None): Optional frame of the video.
                Shape: ([D, ]H, W, C), dtype: float
            detections (byotrack.Detections): Detections for the given frame.
        """

    @abstractmethod
    def collect(self) -> Collection[byotrack.Track]:
        """Collect and build all the tracks up to the last given frame.

        Returns:
            Collection[byotrack.Track]: Tracks from the last reset to the last given frame.

        """

    @override
    def run(
        self,
        video: Sequence[np.ndarray] | np.ndarray | None,
        detections_sequence: Sequence[byotrack.DetectionsLike],
    ) -> Collection[byotrack.Track]:
        if video is not None and byotrack.video.video_length(video) != len(detections_sequence):
            if byotrack.video.video_length(video) < len(detections_sequence):
                warnings.warn(
                    f"""Found less frames ({len(video)}) than Detections ({len(detections_sequence)}).

                    By default we assume that the first Detections are aligned with the first video frame.
                    Tracking will be stopped with the last video frame.
                    """,
                    stacklevel=2,
                )
                detections_sequence = detections_sequence[: len(video)]

            if byotrack.video.video_length(video) > len(detections_sequence):
                warnings.warn(
                    f"""Found more frames ({len(video)}) than Detections ({len(detections_sequence)}).

                    By default we assume that the first Detections are aligned with the first video frame.
                    Tracking will be stopped with the last Detections.
                    """,
                    stacklevel=2,
                )
                video = video[: len(detections_sequence)]

        if len(detections_sequence) == 0:
            return []

        progress_bar = tqdm.tqdm(desc=self.progress_bar_description, total=len(detections_sequence))

        for frame_id, detections in enumerate(detections_sequence):
            _detections = byotrack.as_detections(detections)

            if frame_id == 0:  # Let's reset
                self.reset(_detections.dim)

            self.update(video[frame_id] if video is not None else None, _detections)
            progress_bar.update()

        progress_bar.close()

        tracks = self.collect()

        # Check produced tracks
        byotrack.Track.check_tracks(tracks, warn=True)

        return tracks
