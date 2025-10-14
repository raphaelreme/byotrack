from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

import numpy as np
import tqdm.auto as tqdm

import byotrack  # pylint: disable=cyclic-import
from ..parameters import ParametrizedObjectMixin


class Detector(ABC, ParametrizedObjectMixin):  # pylint: disable=too-few-public-methods
    """Base class for detections in videos

    Detect on each frame the objects of interest.

    Each detector can define a set of parameters (See `ParametrizedObjectMixin`)

    """

    @abstractmethod
    def run(self, video: Union[Sequence[np.ndarray], np.ndarray]) -> Sequence[byotrack.Detections]:
        """Run the detector on a whole video

        Args:
            video (Sequence[np.ndarray] | np.ndarray): Sequence of T frames (array).
                Each array is expected to have a shape ([D, ]H, W, C)

        Returns:
            Sequence[byotrack.Detections]: Detections for each frame (ordered by frames)

        """


class BatchDetector(Detector):
    """Abstract detector that performs detection directly by batch

    Usually leads to faster implementation of the detection process
    when batch size is greater than 1

    Attributes:
        batch_size (int): Size of the frame batch
            Default: 20
        add_true_frames (bool): If the input is a Video, it will exploits the VideoReader knowledge
            to extract the true frame id for each detections.
            If False, it will register the index of the frame as the frame_id
            Default: False

    """

    progress_bar_description = "Detections"

    def __init__(self, batch_size=20, add_true_frames=False) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.add_true_frames = add_true_frames

    def run(self, video: Union[Sequence[np.ndarray], np.ndarray]) -> List[byotrack.Detections]:
        if len(video) == 0:
            return []

        reader = None
        if self.add_true_frames and isinstance(video, byotrack.Video):
            reader = video.reader

        detections_sequence: List[byotrack.Detections] = []
        frame_ids: List[int] = []
        progress_bar = tqdm.tqdm(desc=self.progress_bar_description, total=len(video))

        first = video[0]
        batch = np.zeros((self.batch_size, *first.shape), dtype=first.dtype)
        batch[0] = first
        n = 1
        frame_ids.append(reader.tell() if reader else 0)

        for frame in video[1:]:
            if n >= self.batch_size:
                detections_sequence.extend(self.detect(batch))
                progress_bar.update(n)
                n = 0

            batch[n] = frame
            frame_ids.append(reader.tell() if reader else len(frame_ids))
            n += 1

        detections_sequence.extend(self.detect(batch[:n]))
        progress_bar.update(n)
        progress_bar.close()

        # Set frame ids
        for frame_id, detections in zip(frame_ids, detections_sequence):
            detections.frame_id = frame_id

        return detections_sequence

    @abstractmethod
    def detect(self, batch: np.ndarray) -> Sequence[byotrack.Detections]:
        """Apply the detection on a batch of frames

        By default, the frame ids are set from 0 to n-1 with n the size of the batch.
        The aggregattion of batches and frame ids correction is automatically handled when called
        the `run` method.

        Args:
            batch (np.ndarray): Batch of video frames
                Shape: (B, [D, ]H, W, C)

        Returns:
            Sequence[byotrack.Detections]: Detections for each given frame

        """


class DetectionsRefiner(ABC):
    """Abstract class for method to improve a coarse detection result

    This assumes the refining can be done independently for each frame.

    Warning: The class is still experimental and may change in future versions

    """

    progress_bar_description = "Detections Refiner"

    @abstractmethod
    def apply(self, detections: byotrack.Detections, frame: Optional[np.ndarray] = None) -> byotrack.Detections:
        """Refines Detections of the given frame

        Args:
            detections (byotrack.Detections): Detections to refine
            frame (Optional[np.ndarray]): The associated frame of the video (optional)
                Shape: ([D, ]H, W, C)
                Default: None

        Returns:
            byotrack.Detections: Refined detections
        """

    def run(
        self,
        detections_sequence: Sequence[byotrack.Detections],
        video: Union[Sequence[np.ndarray], np.ndarray, None] = None,
    ) -> List[byotrack.Detections]:
        """Run the refiner on a full video / detections sequence

        Args:
            detections_sequence (Sequence[byotrack.Detections]): Sequence of T detections to refine
            video (Sequence[np.ndarray] | np.ndarray | None): Optional corresponding sequence of T frames.
                Each frame is expected to have a shape ([D, ]H, W, C)

        Returns:
            Sequence[byotrack.Detections]: Sequence of the T refined detections
        """
        # XXX: Could run with MP? (depends if apply itself is MP)
        refined_detections = []
        for i, detections in enumerate(tqdm.tqdm(detections_sequence, desc=self.progress_bar_description)):
            refined_detections.append(self.apply(detections, video[i] if video is not None else None))

        return refined_detections
