from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Union

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
