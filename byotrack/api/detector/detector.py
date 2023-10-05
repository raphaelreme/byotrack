from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection, Iterable, List

import numpy as np
import tqdm

import byotrack  # pylint: disable=cyclic-import
from ..parameters import ParametrizedObjectMixin


class Detector(ABC, ParametrizedObjectMixin):  # pylint: disable=too-few-public-methods
    """Base class for detections in videos

    Detect on each frame the objects of interest.

    Each detector can define a set of parameters (See `ParametrizedObjectMixin`)

    """

    @abstractmethod
    def run(self, video: Iterable[np.ndarray]) -> Collection[byotrack.Detections]:
        """Run the detector on a whole video

        Args:
            video (Iterable[np.ndarray]): Sequence of frames (video)
                Each array is expected to have a shape (H, W, C)

        Returns:
            Collection[byotrack.Detections]: Detections for each frame (ordered by frames)

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
            Default: True

    """

    progress_bar_description = "Detections"

    def __init__(self, batch_size=20, add_true_frames=True) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.add_true_frames = add_true_frames

    def run(self, video: Iterable[np.ndarray]) -> List[byotrack.Detections]:
        reader = None
        if self.add_true_frames and isinstance(video, byotrack.Video):
            reader = video.reader

        detections_sequence: List[byotrack.Detections] = []
        batch: List[np.ndarray] = []
        frame_ids: List[int] = []

        for i, frame in enumerate(tqdm.tqdm(video, self.progress_bar_description, miniters=self.batch_size)):
            batch.append(frame[None, ...])
            frame_ids.append(reader.tell() if reader else i)

            if len(batch) >= self.batch_size:
                detections_sequence.extend(self.detect(np.concatenate(batch, axis=0)))
                batch = []

        if batch:
            detections_sequence.extend(self.detect(np.concatenate(batch, axis=0)))

        # Set frame ids
        for frame_id, detections in zip(frame_ids, detections_sequence):
            detections.frame_id = frame_id

        return detections_sequence

    @abstractmethod
    def detect(self, batch: np.ndarray) -> Collection[byotrack.Detections]:
        """Apply the detection on a batch of frames

        By default, the frame ids are set from 0 to n-1 with n the size of the batch.
        The aggregattion of batches and frame ids correction is automatically handled when called
        the `run` method.

        Args:
            batch (np.ndarray): Batch of video frames
                Shape: (B, H, W, C)

        Returns:
            Collection[byotrack.Detections]: Detections for each given frame

        """
