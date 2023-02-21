from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection, List

import numpy as np
import tqdm

from ..parameters import ParametrizedObjectMixin
from ..video.reader import VideoReader
from .detections import Detections


class Detector(ABC, ParametrizedObjectMixin):  # pylint: disable=too-few-public-methods
    """Base class for detections in videos

    Detect on each frame the objects of interest.

    Each detector can define a set of parameters (See `ParametrizedObjectMixin`)
    """

    @abstractmethod
    def run(self, video: VideoReader) -> Collection[Detections]:
        """Run the detector on a whole video

        Args:
            video (byotrack.VideoReader): Input video

        Returns:
            Collection[byotrack.Detections]: Detections for each frame (ordered by frames)
        """


class BatchDetector(Detector):
    """Abstract detector that performs detection directly by batch

    Usually leads to faster implementation of the detection process
    when batch size is greater than 1

    Attrs:
        batch_size (int): Size of the frame batch
    """

    progress_bar_description = "Detections"

    def __init__(self, batch_size=20) -> None:
        super().__init__()
        self.batch_size = batch_size

    def run(self, video: VideoReader) -> List[Detections]:
        frame_id = video.tell()
        video.seek(0)
        detections_sequence: List[Detections] = []
        batch = []

        progress_bar = tqdm.tqdm(desc=self.progress_bar_description, total=video.length)
        has_next = True
        while has_next:
            frame, has_next = video.read()
            batch.append(frame[None, ...])

            if len(batch) >= self.batch_size or not has_next:
                detections_sequence.extend(self.detect(np.concatenate(batch, axis=0)))
                progress_bar.update(len(batch))
                batch = []

        # Reset the video at the initial frame
        video.seek(frame_id)

        # Set frames
        for i, detections in enumerate(detections_sequence):
            detections.frame = i

        return detections_sequence

    @abstractmethod
    def detect(self, batch: np.ndarray) -> Collection[Detections]:
        """Apply the detection on a batch of frames

        The frame id of each detections is set afterward by the BatchDetector `run` method

        Args:
            batch (np.ndarray): Batch of video frames
                Shape: (B, H, W, 3) or (B, H, W, 1) (Grayscale)

        Returns:
            Collection[byotrack.Detections]: Detections for each given frame
        """
