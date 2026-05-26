from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
import tqdm.auto as tqdm

import byotrack

if TYPE_CHECKING:
    from collections.abc import Sequence


if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


# TODO: add_refiners + _run to overwrite ? and run runs _run + refiners.
# then integrates with BatchDetector
class Detector(ABC):
    """Base class for detections in videos.

    Detect on each frame the objects of interest.

    """

    @abstractmethod
    def run(self, video: Sequence[np.ndarray] | np.ndarray) -> Sequence[byotrack.Detections]:
        """Run the detector on a whole video.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): Sequence of T frames (array).
                Each array is expected to have a shape ([D, ]H, W, C)

        Returns:
            Sequence[byotrack.Detections]: Detections for each frame (ordered by frames)

        """


class BatchDetector(Detector):
    """Abstract detector that performs detection directly by batch.

    Usually leads to faster implementation of the detection process
    when batch size is greater than 1.

    Attributes:
        batch_size (int): Size of the frame batch.
            Default: 20

    """

    progress_bar_description = "Detections"

    def __init__(self, batch_size: int = 20) -> None:
        super().__init__()
        self.batch_size = batch_size

    @override
    def run(self, video: Sequence[np.ndarray] | np.ndarray) -> list[byotrack.Detections]:
        if len(video) == 0:
            return []

        detections_sequence: list[byotrack.Detections] = []
        progress_bar = tqdm.tqdm(desc=self.progress_bar_description, total=len(video))

        first = video[0]
        batch = np.zeros((self.batch_size, *first.shape), dtype=first.dtype)
        batch[0] = first
        n = 1

        for frame in video[1:]:
            if n >= self.batch_size:
                detections_sequence.extend(self.detect(batch))
                progress_bar.update(n)
                n = 0

            batch[n] = frame
            n += 1

        detections_sequence.extend(self.detect(batch[:n]))
        progress_bar.update(n)
        progress_bar.close()

        return detections_sequence

    @abstractmethod
    def detect(self, batch: np.ndarray) -> Sequence[byotrack.Detections]:
        """Apply the detection on a batch of frames.

        Args:
            batch (np.ndarray): Batch of video frames
                Shape: (B, [D, ]H, W, C)

        Returns:
            Sequence[byotrack.Detections]: Detections for each given frame

        """


class DetectionsRefiner(ABC):
    """Abstract class for method to improve a coarse detection result.

    This assumes the refining can be done independently for each frame.

    Warning: The class is still experimental and may change in future versions

    """

    progress_bar_description = "Detections Refiner"

    @abstractmethod
    def apply(self, detections: byotrack.DetectionsLike, frame: np.ndarray | None = None) -> byotrack.Detections:
        """Refines Detections of the given frame.

        Args:
            detections (byotrack.DetectionsLike): Detections to refine
            frame (np.ndarray | None): The associated frame of the video (optional)
                Shape: ([D, ]H, W, C)
                Default: None

        Returns:
            byotrack.Detections: Refined detections
        """
        # NOTE: apply is responsible for converting into Detections with `as_detection`.
        #       This allows the user to use apply directly on a single np.array segmentation.

    def run(
        self,
        detections_sequence: Sequence[byotrack.DetectionsLike],
        video: Sequence[np.ndarray] | np.ndarray | None = None,
    ) -> list[byotrack.Detections]:
        """Run the refiner on a full video / detections sequence.

        Args:
            detections_sequence (Sequence[byotrack.DetectionsLike]): Sequence of T detections to refine
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


class GroundTruthDetector(BatchDetector):
    """Convert a video of segmentations into Detections using the BatchDetector API.

    Each frame in the video is expected to be an instance segmentation mask of shape ([D, ]H, W, 1)
    The video should not be normalized and each pixel is expected to be an integer.

    Note: The segmented video can be given at construction time of the detector. In such case,
          this video will be used over the one given in `run`. This allows to use precomputed detections
          as well as the original video in a [Batch]MultiStepTracker pipeline.

    Example:

    .. code-block:: python

        ### Example: Loading CTC segmentation
        video = byotrack.Video("dataset/01_ERR_SEG")  # Load segmentation for CLB
        # video = byotrack.Video("dataset/01_GT/SEG")  # Load ground-truth segmentation
        # video = byotrack.Video("dataset/01_RES/TRA")  # Load predicted tracks segmentation

        detector = GroundTruthDetector()
        detections_sequence = detector.run(video)

        ### Example: Loading precomputed segmentations
        segmentations = byotrack.Video("segmentations.tiff")  # Shape (T, [D, ]H, W, 1), dtype: int
        detections_sequence = GroundTruthDetector().run(segmentations)

        ### Example: Tracking with precomputed segmentations with online loading
        video = byotrack.Video("video.tiff").normalize()  # Shape (T, [D, ]H, W, 1), dtype: float
        segmentations = byotrack.Video("segmentations.tiff")  # Shape (T, [D, ]H, W, 1), dtype: int
        detector = GroundTruthDetector(segmentations)
        linker = ...
        tracker = BatchMultiStepTracker(detector, linker)

        # This will forward the segmented frame to the Detector,
        # but the video frame to the linker.
        tracks = tracker.run(video)


    """

    progress_bar_description = "Ground-Truth Detector (Load from segmented video)"

    def __init__(self, segmentations: Sequence[np.ndarray] | np.ndarray | None = None, batch_size=1):
        super().__init__(batch_size)
        self.segmentations = segmentations

    @override
    def run(self, video: Sequence[np.ndarray] | np.ndarray) -> list[byotrack.Detections]:
        if self.segmentations is None:
            return super().run(video)

        self._check_shape(video)

        return super().run(self.segmentations)

    def _check_shape(self, video: Sequence[np.ndarray] | np.ndarray) -> None:
        if self.segmentations is None:
            return

        video_shape = video.shape if hasattr(video, "shape") else (len(video), *video[0].shape)

        segmentations_shape = (
            self.segmentations.shape
            if hasattr(self.segmentations, "shape")
            else (len(self.segmentations), *self.segmentations[0].shape)
        )

        if video_shape[:-1] != segmentations_shape[:-1]:
            raise ValueError("Segmented video do not match video shape.")

    @override
    def detect(self, batch: np.ndarray) -> list[byotrack.SegmentationDetections]:
        if batch.shape[-1] != 1:
            raise ValueError("Multichannel segmentations are not supported")

        if not np.issubdtype(batch.dtype, np.integer):
            raise ValueError("GroundTruthDetector expects label (integer) encoded frames.")

        return [byotrack.SegmentationDetections(torch.from_numpy(frame[..., 0].astype(np.int32))) for frame in batch]
