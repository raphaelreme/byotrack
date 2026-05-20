from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from byotrack.video.preprocessor import preprocessor
from byotrack.video.reader import slice_length

if TYPE_CHECKING:
    from collections.abc import Sequence

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class FrameSlicer(preprocessor.VideoPreprocessor):
    """Slice the frame with given slices.

    Attributes:
        slices (tuple[slice, ...]): slices to apply.
    """

    def __init__(self, slices: tuple[slice, ...]):
        super().__init__()
        self.slices = slices

    @override
    def initialize(self, video: Sequence[np.ndarray] | np.ndarray) -> None:
        """Initialize the preprocessor for the given video.

        This will update the `shape` attribute to reflect the output shape after slicing.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): The video to preprocess.
                Sequence of T frames (array). Each array is expected to have a shape ([D, ]H, W, C).

        """
        super().initialize(video)

        if len(self.slices) > len(self.shape):
            raise IndexError("Too many indices for video.")

        self._shape = tuple(
            slice_length(self.slices[i], shape_) if i < len(self.slices) else shape_
            for i, shape_ in enumerate(self.shape)
        )

        if np.prod(self._shape) == 0:
            raise ValueError("Slice is empty.")

    @override
    def preprocess_frame(self, frame: np.ndarray, frame_id=0) -> np.ndarray:
        return frame[self.slices]

    @override
    def preprocess_video(self, video: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        if not isinstance(video, np.ndarray):
            return super().preprocess_video(video)

        slices = (slice(None), *self.slices)
        return video[slices]
