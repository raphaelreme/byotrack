from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class VideoPreprocessor(ABC):
    """Video preprocessor base class.

    A preprocessor can both change the intensity (normalization, denoising, frame filtering, ...)
    as well as the shape (Z-projection, channel aggregation, slicing, ...) of each video frame.

    It first needs to be initialized on the given video (where it can read the video if needed).
    Then, it is applied online, i.e., for each frame of the video.

    Attributes:
        shape (tuple[int, ...]): Output shape of each frame ([D, ]H, W, C).
        dtype (np.dtype): Output dtype of each frame. Usually independent of the input video.

    """

    def __init__(self):
        super().__init__()
        self._shape: tuple[int, ...] | None = None
        self._dtype: np.dtype = np.dtype(np.float32)

    @property
    def shape(self) -> tuple[int, ...]:  # noqa: D102
        if self._shape is None:
            raise RuntimeError("VideoPreprocessor is not initialized yet.")

        return self._shape

    @property
    def dtype(self) -> np.dtype:  # noqa: D102
        return self._dtype

    def initialize(self, video: Sequence[np.ndarray] | np.ndarray) -> None:
        """Initialize the preprocessor for the given video.

        The default implementation preserve the video `shape` and `dtype`.
        This should be overwritten by VideoProcessor implementations.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): The video to preprocess.
                Sequence of T frames (array). Each array is expected to have a shape ([D, ]H, W, C).

        """
        if len(video) == 0:
            raise ValueError("No frame found in the video.")

        if hasattr(video, "shape"):
            self._shape = video.shape[1:]
        else:
            self._shape = video[0].shape

        if np.prod(self._shape) == 0:
            raise ValueError("No pixel found in the video.")

        if hasattr(video, "dtype"):
            self._dtype = video.dtype
        else:
            self._dtype = video[0].dtype

    @abstractmethod
    def preprocess_frame(self, frame: np.ndarray, frame_id=0) -> np.ndarray:
        """Preprocess the given frame.

        Args:
            frame (np.ndarray): Frame to be preprocessed.
                Shape: ([D, ]H, W, C)
            frame_id (int): Optional index of the frame in the video.
                Default to 0.

        Returns:
            np.ndarray: Preprocessed frame.
                Shape ([D', ]H', W', C')
        """

    def preprocess_video(self, video: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """Preprocess the given video directly.

        It will re-initialize the preprocessor at each call.

        Warning: Consider using the online version, which is integrated into the Video class.
                 This will requires much more memory than its online counterpart.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): The video to preprocess.
                Sequence of T frames (array). Each array is expected to have a shape ([D, ]H, W, C).

        Returns:
            np.ndarray: The preprocessed video loaded as a np.ndarray.
        """
        # Initialize for this video
        self.initialize(video)

        output = np.empty((len(video), *self.shape), dtype=self.dtype)
        for frame_id, frame in enumerate(video):
            output[frame_id] = self.preprocess_frame(frame, frame_id)

        return output
