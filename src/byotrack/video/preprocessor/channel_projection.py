from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal

import numpy as np

from byotrack.video.preprocessor import preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class ChannelProjection(preprocessor.VideoPreprocessor):
    """Projection of the video channel.

    Allows to reduce multi-channel videos into single channel videos.

    Attributes:
        method (Literal["mean", "min", "max", "select"]): Projection method.
            "mean", "min" and "max" aggregate the channels with the appropriate function.
            "select" simply selects one specific channel.
            Default: "mean".
        selected (int): Selected channel if method is "select".
            Default: 0.
    """

    def __init__(
        self,
        method: Literal["mean", "min", "max", "select"] = "mean",
        selected: int = 0,
    ):
        super().__init__()
        self.method = method
        self.selected = selected

    @override
    def initialize(self, video: Sequence[np.ndarray] | np.ndarray) -> None:
        """Initialize the preprocessor for the given video.

        This will reduce the channel in the `shape` attribute.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): The video to preprocess.
                Sequence of T frames (array). Each array is expected to have a shape ([D, ]H, W, C).

        """
        super().initialize(video)

        shape = self.shape[-1]
        if not -shape < self.selected <= shape:
            raise IndexError(f"index {self.selected} is out of bounds for channel axis with size {shape}.")

        self._shape = (*self.shape[:-1], 1)  # Reduce the channel axis

    @override
    def preprocess_frame(self, frame: np.ndarray, frame_id=0) -> np.ndarray:
        if self.method == "max":
            return frame.max(-1, keepdims=True)
        if self.method == "min":
            return frame.min(-1, keepdims=True)
        if self.method == "mean":
            return frame.mean(-1, keepdims=True)

        return frame[..., self.selected : self.selected + 1]

    @override
    def preprocess_video(self, video: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        if not isinstance(video, np.ndarray):
            return super().preprocess_video(video)

        # Let's do it directly on the global np.ndarray
        if self.method == "max":
            return video.max(-1, keepdims=True)
        if self.method == "min":
            return video.min(-1, keepdims=True)
        if self.method == "mean":
            return video.mean(-1, keepdims=True)

        return video[..., self.selected : self.selected + 1]
