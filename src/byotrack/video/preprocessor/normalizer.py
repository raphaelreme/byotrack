from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from byotrack.video.preprocessor import preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class IntensityNormalizer(preprocessor.VideoPreprocessor):
    """Normalize each channel intensity into [0, 1].

    `mini` and `maxi` values are computed using quantile of the video to improve stability.
    The quantiles are computed using only the first `compute_stats_on` frames.

    Frame shape is preserved, but dtype is changed to float32.

    Note: A `smooth_clip` can be performed by log clipping values above `maxi` up until the log max.

    Attributes:
        q_min (float): Quantile of the minimum value to consider
        q_max (float): Quantile of the maximum value to consider
        mini (np.ndarray): Minimum value kept (one for each channel)
            Shape: (C, ), dtype: float32
        maxi (np.ndarray): Maximum value kept (one for each channel)
            Shape: (C, ), dtype: float32
        smooth_clip (float): Smoothness of the clipping process (`a`)
            If 0, values are clipped on mini/maxi
            Else, values above maxi are log clipped:
            I = 1 + a log((I - 1)/a + 1) for I > 1, with `a` the `smooth_clip` factor
            Typical values are between 0 and 1.
            Default: 0 (hard clipping)
        max (np.ndarray): True maximum values (one for each channel) when using smooth clipping
            Shape: (C, ), dtype: float32
        compute_stats_on (int): Max number of frames to compute stats on.
            It prevents heavy computations that may occur on large videos.
            Default: 50

    """

    def __init__(self, q_min: float, q_max: float, smooth_clip: float = 0, compute_stats_on: int = 50) -> None:
        super().__init__()
        self._dtype = np.dtype(np.float32)

        self.q_min = q_min
        self.q_max = q_max
        self.mini = np.array([0])
        self.maxi = np.array([1])
        self.smooth_clip = smooth_clip
        self.max = np.array([1.0])
        self.compute_stats_on = compute_stats_on

    @override
    def initialize(self, video: Sequence[np.ndarray] | np.ndarray) -> None:
        """Initialize the preprocessor for the given video.

        It computes `mini` and `maxi` values based on the first frames of the video.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): The video to preprocess.
                Sequence of T frames (array). Each array is expected to have a shape ([D, ]H, W, C).
        """
        super().initialize(video)

        self._dtype = np.dtype(np.float32)  # Change the dtype of the video

        frames = np.asarray(video[: self.compute_stats_on])

        axis = tuple(range(frames.ndim - 1))
        frames = frames[: self.compute_stats_on]
        self.mini = np.quantile(frames, self.q_min, axis=axis).astype(frames.dtype)
        self.maxi = np.quantile(frames, self.q_max, axis=axis).astype(frames.dtype)

        if self.smooth_clip > 0:
            ratio = frames.max(axis=axis).astype(np.float32) / (self.maxi + (self.maxi == 0))
            self.max = 1 + 0.5 * np.log(np.maximum(1, 1 + (ratio - 1) / 0.5))

    @override
    def preprocess_frame(self, frame: np.ndarray, frame_id=0) -> np.ndarray:
        if self.smooth_clip <= 0:  # No smooth clip
            np.clip(frame, self.mini, self.maxi, out=frame)
            frame -= self.mini

            # Divide by one if mini == maxi
            return frame.astype(np.float32) / (self.maxi - self.mini + (self.maxi == self.mini))

        np.clip(frame, self.mini, None, out=frame)
        frame -= self.mini
        frame = frame.astype(np.float32) / (self.maxi - self.mini + (self.maxi == self.mini))

        # Log clipping high values
        mask = frame > 1
        frame[mask] = 1 + self.smooth_clip * np.log((frame[mask] - 1) / self.smooth_clip + 1)
        np.clip(frame, 0, self.max, out=frame)
        return frame / self.max
