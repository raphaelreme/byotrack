from __future__ import annotations

import numpy as np


# pylint: disable=too-few-public-methods


class ChannelSelect:
    """Select a given channel

    Attributes:
        channel (int): Channel to keep

    Args:
        frame (np.ndarray): Frame of a video
            Shape: (..., C)

    Returns:
        np.ndarray: Filtered frame with a single channel
            Shape: (..., 1)

    """

    def __init__(self, channel: int) -> None:
        """Constructor

        Args:
            channel (int): Selected channel

        """
        self.channel = channel

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return frame[..., self.channel : self.channel + 1]


class ChannelAvg:
    """Average channels into a single one

    Args:
        frame (np.ndarray): Frame of a video
            Shape: (..., C)

    Returns:
        np.ndarray: Average of channels
            Shape: (..., 1)

    """

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[-1] == 1:
            return frame
        if frame.dtype is np.float32:
            np.mean(frame, axis=-1, keepdims=True, out=frame[..., :1])
            return frame
        return np.mean(frame, axis=-1, keepdims=True, out=np.zeros(frame[..., :1].shape, dtype=np.float32))


class ScaleAndNormalize:
    """Scale and Normalize each channel into [0, 1]

    min and max values are computed using quantile of the video to improve stability

    Attributes:
        q_min (float): Quantile of the minimum value to consider
        q_max (float): Quantile of the maximum value to consider
        mini (np.ndarray): Minimum value kept (one for each channel)
            Shape: (C, ), dtype: float32
        maxi (np.ndarray): Maximum value kept (one for each channel)
            Shape: (C, ), dtype: float32
        smooth_clip (float): Smoothness of the clipping process
            If 0, values are clipped on mini/maxi
            Else, values above maxi are log clipped:
            v = 1 + a log((v - 1)/a + 1) for v > 1, with a the smooth_clip factor
            Typical values are between 0 and 1.
            Default: 0 (hard clipping)
        max (np.ndarray): True maximum values (one for each channel) when using smooth clipping
            Shape: (C, ), dtype: float32
        compute_stats_on (int): Max number of frames to compute stats on.
            It prevents heavy computations that can occurs with large videos.
            Default: 50

    Args:
        frame (np.ndarray): Frame of the video
            Shape: (..., C)

    Returns:
        np.ndarray: Normalized version of the frame in [0, 1]
            Shape: (..., C), dtype: float32

    """

    def __init__(self, q_min: float, q_max: float, smooth_clip: float = 0, compute_stats_on: int = 50) -> None:
        self.q_min = q_min
        self.q_max = q_max
        self.mini = np.array([0])
        self.maxi = np.array([1])
        self.smooth_clip = smooth_clip
        self.max = np.array([1.0])
        self.compute_stats_on = compute_stats_on

    def update_stats(self, frames: np.ndarray) -> None:
        """Update mini and maxi values based on the given frames

        Args:
            frames (np.ndarray): Several frames of the same video to compute the stats
                Shape: (..., C)

        """
        axis = tuple(range(frames.ndim - 1))
        frames = frames[: self.compute_stats_on]
        self.mini = np.quantile(frames, self.q_min, axis=axis).astype(frames.dtype)
        self.maxi = np.quantile(frames, self.q_max, axis=axis).astype(frames.dtype)

        if self.smooth_clip > 0:
            ratio = frames.max(axis=axis).astype(np.float32) / (self.maxi + (self.maxi == 0))
            self.max = 1 + 0.5 * np.log(np.maximum(1, 1 + (ratio - 1) / 0.5))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.smooth_clip <= 0:  # No smooth clip
            np.clip(frame, self.mini, self.maxi, out=frame)
            frame -= self.mini

            # Divide by one if mini == maxi
            return frame.astype(np.float32) / (self.maxi - self.mini + (self.maxi == self.mini))

        np.clip(frame, self.mini, None, out=frame)
        frame -= self.mini
        frame = frame.astype(np.float32) / (self.maxi - self.mini + (self.maxi == self.mini))

        # Log cliping high values
        mask = frame > 1
        frame[mask] = 1 + self.smooth_clip * np.log((frame[mask] - 1) / self.smooth_clip + 1)
        np.clip(frame, 0, self.max, out=frame)
        return frame / self.max
