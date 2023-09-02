from __future__ import annotations

import numpy as np


# pylint: disable=too-few-public-methods


class ChannelSelect:
    """Select a given channel

    Attributes:
        channel (int): Channel to keep (0, 1 or 2)

    Args:
        frame (np.ndarray): Frame of the video
            Shape: (..., H, W, C)

    Returns:
        np.ndarray: Filtered frame with a single channel
            Shape: (..., H, W, 1)

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
        frame (np.ndarray): Frame of the video
            Shape: (..., H, W, C)

    Returns:
        np.ndarray: Average of channels
            Shape: (..., H, W, 1)

    """

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return np.mean(frame, axis=-1, keepdims=True)


class ScaleAndNormalize:
    """Scale and Normalize each channel into [0, 1]

    min and max values are computed using quantile of the video to improve stability

    Attributes:
        q_min (float): Quantile of the minimum value to consider
        q_max (float): Quantile of the maximum value to consider
        mini (np.ndarray): Minimum value kept (one for each channel)
            Shape: (C, )
        maxi (np.ndarray): Maximum value kept (one for each channel)
            Shape: (C, )
        smooth_clip (float): Smoothness of the clipping process
            If 0, values are clipped on mini/maxi
            Else, values above maxi are log clipped:
            v = 1 + a log((v - 1)/a + 1) for v > 1, with a the smooth_clip factor
            Typical values are between 0 and 1.
            Default: 0 (hard clipping)
        max (np.ndarray): True maximum values (one for each channel) when using smooth clipping
            Shape: (C, )

    Args:
        frame (np.ndarray): Frame of the video
            Shape: (..., H, W, C)

    Returns:
        np.ndarray: Normalized version of the frame in [0, 1]
            Shape: (..., H, W, C)

    """

    # Do not use all the frames of a video because it is both time and memory expensive
    max_frames_for_stats = 100

    def __init__(self, q_min: float, q_max: float, smooth_clip: float = 0) -> None:
        self.q_min = q_min
        self.q_max = q_max
        self.mini = np.array([0.0])
        self.maxi = np.array([1.0])
        self.smooth_clip = smooth_clip
        self.max = np.array([1.0])

    def update_stats(self, frames: np.ndarray) -> None:
        """Update mini and maxi values based on the given frames

        Args:
            frames (np.ndarray): Several frames of the same video to compute the stats
                Shape: (N, H, W, C)

        """
        frames = frames[: self.max_frames_for_stats]
        self.mini = np.quantile(frames, self.q_min, axis=(0, 1, 2))
        self.maxi = np.quantile(frames, self.q_max, axis=(0, 1, 2))
        if self.smooth_clip > 0:
            self.max = 1 + self.smooth_clip * np.log((frames.max() / self.maxi - 1) / self.smooth_clip + 1)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.smooth_clip <= 0:  # No smooth clip
            frame = np.clip(frame, self.mini, self.maxi)
            frame -= self.mini
            frame /= self.maxi - self.mini
            return frame

        frame = (frame - self.mini) / (self.maxi - self.mini)
        # Log cliping high values
        frame[frame > 1] = 1 + self.smooth_clip * np.log((frame[frame > 1] - 1) / self.smooth_clip + 1)
        np.clip(frame, 0, self.max, frame)
        return frame / self.max
