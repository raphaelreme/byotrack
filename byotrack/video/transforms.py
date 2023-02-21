from __future__ import annotations

import dataclasses
from typing import Optional, Union

import numpy as np

from . import reader


# pylint: disable=too-few-public-methods


@dataclasses.dataclass
class VideoTransformConfig:
    """Configuration for default video transformations

    Attrs:
        aggregation (bool): Use channel aggregation (from 3 to 1 if possible)
        normalize (bool): Scale and Normalize the video in [0, 1]
        selected_channel (Optional[int]): Channel to use for aggregation
            If None, channel average is done. If any, it performs channel selection
        q_min (float): Minimum quantile to use when scaling the video
        q_max (float): Maximum quantile to use when scaling the video
    """

    aggregation: bool = False
    normalize: bool = False
    selected_channel: Optional[int] = None
    q_min: float = 0.0
    q_max: float = 1.0


class VideoTransform:
    """Transform each image of a video using some default useful transformations

    It follows the two optional steps:
        1- Channel aggregation (Selection or average)
        2- Scaling and normalization
    """

    def __init__(
        self,
        video: reader.VideoReader,
        config: VideoTransformConfig,
    ) -> None:
        self.config = config
        self.aggregation = video.channels > 1 and config.aggregation
        self.normalize = config.normalize

        self.aggregator: Union[ChannelAvg, ChannelSelect] = ChannelAvg()
        self.normalizer = ScaleAndNormalize()

        if config.selected_channel is not None:
            self.aggregator = ChannelSelect(config.selected_channel)

        if self.normalize:
            self._set_normalizer(video)

    def _set_normalizer(self, video: reader.VideoReader):
        """Set the normalizer stats (after aggregation)"""
        frame_id = video.tell()
        video.seek(0)
        frames = []

        has_next = True
        while has_next:
            frame = video._retrieve()  # pylint: disable=protected-access
            has_next = video.grab()
            if self.aggregation:
                frame = self.aggregator(frame)

            frames.append(frame[None, ...])
            if len(frames) >= self.normalizer.MAX_FRAMES_FOR_STATS:
                break

        self.normalizer.update_stats(self.config.q_min, self.config.q_max, np.concatenate(frames, axis=0))

        video.seek(frame_id)  # Reset video where it was

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.aggregation:
            frame = self.aggregator(frame)

        if self.normalize:
            return self.normalizer(frame)

        return frame


class ChannelSelect:
    """Select a given channel

    Attrs:
        channel (int): Channel to keep (0, 1 or 2) => (B, G, R)

    Args:
        frame (np.ndarray): Frame of the video
            Shape: (H, W, C)

    Returns:
        np.ndarray: Filtered frame with a single channel
            Shape: (H, W, 1)
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
            Shape: (H, W, C)

    Returns:
        np.ndarray: Average of channels
            Shape: (H, W, 1)
    """

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return np.mean(frame, axis=-1, keepdims=True)


class ScaleAndNormalize:
    """Scale and Normalize each channel into [0, 1]

    min and max values are computed using quantile of the video to improve stability

    Attrs:
        mini (np.ndarray): Minimum value kept (one for each channel)
            Shape: (C, )
        maxi (np.ndarray): Maximum value kept (one for each channel)
            Shape (C, )

    Args:
        frame (np.ndarray): Frame of the video
            Shape: (H, W, C)

    Returns:
        np.ndarray: Normalized version of the frame in [0, 1]
            Shape: (H, W, C)
    """

    MAX_FRAMES_FOR_STATS = 100  # Do not use all the frames of a video
    # because it is both time and memory expensive

    def __init__(self) -> None:
        self.mini = np.array([0.0])
        self.maxi = np.array([1.0])

    def update_stats(self, q_min: float, q_max: float, frames: np.ndarray) -> None:
        """Update mini and maxi values based on the given frames and quantiles

        Args:
            q_min (float): Quantile of the minimum value to consider
            q_max (float): Quantile of the maximum value to consider
            frames (np.ndarray): Several frames of the same video to compute the stats
        """

        self.mini = np.quantile(frames, q_min, axis=(0, 1, 2))
        self.maxi = np.quantile(frames, q_max, axis=(0, 1, 2))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        frame = np.clip(frame, self.mini, self.maxi)
        frame -= self.mini
        frame /= self.maxi - self.mini
        return frame
