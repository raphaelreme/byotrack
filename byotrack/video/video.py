from __future__ import annotations

import dataclasses
import os
from typing import overload, Optional, Sequence, Tuple, Union

import numpy as np

from .reader import VideoReader, slice_length
from .transforms import ChannelSelect, ChannelAvg, ScaleAndNormalize


@dataclasses.dataclass
class VideoTransformConfig:
    """Configuration for video transformations

    Attributes:
        aggregate (bool): Aggregate channels
        normalize (bool): Scale and Normalize the video in [0, 1]
        selected_channel (Optional[int]): Channel to use for aggregation
            If None, channel average is done. If any, it performs channel selection
        q_min (float): Minimum quantile to use when scaling the video
        q_max (float): Maximum quantile to use when scaling the video
        smooth_clip (float): Smoothness of the clipping process (log clipping)
            See `ScaleAndNormalize`: it logs clip the highest values on q_max.
            If 0.0, hard clipping is done.
        compute_stats_on (int): Number of frames to use to compute the quantiles.

    """

    aggregate: bool = False
    normalize: bool = False
    selected_channel: Optional[int] = None
    q_min: float = 0.0
    q_max: float = 1.0
    smooth_clip: float = 0.0
    compute_stats_on: int = 50


class Video(Sequence[np.ndarray]):
    """Video: Iterable, indexable and sliceable sequence of frames wrapping a VideoReader

    It wraps VideoReader in order to add video transformation (Channel Aggregation, Scaling, Normalization)
    and to add useful pythonic protocols (Sliceable, Indexable, Iterable).

    Frames are 2D or 3D with a channel axis. It behaves similarly as a 5D/4D numpy array of shape (T[, D], H, W, C).

    Example:
        .. code-block:: python

            import byotrack

            # Read a video (Usually 2D RGB)
            video = byotrack.Video(video_path)

            # Add a transform that will aggregate channel and normalize in [0, 1] the intensities
            transform_config = byotrack.VideoTransformConfig(aggregate=True, normalize=True, q_min=0.01, q_max=0.999)
            video.set_transform(transform_config)

            # Iterate through the video
            for frame in video:
                pass

            # Temporal slicing
            sliced = video[10:50:3]  # Take one frame every three from frame 10 to frame 50.

            # Spatial slicing
            sliced = video[:, 100:200, 150:250]  # All frames on the roi (100:200 x 150:250)


    Attributes:
        ndim (int): Either 4 (2D) or 5 (3D). (T, H, W, C) in 2D or (T, D, H, W, C) in 3D.
        shape (Tuple[int, ...]): Shape of the video (Time, [Depth, ]Height, Width)
        channels (int): Number of channels
        reader (byotrack.VideoReader): Underlying video reader

    """

    def __init__(self, data_source: Union[str, os.PathLike, VideoReader], **kwargs) -> None:
        """Constructor.


        Args:
            data_source (Union[str, os.PathLike, VideoReader]): Source of the data. If a path is given,
                it will be converted in a VideoReader.
            **kwargs: Additional arguments given to the construction of the video reader.

        """
        super().__init__()

        if isinstance(data_source, (str, os.PathLike)):
            self.reader = VideoReader.open(data_source, **kwargs)
        else:
            self.reader = data_source

        self._slices: Tuple[slice, ...] = tuple(slice(None) for _ in (self.reader.length, *self.reader.shape))
        self._channel_aggregator: Optional[Union[ChannelAvg, ChannelSelect]] = None
        self._normalizer: Optional[ScaleAndNormalize] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(
            slice_length(slice_, shape) for slice_, shape in zip(self._slices, (self.reader.length, *self.reader.shape))
        )

    @property
    def ndim(self) -> int:
        return 2 + len(self.reader.shape)

    @property
    def channels(self) -> int:
        return self.reader.channels if self._channel_aggregator is None else 1

    def set_transform(self, transform_config: VideoTransformConfig) -> None:
        """Set the transform (channel_selector and normalizer)

        Args:
            transform_config (byotrack.VideoTransformConfig): Configuration of the transformations

        """
        self._channel_aggregator = None
        if transform_config.aggregate:
            if transform_config.selected_channel is not None:
                self._channel_aggregator = ChannelSelect(transform_config.selected_channel)
            else:
                self._channel_aggregator = ChannelAvg()

        self._normalizer = None
        if transform_config.normalize:
            frames = np.asarray(self[: transform_config.compute_stats_on])
            self._normalizer = ScaleAndNormalize(
                transform_config.q_min,
                transform_config.q_max,
                transform_config.smooth_clip,
                transform_config.compute_stats_on,
            )
            self._normalizer.update_stats(frames)

    def transform(self, frame: np.ndarray) -> np.ndarray:
        """Transform a frame using channel aggregation and normalization"""
        if self._channel_aggregator:
            frame = self._channel_aggregator(frame)
        if self._normalizer:
            frame = self._normalizer(frame)

        return frame

    def __len__(self) -> int:
        return slice_length(self._slices[0], self.reader.length)  # length of temporal slice

    @overload
    def __getitem__(self, index: int) -> np.ndarray: ...

    @overload
    def __getitem__(self, slice_: slice) -> Video: ...

    @overload
    def __getitem__(self, slices: Tuple[slice, ...]) -> Video: ...

    def __getitem__(self, key):  # pylint: disable=too-many-branches
        """Indexing and slicing operations

        When indexed, it returns the ith frame in the slice
        When sliced, it duplicates the video (wrapper) with the right slice

        Args:
            key (int | slice | Tuple[slice, ...]): index or slice of the video

        Returns:
            np.ndarray | Video: Frame at index or a shallow copy of the video with the right slice

        """
        if isinstance(key, int):
            start, _, step = self._slices[0].indices(self.reader.length)

            if key < 0:
                key += len(self)

            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} out of range")

            frame_id = start + key * step

            if frame_id == self.reader.frame_id:
                pass  # Skip expensive seek
            if frame_id == self.reader.frame_id + 1:
                if not self.reader.grab():  # Much faster for cv2: Allows a fast frame by frame reading
                    raise RuntimeError(f"Unable to grab frame {frame_id}")
            else:
                try:
                    self.reader.seek(frame_id)
                except EOFError:
                    raise RuntimeError(f"Unable to seek frame {frame_id}") from None

            return self.transform(self.reader.retrieve()[self._slices[1:]])

        if isinstance(key, slice):
            key = (key,)

        if isinstance(key, tuple):
            if len(key) > len(self._slices):
                raise IndexError("Too many indices for video. Only support 3 dimensions slicing (time, height, width).")

            slices = list(self._slices)
            shapes = (self.reader.length, *self.reader.shape)
            for i, slice_ in enumerate(key):
                if not isinstance(slice_, slice):
                    raise TypeError("Unsupported index for Video. Supports only int, slice and Tuple[slice, ...]")

                slices[i] = compose_slice(self._slices[i], slice_, shapes[i])

            # Duplicate
            other = Video(self.reader)
            other._channel_aggregator = self._channel_aggregator
            other._normalizer = self._normalizer
            other._slices = tuple(slices)

            return other

        raise TypeError("Unsupported index for Video. Supports only int, slice and Tuple[slice, ...]")


def compose_slice(slice_1: slice, slice_2: slice, length: int) -> slice:
    """Compose two slices in the given order"""
    # Compute start, stop, step and length of the first slice
    start_1, _, step_1 = slice_1.indices(length)
    length_1 = slice_length(slice_1, length)

    # Same for the second slice
    start_2, stop_2, step_2 = slice_2.indices(length_1)
    length = slice_length(slice_2, length_1)

    # Compute the final start, stop and step
    start = start_1 + start_2 * step_1
    stop = start_1 + stop_2 * step_1
    step = step_1 * step_2

    if start > stop:
        if stop < 0:  # (5, -1, -1) is empty, instead use (5, None, -1)
            return slice(start, None, step)

    return slice(start, stop, step)
