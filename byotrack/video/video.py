from __future__ import annotations

import dataclasses
import os
from typing import overload, Optional, Sequence, Tuple, Union

import numpy as np

from .reader import VideoReader
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
        smooth_clip (float): Smoothness of the clipping process (scaling)

    """

    aggregate: bool = False
    normalize: bool = False
    selected_channel: Optional[int] = None
    q_min: float = 0.0
    q_max: float = 1.0
    smooth_clip: float = 0.0


class Video(Sequence[np.ndarray]):
    """Video: Iterable, indexable and sliceable sequence of frames wrapping a VideoReader

    It wraps VideoReader in order to add video transformation (Channel Aggregation, Scaling, Normalization)
    and to add useful pythonic protocols (Sliceable, Indexable, Iterable).

    Return images in BGR by default like opencv as frames are mostly used with opencv afterwards for display.
    Can also return grayscale image (H, W, 1)

    Example:
        .. code-block:: python

            import byotrack

            # Read a video (Usually BGR)
            video = byotrack.Video(video_path)

            # Add a transform that will aggregate channel and normalize in [0, 1] the intensities
            transform_config = byotrack.VideoTransformConfig(aggregate=True, normalize=True, q_min=0.01, q_max=0.999)
            video.set_transform(transform_config)

            # Iterate through the video
            for frame in video:
                pass

            # Temporal slicing

            sliced = video[10:50:3]  # Take one frame every three from frame 10 to frame 50.


    Attributes:
        shape (Tuple[int, int, int, int]): Shape of the video (Time, Height, Width, Channel)
        reader (byotrack.VideoReader): Underlying video reader

    """

    def __init__(self, data_source: Union[str, os.PathLike, VideoReader]) -> None:
        super().__init__()

        if isinstance(data_source, (str, os.PathLike)):
            self.reader = VideoReader.open(data_source)
        else:
            self.reader = data_source

        self._slice: Tuple[int, int, int] = (0, self.reader.length, 1)
        self._channel_aggregator: Optional[Union[ChannelAvg, ChannelSelect]] = None
        self._normalizer: Optional[ScaleAndNormalize] = None

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (len(self), *self.reader.shape, self.reader.channels)

    def set_transform(self, transform_config: VideoTransformConfig) -> None:
        """Set the transform (channel_selector and normalizer)

        Args:
            config (VideoTransformConfig): Configuration of the transformations

        """
        self._channel_aggregator = None
        if transform_config.aggregate:
            if transform_config.selected_channel is not None:
                self._channel_aggregator = ChannelSelect(transform_config.selected_channel)
            else:
                self._channel_aggregator = ChannelAvg()

        self._normalizer = None
        if transform_config.normalize:
            self._normalizer = ScaleAndNormalize(
                transform_config.q_min, transform_config.q_max, transform_config.smooth_clip
            )
            self._set_normalizer()  # Set normalizer stats

    def _set_normalizer(self) -> None:
        """Set the normalizer stats (after aggregation)"""
        assert self._normalizer

        frame_id = self.reader.tell()
        self.reader.seek(self._slice[0])

        frames = []

        has_next = True
        while has_next:
            frame, has_next = self.reader.read()
            if self._channel_aggregator:
                frame = self._channel_aggregator(frame)

            frames.append(frame[None, ...])
            if len(frames) >= self._normalizer.max_frames_for_stats:
                break

        self._normalizer.update_stats(np.concatenate(frames, axis=0))

        self.reader.seek(frame_id)  # Reset reader where it was

    def transform(self, frame: np.ndarray) -> np.ndarray:
        """Transform a frame using channel aggregation and normalization"""
        if self._channel_aggregator:
            frame = self._channel_aggregator(frame)
        if self._normalizer:
            frame = self._normalizer(frame)

        return frame

    def __len__(self) -> int:
        return (self._slice[1] - self._slice[0] - 1) // self._slice[2] + 1  # Number of elt in the slice

    @overload
    def __getitem__(self, index: int) -> np.ndarray:
        ...

    @overload
    def __getitem__(self, slice_: slice) -> Video:
        ...

    def __getitem__(self, key):
        """Indexing and slicing operations

        When indexed, it returns the ith frame in the slice
        When sliced, it duplicates the video (wrapper) with the right slice

        Args:
            key (int | slice): index or slice of the video

        Returns:
            np.ndarray | Video: Frame at index or a shallow copy of the video with the right slice

        """
        if isinstance(key, int):
            frame_id = self._slice[0] + key * self._slice[2]

            if frame_id >= self._slice[1]:
                raise IndexError("Frame index out of range")

            if frame_id == self.reader.frame_id:
                pass  # Skip expensive seek
            if frame_id == self.reader.frame_id + 1:
                if not self.reader.grab():  # Much faster for cv2: Allows a fast frame by frame reading
                    raise IndexError("Frame index out of video")
            else:
                try:
                    self.reader.seek(frame_id)
                except EOFError as eof:
                    raise IndexError(eof.args) from None

            return self.transform(self.reader.retrieve())

        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))  # Return int pos, int pos, int

            # Duplicate
            other = Video(self.reader)
            other._channel_aggregator = self._channel_aggregator
            other._normalizer = self._normalizer

            # Set the new slice
            other._slice = (
                self._slice[0] + start * self._slice[2],
                self._slice[0] + stop * self._slice[2],
                step * self._slice[2],
            )

            return other

        raise TypeError(f"Cannot index Video with type: {type(key)}")
