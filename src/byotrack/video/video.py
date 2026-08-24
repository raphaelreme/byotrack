from __future__ import annotations

import dataclasses
import os
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from byotrack.video import ChannelProjection, FrameSlicer, IntensityNormalizer, SpatialProjection, Zoom
from byotrack.video.reader import ArrayVideoReader, VideoReader, slice_length

if TYPE_CHECKING:
    from types import EllipsisType

    from byotrack.video import VideoPreprocessor
    from byotrack.video.preprocessor.zoom import ZoomMode, ZoomOrder


@dataclasses.dataclass
class VideoTransformConfig:
    """Configuration for video transformations.

    Attributes:
        aggregate (bool): Aggregate channels
        normalize (bool): Scale and Normalize the video in [0, 1]
        selected_channel (int | None): Channel to use for aggregation
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
    selected_channel: int | None = None
    q_min: float = 0.0
    q_max: float = 1.0
    smooth_clip: float = 0.0
    compute_stats_on: int = 50


class Video(Sequence[np.ndarray]):
    """Video: Iterable, indexable and sliceable sequence of frames wrapping a VideoReader.

    It wraps VideoReader in order to add video preprocessing (Channel Aggregation, Normalization, Projection, ...)
    and to add useful pythonic protocols (Sliceable, Indexable, Iterable).

    Frames are 2D or 3D with a channel axis. It behaves similarly as a 5D/4D numpy array of shape (T[, D], H, W, C).

    Example:
        .. code-block:: python

            import byotrack

            # Read a video (Usually 2D RGB)
            video = byotrack.Video(video_path)

            # Normalize the video
            video = video.normalize()

            # Iterate through the video
            for frame in video:
                pass

            # Temporal slicing
            sliced = video[10:50:3]  # Take one frame every three from frame 10 to frame 50.

            # Spatial slicing
            sliced = video[:, 100:200, 150:250]  # All frames on the roi (100:200 x 150:250)


    Attributes:
        ndim (int): Either 4 (2D) or 5 (3D). (T, H, W, C) in 2D or (T, D, H, W, C) in 3D.
        shape (tuple[int, ...]): Shape of the video (T, [D, ]H, W, C).
        dtype (np.dtype): Data type of the video.
        reader (byotrack.video.VideoReader): Underlying video reader

    """

    def __init__(self, data_source: str | os.PathLike | VideoReader | np.ndarray, **kwargs: Any) -> None:
        """Constructor.

        Args:
            data_source (str | os.PathLike | byotrack.video.VideoReader | np.ndarray): Source of the data.
                If a path is given, it will be converted in a VideoReader.
                If an array-like is given, it will be wrapped in an ArrayVideoReader.
            **kwargs: Additional arguments given to the construction of the video reader.

        """
        super().__init__()

        if isinstance(data_source, (str, os.PathLike)):
            self.reader = VideoReader.open(data_source, **kwargs)
        elif isinstance(data_source, VideoReader):
            self.reader = data_source
        else:
            self.reader = ArrayVideoReader("", data_source, **kwargs)

        self._temporal_slice = slice(None)
        self._preprocessors: list[VideoPreprocessor] = []

        self._reader_frame_shape = (*self.reader.shape, self.reader.channels)

        if self.reader.length == 0:
            raise ValueError("No frame found in the video.")

        if np.prod(self._reader_frame_shape) == 0:
            raise ValueError("No pixel found in the video.")

    @property
    def dtype(self) -> np.dtype:  # noqa: D102
        if not self._preprocessors:
            return self.reader.dtype

        return self._preprocessors[-1].dtype

    @property
    def shape(self) -> tuple[int, ...]:  # noqa: D102
        if not self._preprocessors:
            return (len(self), *self._reader_frame_shape)

        return (len(self), *self._preprocessors[-1].shape)

    @property
    def ndim(self) -> int:  # noqa: D102
        return len(self.shape)

    def __len__(self) -> int:  # noqa: D105
        return slice_length(self._temporal_slice, self.reader.length)  # length of temporal slice

    def add_preprocessor(self, preprocessor: VideoPreprocessor) -> Video:
        """Add a preprocessor to the video.

        Added preprocessors are applied sequentially (you may check the order in `_preprocessors`).

        Note: This may change the shape or dtype of the Video.

        Args:
            preprocessor (byotrack.video.VideoPreprocessor): The preprocessor to add.
                Will be initialized with this video.

        Returns:
            byotrack.Video: self
        """
        start, _, step = self._temporal_slice.indices(self.reader.length)
        preprocessor.initialize(self, [start + key * step for key in range(len(self))])

        self._preprocessors.append(preprocessor)

        return self

    def normalize(
        self,
        q_min: float = 0.0,
        q_max: float = 1.0,
        gamma: float = 1.0,
        smooth_clip: float = 0,
        compute_stats_on: int = 50,
    ) -> Video:
        """Normalize each channel of the video into [0, 1].

        Copy the video and adds the `IntensityNormalizer` preprocessor with the given arguments.

        Args:
            q_min (float): Quantile of the minimum value to consider.
                Default: 0.0 (min value)
            q_max (float): Quantile of the maximum value to consider.
                Default: 1.0 (max value)
            gamma (float): Gamma correction to apply.
                Default: 1.0 (Disabled)
            smooth_clip (float): Smoothness of the clipping process (`a`).
                If 0, values are clipped on the quantiles
                Else, values above the maximum quantile are log clipped:
                I = 1 + a log((I - 1)/a + 1) for I > 1, with `a` the `smooth_clip` factor
                Typical values are between 0 and 1.
                Default: 0 (hard clipping)
            compute_stats_on (int): Max number of frames to compute stats on.
                It prevents heavy computations that may occur on large videos.
                Default: 50

        Returns:
            byotrack.Video: the normalized video
        """
        for preprocessor in self._preprocessors:
            if isinstance(preprocessor, IntensityNormalizer):
                warnings.warn(
                    "The video is already normalized. Consider removing this second normalization.", stacklevel=2
                )

        return self._copy().add_preprocessor(IntensityNormalizer(q_min, q_max, gamma, smooth_clip, compute_stats_on))

    def rescale(
        self,
        scale_factors: tuple[float, ...],
        *,
        order: ZoomOrder = 1,
        mode: ZoomMode = "reflect",
        cval: float = 0.0,
        anti_aliasing: bool = False,
    ) -> Video:
        """Rescale the video with the given spatial scales.

        Copy the video and adds the `Zoom` preprocessor with the given arguments.

        Note: For pure integer down-scaling by local averaging, see `LocalMeanDownscaler`,
            usable directly through `add_preprocessor`.

        Args:
            scale_factors (tuple[float, ...]): Zoom factor for each spatial axis ([D, ]H, W).
                Values above 1 up-scale, values below 1 down-scale.
            order (ZoomOrder): Spline interpolation order used by `scipy.ndimage.zoom`
                (0: nearest neighbor, 1: linear, up to 5: quintic spline).
                Default: 1
            mode (ZoomMode): Boundary handling mode used by `scipy.ndimage`, used both by the
                optional anti-aliasing filter and by the zoom itself.
                Default: "reflect"
            cval (float): Constant value used when `mode` is "constant" (or "grid-constant").
                Default: 0.0
            anti_aliasing (bool): Whether to Gaussian-blur each frame before zooming, to reduce
                aliasing artifacts when down-scaling.
                Default: False

        Returns:
            byotrack.Video: the rescaled video
        """
        return self._copy().add_preprocessor(
            Zoom(scale_factors, order=order, mode=mode, cval=cval, anti_aliasing=anti_aliasing)
        )

    def resize(
        self,
        size: tuple[int, ...],
        *,
        order: ZoomOrder = 1,
        mode: ZoomMode = "reflect",
        cval: float = 0.0,
        anti_aliasing: bool = False,
    ) -> Video:
        """Resize the video to the given spatial size.

        Copy the video and adds the `Zoom` preprocessor with the scale factors required to reach
        the given `size`.

        Args:
            size (tuple[int, ...]): Target size for each spatial axis ([D, ]H, W).
            order (ZoomOrder): Spline interpolation order used by `scipy.ndimage.zoom`
                (0: nearest neighbor, 1: linear, up to 5: quintic spline).
                Default: 1
            mode (ZoomMode): Boundary handling mode used by `scipy.ndimage`, used both by the
                optional anti-aliasing filter and by the zoom itself.
                Default: "reflect"
            cval (float): Constant value used when `mode` is "constant" (or "grid-constant").
                Default: 0.0
            anti_aliasing (bool): Whether to Gaussian-blur each frame before zooming, to reduce
                aliasing artifacts when down-scaling.
                Default: False

        Returns:
            byotrack.Video: the resized video
        """
        return self.rescale(
            tuple(out_size / in_size for out_size, in_size in zip(size, self.shape[1:-1], strict=True)),
            order=order,
            mode=mode,
            cval=cval,
            anti_aliasing=anti_aliasing,
        )

    def isotrope(
        self,
        anisotropy: tuple[float, float, float],
        target: Literal["min", "max", "same"],
        *,
        order: ZoomOrder = 1,
        mode: ZoomMode = "reflect",
        cval: float = 0.0,
        anti_aliasing: bool = False,
    ) -> Video:
        """Rescale the video into an isotrope one.

        Copy the video and adds the `Zoom` preprocessor with the scale factors required to make
        every spatial axis span the same physical pixel size, given the current `anisotropy`.

        Args:
            anisotropy (tuple[float, float, float]): Physical pixel size for each spatial axis
                (Z, Y, X). Only the last `ndim - 2` values are used (Y, X for 2D videos).
            target (Literal["min", "max", "same"]): How to pick the common physical pixel size.
                "min": use the largest physical pixel size (shrinks the video).
                "max": use the smallest physical pixel size (grows the video).
                "same": use the geometric mean physical pixel size (roughly preserves the pixel count).
            order (ZoomOrder): Spline interpolation order used by `scipy.ndimage.zoom`
                (0: nearest neighbor, 1: linear, up to 5: quintic spline).
                Default: 1
            mode (ZoomMode): Boundary handling mode used by `scipy.ndimage`, used both by the
                optional anti-aliasing filter and by the zoom itself.
                Default: "reflect"
            cval (float): Constant value used when `mode` is "constant" (or "grid-constant").
                Default: 0.0
            anti_aliasing (bool): Whether to Gaussian-blur each frame before zooming, to reduce
                aliasing artifacts when down-scaling.
                Default: False

        Returns:
            byotrack.Video: the isotropic video
        """
        anisotropy_ = anisotropy[-(self.ndim - 2) :]

        if target == "min":  # min video size => use the largest pixel size
            pixel_size = max(anisotropy_)
        elif target == "max":
            pixel_size = min(anisotropy_)
        else:
            # We keep roughly the same number of pixels
            pixel_size = float(np.prod(anisotropy_)) ** (1 / len(anisotropy_))

        scale_factors = tuple(ani / pixel_size for ani in anisotropy_)

        return self.rescale(scale_factors, order=order, mode=mode, cval=cval, anti_aliasing=anti_aliasing)

    def _preprocess(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        for preprocessor in self._preprocessors:
            frame = preprocessor.preprocess_frame(frame, frame_id)

        return frame

    def set_transform(self, transform_config: VideoTransformConfig) -> None:
        """Deprecated. Will be removed in a future version."""
        warnings.warn(
            "`set_transform` is deprecated and will be removed in a future version. "
            "Use `normalize`, slicing or `VideoPreprocessor` directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if transform_config.aggregate:
            if transform_config.selected_channel is None:
                self.add_preprocessor(ChannelProjection("mean"))
            else:
                self.add_preprocessor(ChannelProjection("select", transform_config.selected_channel))

        if transform_config.normalize:
            self.add_preprocessor(
                IntensityNormalizer(
                    transform_config.q_min,
                    transform_config.q_max,
                    transform_config.smooth_clip,
                    transform_config.compute_stats_on,
                )
            )

    @overload
    def __getitem__(self, index: int) -> np.ndarray: ...

    @overload
    def __getitem__(self, slice_: slice) -> Video: ...

    @overload
    def __getitem__(self, slices: tuple[slice | int | EllipsisType, ...] | EllipsisType) -> Video: ...

    def __getitem__(self, key):  # noqa: C901, PLR0912
        """Indexing and slicing operations.

        When indexed, it returns the ith frame in the slice.
        When sliced, it duplicates the video (wrapper) with the right slicing.

        Args:
            key (int | slice | EllipsisType | tuple[slice | int | EllipsisType, ...]): index or slice of the video

        Returns:
            np.ndarray | Video: Frame at index or a shallow copy of the video with the right slicing

        """
        if isinstance(key, int):
            start, _, step = self._temporal_slice.indices(self.reader.length)

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

            return self._preprocess(self.reader.retrieve(), frame_id)

        if key is Ellipsis:  # Handle video[...] => returns a shallow copy
            return self._copy()

        if isinstance(key, slice):
            key = (key,)

        if isinstance(key, tuple):
            if len(key) > self.ndim:
                raise IndexError("Too many indices for video.")

            if len(key) == 0:
                return self._copy()

            # Expand Ellipsis
            key = expand_ellipsis(key, self.ndim)

            temporal_slice = key[0]
            if isinstance(temporal_slice, int):  # Handle int indexing for temporal axis
                return self[temporal_slice][key[1:]]  # XXX: This breaks typing (this is not a video but a frame)

            temporal_slice = compose_slice(self._temporal_slice, temporal_slice, self.reader.length)

            # Let's check for integer in the spatial axis
            key, projection = _handle_integer_slicing(key[1:], self.ndim)

            # Duplicate the video and register the new temporal_slice, projection and slicer.
            other = self._copy()
            other._temporal_slice = temporal_slice  # noqa: SLF001
            if projection[0] >= 0:
                other.add_preprocessor(SpatialProjection(projection[0], "select", selected=projection[1]))

            if len(key) > 0:
                other.add_preprocessor(FrameSlicer(key))

            return other

        raise TypeError("Unsupported index for Video. Supports only int, slice and tuple[slice, ...]")

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:  # noqa: FBT001
        """Convert the video into a numpy array by reading and copying each frame.

        Implemented so that `np.asarray(video)` (or `np.array(video)`) preallocates the
        output array and fills it frame by frame, rather than falling back to numpy's
        generic sequence conversion which first collects every frame into a Python list
        before copying it into the final array (roughly doubling peak memory usage).

        Args:
            dtype (np.dtype | None): Dtype of the output array. Defaults to `self.dtype`.
            copy (bool | None): Following numpy's `__array__` protocol, a copy is always
                required to build the array. `True` and `None` are therefore accepted, but
                `False` raises as it cannot be honored.

        Returns:
            np.ndarray: The video loaded as a single array of shape `self.shape`.

        """
        if copy is False:
            raise ValueError(
                "Unable to avoid copy while creating an array from a Video: frames must be read and copied."
            )

        array = np.empty(self.shape, dtype=dtype or self.dtype)
        for frame_id, frame in enumerate(self):
            array[frame_id] = frame

        return array

    def _copy(self) -> Video:
        """Create a shallow copy of the video."""
        copy = Video(self.reader)
        copy._temporal_slice = self._temporal_slice
        copy._preprocessors = self._preprocessors.copy()

        return copy


def expand_ellipsis(slices: tuple[int | slice | EllipsisType, ...], ndim: int) -> tuple[int | slice, ...]:
    """Expand ellipsis in slices."""
    slices_: list[int | slice] = []
    expanded = False

    for slice_ in slices:
        if slice_ is Ellipsis:
            if expanded:
                raise IndexError("An index can only have a single ellipsis ('...')")
            slices_.extend(slice(None) for _ in range(ndim - len(slices) + 1))
            expanded = True
        else:
            slices_.append(slice_)

    return tuple(slices_)


def _handle_integer_slicing(slices: tuple[int | slice, ...], ndim: int) -> tuple[tuple[slice, ...], tuple[int, int]]:
    """Handle integer slicing for videos.

    It will raise if not 3D, if multiple integers are found, or if integer slicing on the channel axis.

    If a single integer is given, it is removed from the slice and its axis and value is returned.
    """
    slices_: list[slice] = []
    has_an_integer = False

    axis = -1
    value = 0

    for i, slice_ in enumerate(slices):
        if isinstance(slice_, int):
            if i + 1 == ndim - 1:
                raise IndexError("Channel axis can not be reduced. Use a slice or a ChannelProjection.")
            if ndim < 5 or has_an_integer:  # noqa: PLR2004
                raise IndexError("Spatial projection is only available for 3D videos.")

            value = slice_
            axis = i
            has_an_integer = True
        else:
            slices_.append(slice_)

    return tuple(slices_), (axis, value)


def compose_slice(slice_1: slice, slice_2: slice, length: int) -> slice:
    """Compose two slices in the given order."""
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

    if start > stop and stop < 0:  # (5, -1, -1) is empty, instead use (5, None, -1)
        return slice(start, None, step)

    return slice(start, stop, step)


def video_length(video: Sequence[np.ndarray] | np.ndarray) -> int:
    """Utilities to extract a video length through duck-typing."""
    if hasattr(video, "shape"):
        return video.shape[0]

    return len(video)


def video_shape(video: Sequence[np.ndarray] | np.ndarray) -> tuple[int, ...]:
    """Utilities to extract a video shape through duck-typing."""
    if hasattr(video, "shape"):
        return video.shape

    if len(video) == 0:
        raise ValueError("video is empty: cannot infer its shape.")

    return (len(video), *video[0].shape)


def video_dtype(video: Sequence[np.ndarray] | np.ndarray) -> np.dtype:
    """Utilities to extract a video dtype through duck-typing."""
    if hasattr(video, "dtype"):
        return video.dtype

    if len(video) == 0:
        raise ValueError("video is empty: cannot infer its dtype.")

    return video[0].dtype
