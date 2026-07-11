from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

import byotrack
from byotrack.video.reader import ArrayVideoReader, VideoReader
from byotrack.video.video import (
    _handle_integer_slicing,
    compose_slice,
    expand_ellipsis,
    video_dtype,
    video_length,
    video_shape,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class _StubReader(VideoReader):
    """Minimal VideoReader stub for testing Video constructor guards."""

    def __init__(self, length: int, shape: tuple, channels: int = 1):
        super().__init__("")
        self.length = length
        self.shape = shape
        self.channels = channels

    @override
    def grab(self) -> bool:
        if self.frame_id + 1 >= self.length:
            return False
        self.frame_id += 1
        return True

    @override
    def retrieve(self) -> np.ndarray:
        return np.zeros((*self.shape, self.channels), dtype=np.uint8)

    @override
    def seek(self, frame_id: int) -> None:
        if frame_id < 0 or frame_id >= self.length:
            raise EOFError
        self.frame_id = frame_id


class _GrabFailingReader(_StubReader):
    """Stub reader whose grab() always returns False."""

    @override
    def grab(self) -> bool:
        return False


class _SeekFailingReader(_StubReader):
    """Stub reader whose seek() always raises EOFError."""

    @override
    def seek(self, frame_id: int) -> None:
        raise EOFError("seek failed")


## Video construction


def test_video_construction_from_2d_array(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    assert video.shape == video_2d.shape
    assert video.ndim == video_2d.ndim
    assert video.dtype == video_2d.dtype
    assert isinstance(video.reader, ArrayVideoReader)


def test_video_construction_from_3d_array(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    assert video.shape == video_3d.shape
    assert video.ndim == video_3d.ndim
    assert video.dtype == video_3d.dtype
    assert isinstance(video.reader, ArrayVideoReader)


def test_video_construction_from_reader(video_2d: np.ndarray):
    reader = ArrayVideoReader("", video_2d)
    video = byotrack.Video(reader)
    assert video.shape == video_2d.shape
    assert video.ndim == video_2d.ndim
    assert video.dtype == video_2d.dtype
    assert video.reader is reader


def test_video_construction_from_path(tiff_2d):
    path, data = tiff_2d
    video = byotrack.Video(path)

    assert video.shape == data.shape
    assert video.ndim == data.ndim
    assert video.dtype == data.dtype


def test_video_construction_empty_raises():
    with pytest.raises(ValueError, match="No frame"):
        byotrack.Video(_StubReader(length=0, shape=(10, 10)))

    with pytest.raises(IndexError):
        byotrack.Video(np.zeros((0, 10, 10, 2)))


def test_video_construction_zero_pixel_raises():
    with pytest.raises(ValueError, match="No pixel"):
        byotrack.Video(_StubReader(length=5, shape=(0, 10)))

    with pytest.raises(ValueError, match="No pixel"):
        byotrack.Video(np.zeros((10, 0, 10, 2)))


## Video Length


def test_video_len_2d(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    assert len(video) == len(video_2d)


def test_video_len_3d(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    assert len(video) == len(video_3d)


def test_video_len_after_temporal_slice(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    assert len(video[2:8]) == 6
    assert len(video[::2]) == 5
    assert len(video[1:9:3]) == 3
    assert len(video[9:1:-3]) == 3


## Iterate on frames


def test_video_iteration_2d(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    assert len(list(video)) == 10

    for frame_id, frame in enumerate(video):
        assert frame.shape == video_2d.shape[1:]
        assert frame.dtype == video_2d.dtype
        assert (frame == video_2d[frame_id]).all()


def test_video_iteration_3d(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    assert len(list(video)) == 8

    for frame_id, frame in enumerate(video):
        assert frame.shape == video_3d.shape[1:]
        assert frame.dtype == video_3d.dtype
        assert (frame == video_3d[frame_id]).all()


## __getitem__ —> frame indexing


def test_video_getitem_int_first_frame(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    frame = video[0]
    assert (frame == video_2d[0]).all()


def test_video_getitem_int_last_frame(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    frame = video[len(video_3d) - 1]
    frame_ = video[-1]

    assert (frame_ == frame).all()
    assert (frame == video_3d[-1]).all()


@pytest.mark.parametrize("index", [3, -3, 7, -7])
def test_video_getitem_int(index: int, video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    assert (video[index] == video_2d[index]).all()


@pytest.mark.parametrize("index", [10, -10])
def test_video_getitem_int_out_of_range_raises(index: int, video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    with pytest.raises(IndexError):
        video[index]


def test_video_getitem_int_sequential_read(video_2d: np.ndarray):
    # Accessing frames sequentially leverages the fast grab path and should still work
    video = byotrack.Video(video_2d)
    for i in range(len(video)):
        frame = video[i]
        assert (frame == video_2d[i]).all()


## __getitem__ —> temporal slicing


def test_video_temporal_slice_basic(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    sliced = video[2:8]
    assert isinstance(sliced, byotrack.Video)
    assert len(sliced) == 6
    assert sliced.shape == (6, 20, 30, 3)
    assert len(video) == 10  # unchanged by slicing (copy)

    assert (sliced[4] == video[6]).all()
    assert (sliced[-1] == video[7]).all()


def test_video_temporal_slice_step(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    sliced = video[::3]
    assert len(sliced) == 3  # (0 , 3, 6)
    assert len(video) == 8  # unchanged by slicing (copy)
    assert sliced.shape == (3, 5, 20, 30, 2)

    assert (sliced[0] == video[0]).all()
    assert (sliced[1] == video_3d[3]).all()
    assert (sliced[2] == video[6]).all()


def test_video_temporal_slice_reverse(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    sliced = video[9:2:-2]
    assert len(sliced) == 4

    assert (sliced[0] == video[9]).all()
    assert (sliced[1] == video[7]).all()


def test_video_temporal_slice_chained(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)

    chained = video[1:9][1:5:2][::-1][::-1]
    direct = video[2:6:2]

    assert len(chained) == len(direct)
    for frame_chained, frame_direct in zip(chained, direct, strict=True):
        assert (frame_chained == frame_direct).all()


## __getitem__ —> Spatial & Channel slicing


def test_video_spatial_slice_2d(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    sliced = video[:, 5:15, 10:20]
    assert sliced.shape == (10, 10, 10, 3)
    assert video.shape == (10, 20, 30, 3)  # unchanged by slicing (copy)
    assert (sliced[0] == video_2d[0, 5:15, 10:20]).all()


def test_video_spatial_slice_3d(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    sliced = video[:, 2::-1, :, 10:20]
    assert sliced.shape == (8, 3, 20, 10, 2)
    assert video.shape == (8, 5, 20, 30, 2)  # unchanged by slicing (copy)
    assert (sliced[0] == video[0][2::-1, :, 10:20]).all()


def test_video_spatial_slice_with_channel(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    sliced = video[:, 5:15, :, 1:2]
    assert sliced.shape == (10, 10, 30, 1)
    assert video.shape == (10, 20, 30, 3)  # unchanged by slicing (copy)
    assert (sliced[0] == video[0][5:15, :, 1:2]).all()


def test_video_complete_slice(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    sliced = video[2:4, :, :10, -5:, -1:]
    assert sliced.shape == (2, 5, 10, 5, 1)
    assert video.shape == (8, 5, 20, 30, 2)  # unchanged by slicing (copy)

    assert (sliced[0] == video[2][:, :10, -5:, -1:]).all()


## __getitem__ —> integer in tuple slicing.
## Leads to projection for spatial axis, raises on channel, and get frame on time with the given slice


def test_video_3d_slice_as_z_projection(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    projected = video[:, 2]
    assert isinstance(projected, byotrack.Video)
    assert projected.shape == (8, 20, 30, 2)
    assert video.shape == (8, 5, 20, 30, 2)
    assert (projected[0] == video_3d[0, 2]).all()


def test_video_3d_slice_as_x_projection_negative(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    projected = video[:, :, :, -1]
    assert projected.shape == (8, 5, 20, 2)
    assert (projected[0] == video[0][..., -1, :]).all()


def test_video_3d_slice_with_integer_on_temporal_axis_returns_frame(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    frame: np.ndarray = video[5, 5:15, 10, 0]  # type: ignore[assignment]  # XXX: Typing is incorrect for this behavior.

    assert frame.shape == (10,)
    assert (frame == video_2d[5, 5:15, 10, 0]).all()


def test_video_2d_slice_with_integer_on_channel_raises(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    with pytest.raises(IndexError, match="Channel axis"):
        video[:, :, :, 0]


def test_video_3d_slice_with_integer_on_channel_raises(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    with pytest.raises(IndexError, match="Channel axis"):
        video[:, :, :, :, 0]


def test_video_2d_slice_as_projection_raises(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    with pytest.raises(IndexError, match="3D"):
        video[:, 5]


## __getitem__ —> ellipsis handling


def test_video_ellipsis_returns_copy(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    copy = video[...]

    assert isinstance(copy, byotrack.Video)
    assert copy.shape == video.shape
    assert copy is not video


def test_video_ellipsis_in_tuple_with_spatial(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    sliced = video[..., 5:15, :]
    assert sliced.shape == (8, 5, 20, 10, 2)


def test_video_ellipsis_with_temporal_and_channel(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    sliced = video[2:5, ..., 1:2]
    assert sliced.shape == (3, 20, 30, 1)


def test_video_empty_tuple_returns_copy(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    copy = video[()]
    assert isinstance(copy, byotrack.Video)
    assert copy.shape == video.shape
    assert copy is not video


## __getitem__ —> Remaining error cases


def test_video_getitem_too_many_indices_raises(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    with pytest.raises(IndexError, match="Too many indices"):
        video[:, :, :, :, :]


def test_video_getitem_unsupported_type_raises(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    with pytest.raises(TypeError, match="Unsupported index"):
        video["bad"]  # type: ignore[call-overload]


def video_empty_slicing_raises(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)

    with pytest.raises(ValueError, match="No frame"):
        video[5:5]

    with pytest.raises(ValueError, match="No frame"):
        video[15:20]

    with pytest.raises(ValueError, match="No pixel"):
        video[:, 5:3]

    with pytest.raises(ValueError, match="No pixel"):
        video[:, -30:-25]

    with pytest.raises(ValueError, match="No pixel"):
        video[..., 4:]

    with pytest.raises(ValueError, match="No pixel"):
        video[..., :0]


def test_video_getitem_grab_failure_raises():
    # Covers the RuntimeError when grab() returns False (sequential read path)
    reader = _GrabFailingReader(length=2, shape=(10, 10))
    video = byotrack.Video(reader)
    video[0]  # accesses frame 0 via seek(0) —> succeeds

    # Now reader.frame_id=0; frame 1 triggers grab() path which fails
    with pytest.raises(RuntimeError, match="Unable to grab"):
        video[1]


def test_video_getitem_seek_failure_raises():
    # Covers the RuntimeError when seek() raises EOFError
    reader = _SeekFailingReader(length=2, shape=(10, 10))
    video = byotrack.Video(reader)

    with pytest.raises(RuntimeError, match="Unable to seek"):
        video[0]  # seek(0) raises EOFError -> wrapped as RuntimeError


## add_preprocessor


def test_video_add_preprocessor_updates_shape(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    video.add_preprocessor(byotrack.video.ChannelProjection("mean"))
    assert video.shape == (10, 20, 30, 1)
    assert video.dtype == video_2d.dtype  # Type is not changed by ChannelProjection


def test_video_add_preprocessor_updates_dtype(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    video.add_preprocessor(byotrack.video.IntensityNormalizer(0.0, 1.0))
    assert video.dtype == np.float32
    assert video.shape == video_3d.shape  # Shape is not changed by Normalizer


def test_video_add_preprocessor_returns_self(video_3d):
    video = byotrack.Video(video_3d)
    result = video.add_preprocessor(byotrack.video.SpatialProjection())
    assert result is video


def test_video_add_preprocessor_chaining(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    video.add_preprocessor(byotrack.video.IntensityNormalizer(0.0, 1.0)).add_preprocessor(
        byotrack.video.ChannelProjection("mean")
    )

    assert video.shape == (10, 20, 30, 1)
    assert video.dtype == np.float32

    frame = video[0]
    assert frame.min() >= 0.0
    assert frame.max() <= 1.0


def test_video_add_preprocessor_sequential_order(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    proj = byotrack.video.ChannelProjection("mean")
    norm = byotrack.video.IntensityNormalizer(0.0, 1.0)
    video.add_preprocessor(proj).add_preprocessor(norm)

    assert len(video._preprocessors) == 2
    assert isinstance(video._preprocessors[0], byotrack.video.ChannelProjection)
    assert isinstance(video._preprocessors[1], byotrack.video.IntensityNormalizer)

    assert norm.maxi.shape == (1,)  # norm only sees a single channel as it is after ChannelProjection


## Normalize


def test_video_normalize_basics(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)
    normalized = video.normalize()

    assert normalized is not video
    assert normalized.dtype == np.float32
    assert video.dtype == video_3d.dtype  # Unchanged as normalize returns a copy.
    assert normalized.shape == video.shape

    assert normalized[0].min() >= 0.0
    assert normalized[0].max() <= 1.0


def test_video_normalize_double_warns(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    normalized = video.normalize()
    with pytest.warns(UserWarning, match="already normalized"):
        normalized.normalize()


def test_video_normalize_do_not_warns_with_other_preprocessor(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    video.add_preprocessor(byotrack.video.ChannelProjection("mean"))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # ensure no "already normalized" warning
        normalized = video.normalize()

    assert normalized.dtype == np.float32


## set_transform (deprecated)


def test_video_set_transform_warns_deprecated(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        video.set_transform(byotrack.VideoTransformConfig())

    assert len(video._preprocessors) == 0


def test_video_set_transform_aggregate(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        video.set_transform(byotrack.VideoTransformConfig(aggregate=True, selected_channel=None))

    assert video.shape[-1] == 1
    assert isinstance(video._preprocessors[0], byotrack.video.ChannelProjection)


def test_video_set_transform_normalize(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        video.set_transform(byotrack.VideoTransformConfig(normalize=True))

    assert video.dtype == np.float32
    assert isinstance(video._preprocessors[0], byotrack.video.IntensityNormalizer)


def test_video_set_transform_aggregate_and_normalize(video_2d: np.ndarray):
    video = byotrack.Video(video_2d)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        video.set_transform(byotrack.VideoTransformConfig(normalize=True, aggregate=True, selected_channel=1))

    assert video.dtype == np.float32
    assert video.shape[-1] == 1
    assert len(video._preprocessors) == 2
    assert isinstance(video._preprocessors[0], byotrack.video.ChannelProjection)
    assert isinstance(video._preprocessors[1], byotrack.video.IntensityNormalizer)


## expand_ellipsis


def test_expand_ellipsis_no_ellipsis():
    slices = (slice(1, 5), slice(2, 8))
    result = expand_ellipsis(slices, 4)
    assert result == slices


def test_expand_ellipsis_at_start():
    result = expand_ellipsis((Ellipsis, slice(1, 5)), 4)
    assert result == (slice(None), slice(None), slice(None), slice(1, 5))


def test_expand_ellipsis_at_end():
    result = expand_ellipsis((slice(1, 5), Ellipsis), 4)
    assert result == (slice(1, 5), slice(None), slice(None), slice(None))


def test_expand_ellipsis_in_middle():
    result = expand_ellipsis((slice(1, 5), Ellipsis, slice(2, 4)), 4)
    assert result == (slice(1, 5), slice(None), slice(None), slice(2, 4))


def test_expand_ellipsis_alone():
    result = expand_ellipsis((Ellipsis,), 4)
    assert result == (slice(None), slice(None), slice(None), slice(None))


def test_expand_ellipsis_double_raises():
    with pytest.raises(IndexError, match="single ellipsis"):
        expand_ellipsis((Ellipsis, slice(1, 5), Ellipsis), 4)


## _handle_integer_slicing


def test_handle_integer_slicing_no_integers():
    slices = (slice(1, 5), slice(2, 8), slice(None))
    result_slices, projection = _handle_integer_slicing(slices, ndim=4)
    assert projection == (-1, 0)
    assert result_slices == slices


def test_handle_integer_slicing_single_int_z_axis():
    # For 3D video (ndim=5): (D, H, W, C) -> int on D (index 0)
    slices = (2, slice(None), slice(None))
    result_slices, projection = _handle_integer_slicing(slices, ndim=5)
    assert projection == (0, 2)
    assert result_slices == (slice(None), slice(None))


def test_handle_integer_slicing_int_on_channel_raises():
    slices = (slice(None), slice(None), 1)  # int on channel (1D video (ndim=4): H, W, C)
    with pytest.raises(IndexError, match="Channel axis"):
        _handle_integer_slicing(slices, ndim=4)


def test_handle_integer_slicing_int_on_2d_video_raises():
    slices = (5,)
    with pytest.raises(IndexError, match="3D"):
        _handle_integer_slicing(slices, ndim=4)


def test_handle_multiple_integer_slicing_on_3d_video_raises():
    slices = (5, slice(None), 3)
    with pytest.raises(IndexError, match="3D"):
        _handle_integer_slicing(slices, ndim=5)


## compose_slice


def test_compose_slice_basic_forward():
    # video[1:9][::2] == video[1:9:2]
    s1 = slice(1, 9)
    s2 = slice(None, None, 2)
    composed = compose_slice(s1, s2, 10)
    expected = slice(1, 9, 2)
    # Verify by comparing lengths and element positions
    assert composed.indices(10) == expected.indices(10)


def test_compose_slice_step_multiplication():
    s1 = slice(0, 10, 2)  # [0, 2, 4, 6, 8]
    s2 = slice(0, None, 2)  # every other → [0, 4, 8]
    composed = compose_slice(s1, s2, 10)
    result = list(range(*composed.indices(10)))
    expected = list(range(*slice(0, 10, 4).indices(10)))
    assert result == expected


def test_compose_slice_negative_stop_edge_case():
    # Compose reverse slice such that stop < 0 (should use None, not negative)
    s1 = slice(5, None, -1)  # [5, 4, 3, 2, 1, 0]
    s2 = slice(None, None, 2)  # every other → [5, 3, 1]
    composed = compose_slice(s1, s2, 10)
    result = list(range(*composed.indices(10)))
    # Direct: range(5, None, -2) → interpreted from indices
    direct = list(range(*slice(5, None, -2).indices(10)))
    assert result == direct


def test_compose_slice_full_then_partial():
    s1 = slice(None)  # whole array
    s2 = slice(2, 8)
    composed = compose_slice(s1, s2, 10)
    result = list(range(*composed.indices(10)))
    assert result == list(range(2, 8))


## video_length / video_shape / video_dtype


def test_video_utils_on_ndarray(video_3d: np.ndarray):
    assert video_length(video_3d) == video_3d.shape[0]
    assert video_shape(video_3d) == video_3d.shape
    assert video_dtype(video_3d) == video_3d.dtype

    # Test also length 0
    assert video_length(video_3d[:0]) == 0
    assert video_shape(video_3d[:0])[1:] == video_3d.shape[1:]
    assert video_dtype(video_3d[:0]) == video_3d.dtype


def test_video_utils_on_list_of_ndarray(video_2d: np.ndarray):
    frames = list(video_2d)

    assert video_length(frames) == video_2d.shape[0]
    assert video_shape(frames) == video_2d.shape
    assert video_dtype(frames) == video_2d.dtype

    # Test also length 0
    assert video_length(frames[:0]) == 0

    with pytest.raises(ValueError, match="empty"):
        video_shape(frames[:0])

    with pytest.raises(ValueError, match="empty"):
        video_dtype(frames[:0])


def test_video_utils_on_byotrack_video(video_3d: np.ndarray):
    video = byotrack.Video(video_3d)

    assert video_length(video) == video_3d.shape[0]
    assert video_shape(video) == video_3d.shape
    assert video_dtype(video) == video_3d.dtype

    # Test also length 0
    assert video_length(video[:0]) == 0
    assert video_shape(video[:0])[1:] == video_3d.shape[1:]
    assert video_dtype(video[:0]) == video_3d.dtype
