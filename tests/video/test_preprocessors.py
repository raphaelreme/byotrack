from __future__ import annotations

import sys

import numpy as np
import pytest
import scipy.ndimage as ndi

from byotrack.video.preprocessor.channel_projection import ChannelProjection
from byotrack.video.preprocessor.normalizer import IntensityNormalizer
from byotrack.video.preprocessor.preprocessor import VideoPreprocessor
from byotrack.video.preprocessor.registrator import Registrator
from byotrack.video.preprocessor.slicer import FrameSlicer
from byotrack.video.preprocessor.spatial_projection import SpatialProjection

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class IdentityPreprocessor(VideoPreprocessor):
    """Minimal concrete VideoPreprocessor for base-class tests."""

    @override
    def preprocess_frame(self, frame: np.ndarray, frame_id: int = 0) -> np.ndarray:
        return frame


## Base class


def test_preprocessor_shape_before_init_raises():
    proc = IdentityPreprocessor()
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = proc.shape


def test_preprocessor_initialize_ndarray_2d(video_2d: np.ndarray):
    proc = IdentityPreprocessor()
    proc.initialize(video_2d)
    assert proc.shape == (20, 30, 3)
    assert proc.dtype == video_2d.dtype


def test_preprocessor_initialize_ndarray_3d(video_3d: np.ndarray):
    proc = IdentityPreprocessor()
    proc.initialize(video_3d)
    assert proc.shape == (5, 20, 30, 2)
    assert proc.dtype == video_3d.dtype


def test_preprocessor_initialize_list_of_frames(video_2d: np.ndarray):
    frames = [video_2d[i] for i in range(len(video_2d))]
    proc = IdentityPreprocessor()
    proc.initialize(frames)
    assert proc.shape == (20, 30, 3)
    assert proc.dtype == video_2d.dtype


def test_preprocessor_initialize_empty_video_raises():
    proc = IdentityPreprocessor()
    with pytest.raises(ValueError, match="empty"):
        proc.initialize([])


def test_preprocessor_initialize_zero_pixel_frame_raises():
    proc = IdentityPreprocessor()
    with pytest.raises(ValueError, match="No pixel"):
        proc.initialize(np.zeros((5, 0, 10, 1), dtype=np.uint8))


def test_preprocessor_preprocess_video_offline_ndarray(video_2d: np.ndarray):
    proc = IdentityPreprocessor()
    result = proc.preprocess_video(video_2d)
    assert isinstance(result, np.ndarray)
    assert result.shape == video_2d.shape
    assert (result == video_2d).all()


def test_preprocessor_preprocess_video_offline_list(video_2d: np.ndarray):
    frames = [video_2d[i] for i in range(len(video_2d))]
    proc = IdentityPreprocessor()
    result = proc.preprocess_video(frames)
    assert isinstance(result, np.ndarray)
    assert result.shape == video_2d.shape
    assert (result == video_2d).all()


## Intensity normalization


def test_normalizer_hard_clip_output_range_2d(video_2d: np.ndarray):
    norm = IntensityNormalizer(q_min=0.0, q_max=1.0)
    norm.initialize(video_2d)
    for i, frame in enumerate(video_2d):
        out = norm.preprocess_frame(frame, i)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    assert norm.mini.shape == (video_2d.shape[-1],)
    assert norm.maxi.shape == (video_2d.shape[-1],)
    assert norm.max.shape == (1,)  # Not set
    assert (norm.max == 1.0).all()


def test_normalizer_hard_clip_output_range_3d(video_3d: np.ndarray):
    norm = IntensityNormalizer(q_min=0.0, q_max=1.0)
    norm.initialize(video_3d)
    for i, frame in enumerate(video_3d):
        out = norm.preprocess_frame(frame.copy(), i)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    assert norm.mini.shape == (video_3d.shape[-1],)
    assert norm.maxi.shape == (video_3d.shape[-1],)
    assert norm.max.shape == (1,)  # Not set
    assert (norm.max == 1.0).all()


def test_normalizer_output_dtype_is_float32(video_2d: np.ndarray):
    norm = IntensityNormalizer(q_min=0.0, q_max=1.0)
    norm.initialize(video_2d)
    out = norm.preprocess_frame(video_2d[0])
    assert out.dtype == norm.dtype == np.float32


def test_normalizer_uniform_channel_no_division_by_zero():
    # All pixels in a channel are identical → mini == maxi
    uniform = np.full((5, 10, 10, 1), 128, dtype=np.uint8)
    norm = IntensityNormalizer(q_min=0.0, q_max=1.0)
    norm.initialize(uniform)
    out = norm.preprocess_frame(uniform[0].copy())
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))
    # When mini==maxi the result should be 0
    assert (out == 0).all()


def test_normalizer_custom_quantiles():
    video = np.random.randint(50, 200, (5, 10, 10, 1), dtype=np.uint8)
    video[0, 7, 5, 0] = 250
    video[0, 7, 6, 0] = 230
    video[0, 2, 5, 0] = 10
    video[0, 2, 6, 0] = 30

    norm = IntensityNormalizer(q_min=0.1, q_max=0.9)
    norm.initialize(video)
    out = norm.preprocess_frame(video[0])

    assert out.min() >= 0.0
    assert out.max() <= 1.0

    assert out[7, 5, 0] == 1.0  # The max is clipped
    assert out[7, 6, 0] == 1.0  # 2nd max is still clipped
    assert out[2, 5, 0] == 0.0  # The min is clipped
    assert out[2, 6, 0] == 0.0  # 2nd min is still clipped


def test_normalizer_multichannel_independent():
    # Create video where channels have clearly different ranges
    video = np.zeros((5, 10, 10, 2), dtype=np.uint8)
    video[..., 0] = 50  # channel 0: low values
    video[..., 1] = 200  # channel 1: high values
    norm = IntensityNormalizer(q_min=0.0, q_max=1.0)
    norm.initialize(video)
    assert norm.mini.shape == (2,)
    assert norm.maxi.shape == (2,)
    # Both channels should be constant → mini==maxi
    assert norm.mini[0] == norm.maxi[0]
    assert norm.mini[1] == norm.maxi[1]

    out = norm.preprocess_frame(video[0])

    # Both channels should be map to 0
    assert (out == 0).all()


def test_normalizer_smooth_clip_output_range(video_2d: np.ndarray):
    norm = IntensityNormalizer(q_min=0.0, q_max=0.8, smooth_clip=0.5)
    norm.initialize(video_2d)
    for i, frame in enumerate(video_2d):
        out = norm.preprocess_frame(frame.copy(), i)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    assert norm.max.shape == (video_2d.shape[-1],)
    assert (norm.max >= 1.0).all()


def test_normalizer_compute_stats_on_limits_frames():
    # Use only 2 frames for stats; the 3rd frame has extreme values
    video = np.zeros((5, 10, 10, 1), dtype=np.uint8)
    video[:2] = 100
    video[:2, 0] = 80
    video[2:] = 255
    norm = IntensityNormalizer(q_min=0.0, q_max=1.0, compute_stats_on=2)
    norm.initialize(video)

    # mini and maxi should reflect only the first 2 frames (all 100)
    assert norm.mini[0] == 80
    assert norm.maxi[0] == 100

    out = norm.preprocess_frame(video[-1])

    assert (out == 1).all()  # Clipped


def test_normalizer_preprocess_video(video_2d: np.ndarray):
    norm = IntensityNormalizer(q_min=0.0, q_max=1.0)
    result = norm.preprocess_video(video_2d)
    assert result.shape == video_2d.shape
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


## ChannelProjection


def test_channel_projection_mean(video_2d: np.ndarray):
    proj = ChannelProjection("mean")
    proj.initialize(video_2d)
    out = proj.preprocess_frame(video_2d[0])

    assert out.shape == proj.shape == (20, 30, 1)
    assert out.dtype == proj.dtype == video_2d.dtype  # Mean preserve dtype
    assert (out == video_2d[0].mean(-1, keepdims=True, dtype=video_2d.dtype)).all()


def test_channel_projection_min(video_2d: np.ndarray):
    proj = ChannelProjection("min")
    proj.initialize(video_2d)
    out = proj.preprocess_frame(video_2d[0])

    assert out.shape == proj.shape == (20, 30, 1)
    assert out.dtype == proj.dtype == video_2d.dtype
    assert (out == video_2d[0].min(-1, keepdims=True)).all()


def test_channel_projection_max(video_2d: np.ndarray):
    proj = ChannelProjection("max")
    proj.initialize(video_2d)
    out = proj.preprocess_frame(video_2d[0])

    assert out.shape == proj.shape == (20, 30, 1)
    assert out.dtype == proj.dtype == video_2d.dtype
    assert (out == video_2d[0].max(-1, keepdims=True)).all()


def test_channel_projection_select(video_2d: np.ndarray):
    proj = ChannelProjection("select", selected=1)
    proj.initialize(video_2d)
    out = proj.preprocess_frame(video_2d[0])

    assert out.shape == proj.shape == (20, 30, 1)
    assert out.dtype == proj.dtype == video_2d.dtype
    assert (out == video_2d[0, ..., 1:2]).all()


def test_channel_projection_select_out_of_bounds_raises(video_2d: np.ndarray):
    proj = ChannelProjection("select", selected=10)
    with pytest.raises(IndexError, match="out of bounds"):
        proj.initialize(video_2d)


def test_channel_projection_3d_video(video_3d: np.ndarray):
    proj = ChannelProjection("max")
    proj.initialize(video_3d)
    out = proj.preprocess_frame(video_3d[0])

    assert out.shape == proj.shape == (5, 20, 30, 1)
    assert out.dtype == proj.dtype == video_3d.dtype
    assert (out == video_3d[0].max(-1, keepdims=True)).all()


@pytest.mark.parametrize("method", ["mean", "min", "max", "select"])
def test_channel_projection_preprocess_video_ndarray_vs_seq(method, video_2d: np.ndarray):
    proj = ChannelProjection(method)
    out_array = proj.preprocess_video(video_2d)

    # Compare against frame-by-frame
    frames = [video_2d[i] for i in range(len(video_2d))]
    out_list = proj.preprocess_video(frames)

    assert out_array.shape == (10, 20, 30, 1)
    assert (out_array == out_list).all()


def test_channel_projection_do_no_overflow():
    video = np.full((3, 5, 7, 4), 200, dtype=np.uint8)
    proj = ChannelProjection(method="mean")
    proj.initialize(video)
    out = proj.preprocess_frame(video[0])

    assert (out == 200).all()


## FrameSlicer


def test_frame_slicer_crop_2d(video_2d: np.ndarray):
    slicer = FrameSlicer((slice(5, 15), slice(10, 20)))
    slicer.initialize(video_2d)
    out = slicer.preprocess_frame(video_2d[0])

    assert out.shape == slicer.shape == (10, 10, 3)
    assert out.dtype == slicer.dtype == video_2d.dtype
    assert (out == video_2d[0, 5:15, 10:20]).all()


def test_frame_slicer_crop_3d(video_3d: np.ndarray):
    slicer = FrameSlicer((slice(None), slice(10, 20), slice(None), slice(None, None, -1)))
    slicer.initialize(video_3d)
    out = slicer.preprocess_frame(video_3d[0])

    assert out.shape == slicer.shape == (5, 10, 30, 2)
    assert out.dtype == slicer.dtype == video_3d.dtype
    assert (out == video_3d[0, :, 10:20, :, ::-1]).all()


def test_frame_slicer_too_many_slices_raises(video_2d: np.ndarray):
    slicer = FrameSlicer((slice(0, 5),) * 10)
    with pytest.raises(IndexError, match="Too many indices"):
        slicer.initialize(video_2d)


def test_frame_slicer_empty_slice_raises(video_2d: np.ndarray):
    slicer = FrameSlicer((slice(5, 5),))  # empty slice => length 0
    with pytest.raises(ValueError, match="empty"):
        slicer.initialize(video_2d)


def test_frame_slicer_preprocess_video_ndarray_vs_list(video_2d: np.ndarray):
    slicer = FrameSlicer((slice(5, 15), slice(10, 20)))
    out_array = slicer.preprocess_video(video_2d)

    # Compare against frame-by-frame
    frames = [video_2d[i] for i in range(len(video_2d))]
    out_list = slicer.preprocess_video(frames)

    assert out_array.shape == (10, 10, 10, 3)
    assert (out_array == out_list).all()


## SpatialProjection


def test_spatial_projection_axis_0_equivalent_to_z_and_d():
    proj_z = SpatialProjection("Z")
    proj_d = SpatialProjection("D")
    proj_int = SpatialProjection(0)
    assert proj_z.axis == proj_d.axis == proj_int.axis == 0


def test_spatial_projection_axis_1_equivalent_to_y_and_h():
    proj_z = SpatialProjection("Y")
    proj_d = SpatialProjection("H")
    proj_int = SpatialProjection(1)
    assert proj_z.axis == proj_d.axis == proj_int.axis == 1


def test_spatial_projection_axis_2_equivalent_to_x_and_w():
    proj_z = SpatialProjection("W")
    proj_d = SpatialProjection("W")
    proj_int = SpatialProjection(2)
    assert proj_z.axis == proj_d.axis == proj_int.axis == 2


def test_spatial_projection_d_max_equivalent_to_z():
    proj_z = SpatialProjection("Z")
    proj_d = SpatialProjection("D")
    assert proj_z.axis == proj_d.axis == 0


def test_spatial_projection_negative_axis_is_converted():
    proj = SpatialProjection(-2)
    assert proj.axis == 1


def test_spatial_projection_out_of_bounds_axis_raises():
    with pytest.raises(ValueError, match="in \\(0, 1, 2\\)"):
        SpatialProjection(3)

    with pytest.raises(ValueError, match="in \\(0, 1, 2\\)"):
        SpatialProjection("Q")


def test_spatial_projection_on_2d_video_raises(video_2d: np.ndarray):
    proj = SpatialProjection()
    with pytest.raises(ValueError, match="3D"):
        proj.initialize(video_2d)


def test_spatial_projection_on_z_mean(video_3d: np.ndarray):
    proj = SpatialProjection("Z", "mean")
    proj.initialize(video_3d)
    out = proj.preprocess_frame(video_3d[0])

    assert out.shape == proj.shape == video_3d.shape[2:]
    assert out.dtype == proj.dtype == video_3d.dtype  # Mean preserve dtype
    assert (out == video_3d[0].mean(0).astype(video_3d.dtype)).all()


def test_spatial_projection_on_z_min(video_3d: np.ndarray):
    proj = SpatialProjection("Z", "min")
    proj.initialize(video_3d)
    out = proj.preprocess_frame(video_3d[0])

    assert out.shape == proj.shape == video_3d.shape[2:]
    assert out.dtype == proj.dtype == video_3d.dtype
    assert (out == video_3d[0].min(0)).all()


def test_spatial_projection_on_z_max(video_3d: np.ndarray):
    proj = SpatialProjection("Z", "max")
    proj.initialize(video_3d)
    out = proj.preprocess_frame(video_3d[0])

    assert out.shape == proj.shape == video_3d.shape[2:]
    assert out.dtype == proj.dtype == video_3d.dtype
    assert (out == video_3d[0].max(0)).all()


def test_spatial_projection_on_z_select(video_3d: np.ndarray):
    proj = SpatialProjection("Z", "select", selected=3)
    proj.initialize(video_3d)
    out = proj.preprocess_frame(video_3d[0])

    assert out.shape == proj.shape == video_3d.shape[2:]
    assert out.dtype == proj.dtype == video_3d.dtype
    assert (out == video_3d[0, 3]).all()


def test_spatial_projection_select_out_of_bounds_raises(video_3d: np.ndarray):
    proj = SpatialProjection("Z", "select", selected=100)
    with pytest.raises(IndexError, match="out of bounds"):
        proj.initialize(video_3d)


def test_spatial_projection_on_y_max(video_3d: np.ndarray):
    proj = SpatialProjection("Y", "max")
    proj.initialize(video_3d)
    out = proj.preprocess_frame(video_3d[0])

    expected_shape = video_3d.shape[1:]
    expected_shape = (expected_shape[0], expected_shape[2], expected_shape[3])

    assert out.shape == proj.shape == expected_shape
    assert out.dtype == proj.dtype == video_3d.dtype
    assert (out == video_3d[0].max(1)).all()


def test_spatial_projection_on_x_max(video_3d: np.ndarray):
    proj = SpatialProjection("X", "max")
    proj.initialize(video_3d)
    out = proj.preprocess_frame(video_3d[0])

    expected_shape = video_3d.shape[1:]
    expected_shape = (expected_shape[0], expected_shape[1], expected_shape[3])

    assert out.shape == proj.shape == expected_shape
    assert out.dtype == proj.dtype == video_3d.dtype
    assert (out == video_3d[0].max(2)).all()


@pytest.mark.parametrize(("method", "axis"), [("mean", 0), ("min", 1), ("max", 2), ("select", 0)])
def test_spatial_projection_preprocess_video_ndarray_vs_seq(method, axis: int, video_3d: np.ndarray):
    proj = SpatialProjection(axis, method)
    out_array = proj.preprocess_video(video_3d)

    # Compare against frame-by-frame
    frames = [video_3d[i] for i in range(len(video_3d))]
    out_list = proj.preprocess_video(frames)

    assert out_array.shape == (len(video_3d), *proj.shape)
    assert (out_array == out_list).all()


def test_spatial_projection_do_no_overflow():
    video = np.full((3, 5, 5, 7, 2), 200, dtype=np.uint8)
    proj = SpatialProjection("X", method="mean")
    proj.initialize(video)
    out = proj.preprocess_frame(video[0])

    assert (out == 200).all()


## Registrator


def test_registrator_default_reference_frame_is_first_frame(video_2d: np.ndarray):
    reg = Registrator()
    reg.initialize(video_2d)

    assert (reg.reference_frame == video_2d[0]).all()
    assert reg.shape == video_2d.shape[1:]
    assert reg.dtype == video_2d.dtype


def test_registrator_explicit_reference_frame_is_kept(video_2d: np.ndarray):
    reference = video_2d[1].copy()
    reg = Registrator(reference_frame=reference)
    reg.initialize(video_2d)

    assert reg.reference_frame is reference


def test_registrator_invalid_reference_channel_raises(video_2d: np.ndarray):
    reg = Registrator(reference_channel=10)
    with pytest.raises(IndexError, match="out of bounds"):
        reg.initialize(video_2d)


def test_registrator_preprocess_before_init_raises(video_2d: np.ndarray):
    reg = Registrator()
    with pytest.raises(ValueError, match="not initialized"):
        reg.preprocess_frame(video_2d[0])


def test_registrator_preprocess_video_shape_and_dtype_2d(video_2d: np.ndarray):
    reg = Registrator()
    out = reg.preprocess_video(video_2d)

    assert out.shape == video_2d.shape
    assert out.dtype == video_2d.dtype


def test_registrator_preprocess_video_shape_and_dtype_3d(video_3d: np.ndarray):
    reg = Registrator()
    out = reg.preprocess_video(video_3d)

    assert out.shape == video_3d.shape
    assert out.dtype == video_3d.dtype


def test_registrator_recovers_known_integer_shift():
    # Build a small blurred target, and a second frame shifted by a known translation
    target = np.zeros((40, 45), dtype=np.float32)
    target[15:25, 18:30] = 1.0
    target = ndi.gaussian_filter(target, 1.5)

    shifted = ndi.shift(target, (3, -5), order=1, mode="nearest")
    video = np.stack([target, shifted], axis=0)[..., None]  # (T, H, W, C=1)

    reg = Registrator(interpolation_order=0, upsample_factor=1)
    reg.initialize(video)
    out = reg.preprocess_frame(video[1].copy(), 1)

    # Nearest interpolation (order=0) recovers the reference exactly here
    assert np.array_equal(out, video[0])


def test_registrator_applies_same_shift_to_all_channels():
    target = np.zeros((30, 30), dtype=np.float32)
    target[10:15, 12:20] = 1.0
    target = ndi.gaussian_filter(target, 1.0)

    shifted = ndi.shift(target, (2, -3), order=1, mode="nearest")

    reference_frame = np.stack([target, target * 2], axis=-1)
    moving_frame = np.stack([shifted, shifted * 2], axis=-1)
    video = np.stack([reference_frame, moving_frame], axis=0)

    reg = Registrator(reference_channel=0)
    reg.initialize(video)
    out = reg.preprocess_frame(moving_frame.copy(), 1)

    assert np.allclose(out[..., 0], reference_frame[..., 0])
    assert np.allclose(out[..., 1], reference_frame[..., 1])
