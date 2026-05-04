from __future__ import annotations

import numpy as np

from byotrack.api.optical_flow.optical_flow import DummyOpticalFlow

## Preprocessing


def test_preprocess_2d_output_shape_halved():
    of = DummyOpticalFlow(downscale=2.0)
    frame = np.ones((20, 30, 3), dtype=np.float32)
    result = of.preprocess(frame)
    assert result.shape == (10, 15, 3)


def test_preprocess_3d_output_shape_halved():
    of = DummyOpticalFlow(downscale=2.0)
    frame = np.ones((8, 20, 30, 2), dtype=np.float32)
    result = of.preprocess(frame)
    assert result.shape == (4, 10, 15, 2)


def test_preprocess_output_dtype_float32():
    of = DummyOpticalFlow(downscale=2.0)
    frame = np.ones((10, 10, 1), dtype=np.uint8)
    result = of.preprocess(frame)
    assert result.dtype == np.float32


def test_preprocess_channels_not_downscaled():
    of = DummyOpticalFlow(downscale=2.0)
    frame = np.ones((20, 20, 5), dtype=np.float32)
    result = of.preprocess(frame)
    assert result.shape[-1] == 5  # channels unchanged


def test_preprocess_downscale_1_preserves_shape():
    of = DummyOpticalFlow(downscale=1.0)
    frame = np.ones((16, 16, 2), dtype=np.float32)
    result = of.preprocess(frame)
    assert result.shape[:2] == (16, 16)


def test_preprocess_per_axis_downscale():
    of = DummyOpticalFlow(downscale=np.array([2.0, 4.0]))
    frame = np.ones((20, 40, 1), dtype=np.float32)
    result = of.preprocess(frame)
    # rows halved (÷2), cols quartered (÷4)
    assert result.shape[0] == 10
    assert result.shape[1] == 10


def test_preprocess_without_blur():
    of = DummyOpticalFlow(downscale=1.0, blur=0.0)
    frame = np.ones((20, 40, 1), dtype=np.float32)
    result = of.preprocess(frame)

    assert np.allclose(result, frame)


def test_preprocess_with_per_axis_blur():
    of = DummyOpticalFlow(downscale=1.0, blur=np.array([0.0, 1.0]))
    frame = np.zeros((20, 20, 1), dtype=np.float32)
    frame[10, 10] = 1.0
    result = of.preprocess(frame)

    assert result[8, 10] == 0.0
    assert result[10, 8] > 0.0


## Test dummy compute


def test_dummy_compute_2d_output_shape():
    of = DummyOpticalFlow()
    frame = np.ones((10, 15, 3), dtype=np.float32)
    flow = of.compute(frame, frame)
    assert flow.shape == (2, 10, 15)


def test_dummy_compute_3d_output_shape():
    of = DummyOpticalFlow()
    frame = np.ones((5, 10, 15, 2), dtype=np.float32)
    flow = of.compute(frame, frame)
    assert flow.shape == (3, 5, 10, 15)


def test_dummy_compute_all_zeros():
    of = DummyOpticalFlow()
    frame = np.ones((8, 12, 1), dtype=np.float32)
    flow = of.compute(frame, frame)
    assert (flow == 0).all()


## Test flow_at


def test_flow_at_constant_2d():
    of = DummyOpticalFlow(downscale=2.0)
    flow_map = np.zeros((2, 10, 10), dtype=np.float32)
    cte_flow = np.random.default_rng(1337).normal(0, 3, 2)
    flow_map[:] = cte_flow[:, None, None]

    points = np.array([[4.0, 6.0], [8.5, 7.5]], dtype=np.float32)
    displacement = of.flow_at(flow_map, points)

    assert displacement.shape == (2, 2)

    # Account for scaling
    assert np.allclose(displacement, cte_flow * 2)


def test_flow_at_constant_3d():
    of = DummyOpticalFlow(downscale=np.array([1, 2, 3.0]))
    flow_map = np.zeros((3, 10, 10, 10), dtype=np.float32)
    cte_flow = np.random.default_rng(1337).normal(0, 3, 3)
    flow_map[:] = cte_flow[:, None, None, None]

    points = np.array([[4.0, 6.0, 8.0], [8.5, 7.5, 6.5]], dtype=np.float32)
    displacement = of.flow_at(flow_map, points)

    assert displacement.shape == (2, 3)

    # Account for scaling
    assert np.allclose(displacement, cte_flow * of.downscale)


def test_flow_at_empty_returns_empty():
    of = DummyOpticalFlow(downscale=2.0)
    flow_map = np.zeros((2, 10, 10), dtype=np.float32)

    points = np.zeros((0, 2))
    displacement = of.flow_at(flow_map, points)

    assert displacement.shape == (0, 2)


def test_flow_at_interpolates():
    of = DummyOpticalFlow(downscale=1.0)
    flow_map = np.zeros((2, 10, 10), dtype=np.float32)
    flow_map[:, :5, :5] = 1.0

    points = np.array([[5.5, 5.5]])
    displacement = of.flow_at(flow_map, points)

    assert displacement.shape == (1, 2)
    assert (displacement > 0.0).all()
    assert (displacement < 1.0).all()


## Transforms


def test_transform_zero_flow_points_unchanged_2d():
    of = DummyOpticalFlow(downscale=2.0)
    flow_map = np.zeros((2, 10, 10), dtype=np.float32)
    points = np.random.default_rng(1337).random((5, 2))
    result = of.transform(flow_map, points)
    assert np.allclose(result, points)


def test_transform_zero_flow_points_unchanged_3d():
    of = DummyOpticalFlow(downscale=2.0)
    flow_map = np.zeros((3, 5, 10, 10), dtype=np.float32)
    points = np.random.default_rng(1337).random((1, 3))
    result = of.transform(flow_map, points)
    assert np.allclose(result, points)


def test_transform_nonzero_flow_displaces_correctly():
    of = DummyOpticalFlow(downscale=2.0)

    # Move by a constant 1 in direction 0 (=> 2 in original coordinates)
    flow_map = np.zeros((2, 10, 10), dtype=np.float32)
    flow_map[0] = 1.0

    points = np.array([[4.0, 4.0]], dtype=np.float32)

    result = of.transform(flow_map, points)

    assert np.allclose(result[0, 0], 6.0)
    assert np.allclose(result[0, 1], 4.0)


def test_transform_empty_is_empty():
    of = DummyOpticalFlow(downscale=2.0)
    flow_map = np.zeros((3, 10, 10, 10), dtype=np.float32)
    points = np.zeros((0, 3))

    result = of.transform(flow_map, points)

    assert result.shape == (0, 3)


## Warp


def test_warp_zero_flow_2d():
    of = DummyOpticalFlow(downscale=2.0)
    frame_2d = np.random.default_rng(0).random((16, 16, 1)).astype(np.float32)
    flow_map = np.zeros((2, 8, 8), dtype=np.float32)
    warped = of.warp(flow_map, frame_2d)
    assert np.allclose(warped, frame_2d)


def test_warp_zero_flow_3d():
    of = DummyOpticalFlow(downscale=1.0)
    frame_3d = np.random.default_rng(1).random((5, 10, 10, 1)).astype(np.float32)
    flow_map = np.zeros((3, 5, 10, 10), dtype=np.float32)
    warped = of.warp(flow_map, frame_3d)
    assert np.allclose(warped, frame_3d)


def test_warp_channels_preserved():
    of = DummyOpticalFlow(downscale=2.0)
    frame = np.ones((8, 8, 4), dtype=np.float32)
    frame[:, :, 1] = 2.0
    frame[:, :, 2] = 3.0
    frame[:, :, 3] = 4.0
    flow_map = np.zeros((2, 8, 8), dtype=np.float32)
    warped = of.warp(flow_map, frame)

    assert warped.shape[-1] == 4
    assert np.allclose(warped, frame)


def test_warp_constant_flow_shifts_content():
    of = DummyOpticalFlow(downscale=1.0)

    # Moving frame with a bright pixel at (5, 5)
    moving = np.zeros((10, 10, 1), dtype=np.float32)
    moving[5, 5, 0] = 1.0

    # Flow: shift all pixels by 2 in i-direction (the 1.0 at (5,5) came from (3,5) in ref)
    flow_map = np.zeros((2, 10, 10), dtype=np.float32)
    flow_map[0] = 2.0

    ref = of.warp(flow_map, moving)

    # The ref image should have it's bright pixel in (3,5)
    assert ref[3, 5, 0] == 1.0
    ref[3, 5, 0] = 0.0
    assert (ref == 0.0).all()


def test_warp_flow_many_to_one():
    of = DummyOpticalFlow(downscale=1.0)

    # Moving frame with a single bright pixel at (5, 5)
    moving = np.zeros((10, 10, 1), dtype=np.float32)
    moving[5, 5, 0] = 1.0

    # Flow: Let's have several pixels coming to (5, 5)
    flow_map = np.zeros((2, 10, 10), dtype=np.float32)
    flow_map[0, 4, 5] = 1
    flow_map[1, 5, 4] = 1
    flow_map[0, 6, 5] = -1
    # flow_map[5, 6, 1] = -1  # To check asymmetry

    ref = of.warp(flow_map, moving)

    # Each pixel (4, 5), (5, 4), (6, 5), (5, 5) should be bright
    # NOTE: Intensity sum is not preserved
    assert (ref[4:6, 5] == 1.0).all()
    assert (ref[5, 4] == 1.0).all()
    assert ref.sum() == 4.0
