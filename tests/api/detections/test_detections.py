from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

import byotrack
from byotrack.api.detections.detections import (
    compress,
    decompress,
    draw_disk_2d,
    draw_disk_3d,
    fast_relabel,
    labels_of,
    relabel_consecutive,
)

if TYPE_CHECKING:
    import pathlib


## First let's test the helpers


def test_labels_of_consecutive():
    seg = torch.zeros(10, 10, dtype=torch.int32)
    seg[0:3, 0:3] = 1
    seg[4:7, 4:7] = 2
    seg[7:10, 7:10] = 3
    result = labels_of(seg)
    assert result.tolist() == [0, 1, 2]


def test_labels_of_non_consecutive():
    seg = torch.zeros(10, 10, dtype=torch.int32)
    seg[0:3, 0:3] = 1
    seg[5:8, 5:8] = 3
    result = labels_of(seg)
    assert result.tolist() == [0, 2]


def test_labels_of_background_only():
    seg = torch.zeros(10, 10, dtype=torch.int32)
    result = labels_of(seg)
    assert len(result) == 0


def test_labels_of_without_background():
    seg = torch.ones(10, 10, dtype=torch.int32)
    result = labels_of(seg)
    assert result.tolist() == [0]


def test_labels_of_returns_torch_for_torch_input():
    seg = torch.zeros(10, 10, dtype=torch.int32)
    seg[0:3, 0:3] = 1
    result = labels_of(seg)
    assert isinstance(result, torch.Tensor)


def test_labels_of_returns_numpy_for_numpy_input():
    seg = np.zeros((10, 10), dtype=np.int32)
    seg[0:3, 0:3] = 1
    result = labels_of(seg)
    assert isinstance(result, np.ndarray)
    assert result.tolist() == [0]


def test_relabel_consecutive_inplace_pt():
    seg = torch.zeros(10, 10, dtype=torch.int32)
    seg[0:3, 0:3] = 1
    seg[5:8, 5:8] = 3
    original_id = id(seg)
    result = relabel_consecutive(seg, inplace=True)
    assert id(result) == original_id
    assert (result[0:3, 0:3] == 1).all()
    assert (result[5:8, 5:8] == 2).all()


def test_relabel_consecutive_inplace_np():
    seg = np.zeros((10, 10), dtype=np.int32)
    seg[0:3, 0:3] = 1
    seg[5:8, 5:8] = 3
    original_id = id(seg)
    result = relabel_consecutive(seg, inplace=True)
    assert id(result) == original_id
    assert (result[0:3, 0:3] == 1).all()
    assert (result[5:8, 5:8] == 2).all()


def test_relabel_consecutive_copy_pt():
    seg = torch.zeros(10, 10, dtype=torch.int32)
    seg[0:3, 0:3] = 3
    result = relabel_consecutive(seg, inplace=False)
    assert id(result) != id(seg)
    assert (result[0:3, 0:3] == 1).all()
    assert (seg[0:3, 0:3] == 3).all()  # original unchanged


def test_relabel_consecutive_copy_np():
    seg = np.zeros((10, 10), dtype=np.int32)
    seg[0:3, 0:3] = 3
    result = relabel_consecutive(seg, inplace=False)
    assert id(result) != id(seg)
    assert (result[0:3, 0:3] == 1).all()
    assert (seg[0:3, 0:3] == 3).all()  # original unchanged


def test_relabel_consecutive_already_consecutive():
    seg = torch.zeros(10, 10, dtype=torch.int32)
    seg[0:3, 0:3] = 2
    seg[5:8, 5:8] = 1
    result = relabel_consecutive(seg.clone())
    assert (result[0:3, 0:3] == 2).all()
    assert (result[5:8, 5:8] == 1).all()


def test_fast_relabel_basic():
    seg = np.zeros((5, 5), dtype=np.int32)
    seg[0:2, 0:2] = 1
    seg[3:5, 3:5] = 2
    labels = np.array([9, 4], dtype=np.int32)  # instance 1 -> 10, instance 2 -> 5
    fast_relabel(seg, labels)
    assert (seg[0:2, 0:2] == 10).all()
    assert (seg[3:5, 3:5] == 5).all()
    assert seg[2, 2] == 0  # background unchanged


def test_fast_relabel_identity():
    seg = np.zeros((5, 5), dtype=np.int32)
    seg[0:2, 0:2] = 1
    seg[3:5, 3:5] = 2
    labels = np.array([0, 1], dtype=np.int32)  # Identity mapping
    fast_relabel(seg, labels)
    assert (seg[0:2, 0:2] == 1).all()
    assert (seg[3:5, 3:5] == 2).all()


def test_compress_decompress_roundtrip():
    data = torch.randint(0, 100, (50, 50), dtype=torch.int32)
    flat = data.reshape(-1)
    compressed = compress(flat)
    decompressed = decompress(compressed, dtype=torch.int32).reshape(50, 50)
    assert torch.equal(data, decompressed)


def test_compress_output_dtype():
    data = torch.zeros(10, dtype=torch.int32)
    result = compress(data)
    assert result.dtype == torch.uint8


def test_decompress_restores_dtype():
    data = torch.randint(0, 10, (20,), dtype=torch.int64)
    compressed = compress(data)
    result = decompress(compressed, dtype=torch.int64)
    assert result.dtype == torch.int64


def test_compress_level():
    data = torch.arange(100, dtype=torch.int32)
    compressed = compress(data, level=1)
    decompressed = decompress(compressed, dtype=torch.int32)
    assert torch.equal(data, decompressed)


def test_draw_disk_2d_basic():
    segmentation = np.zeros((10, 10), dtype=np.int32)
    positions = np.array([[3.0, 3.0], [7.0, 7.0]])
    radii = np.ones((2, 2))
    labels = np.arange(2, dtype=np.int32)

    draw_disk_2d(segmentation, positions, radii, labels, overwrite=True)

    assert segmentation[3, 3] == 1
    assert segmentation[7, 7] == 2
    assert segmentation[5, 5] == 0


def test_draw_disk_2d_different_radius():
    segmentation = np.zeros((10, 10), dtype=np.int32)
    positions = np.array([[5.0, 5.0]])
    radii = np.array([[2.0, 4.0]])
    labels = np.arange(1, dtype=np.int32)

    draw_disk_2d(segmentation, positions, radii, labels, overwrite=True)

    assert segmentation[5, 5] == 1
    assert segmentation[2, 5] == 0
    assert segmentation[5, 2] == 1


def test_draw_disk_2d_overwrite():
    segmentation = np.ones((10, 10), dtype=np.int32) * 5
    positions = np.array([[3.0, 3.0], [7.0, 7.0]])
    radii = np.ones((2, 2))
    labels = np.arange(2, dtype=np.int32)

    draw_disk_2d(segmentation, positions, radii, labels, overwrite=True)

    assert segmentation[3, 3] == 1
    assert segmentation[7, 7] == 2
    assert segmentation[5, 5] == 5


def test_draw_disk_2d_do_not_overwrite():
    segmentation = np.ones((10, 10), dtype=np.int32) * 5
    positions = np.array([[3.0, 3.0], [7.0, 7.0]])
    radii = np.ones((2, 2))
    labels = np.arange(2, dtype=np.int32)

    draw_disk_2d(segmentation, positions, radii, labels, overwrite=False)

    assert segmentation[3, 3] == 5
    assert segmentation[7, 7] == 5
    assert segmentation[5, 5] == 5


def test_draw_disk_2d_handles_overlap():
    segmentation = np.zeros((10, 10), dtype=np.int32)
    positions = np.array([[4.0, 4.0], [7.0, 7.0]])
    radii = np.ones((2, 2)) * 5
    labels = np.arange(2, dtype=np.int32)

    draw_disk_2d(segmentation, positions, radii, labels, overwrite=True)

    # Center of disk is correct
    assert segmentation[4, 4] == 1
    assert segmentation[7, 7] == 2

    # At overlap
    assert segmentation[5, 5] == 1
    assert segmentation[6, 6] == 2

    # Still some background far away
    assert segmentation[0, 0] == 0


def test_draw_disk_3d_basic():
    segmentation = np.zeros((10, 10, 10), dtype=np.int32)
    positions = np.array([[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]])
    radii = np.ones((2, 3))
    labels = np.arange(2, dtype=np.int32)

    draw_disk_3d(segmentation, positions, radii, labels, overwrite=True)

    assert segmentation[3, 3, 3] == 1
    assert segmentation[7, 7, 7] == 2
    assert segmentation[5, 5, 5] == 0


def test_draw_disk_3d_different_radius():
    segmentation = np.zeros((10, 10, 10), dtype=np.int32)
    positions = np.array([[5.0, 5.0, 5.0]])
    radii = np.array([[0.5, 2.0, 4.0]])
    labels = np.arange(1, dtype=np.int32)

    draw_disk_3d(segmentation, positions, radii, labels, overwrite=True)

    assert segmentation[5, 5, 5] == 1
    assert segmentation[4, 5, 5] == 0
    assert segmentation[5, 4, 5] == 1
    assert segmentation[5, 2, 5] == 0
    assert segmentation[5, 5, 2] == 1


def test_draw_disk_3d_overwrite():
    segmentation = np.ones((10, 10, 10), dtype=np.int32) * 5
    positions = np.array([[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]])
    radii = np.ones((2, 3))
    labels = np.arange(2, dtype=np.int32)

    draw_disk_3d(segmentation, positions, radii, labels, overwrite=True)

    assert segmentation[3, 3, 3] == 1
    assert segmentation[7, 7, 7] == 2
    assert segmentation[5, 5, 5] == 5


def test_draw_disk_3d_do_not_overwrite():
    segmentation = np.ones((10, 10, 10), dtype=np.int32) * 5
    positions = np.array([[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]])
    radii = np.ones((2, 3))
    labels = np.arange(2, dtype=np.int32)

    draw_disk_3d(segmentation, positions, radii, labels, overwrite=False)

    assert segmentation[3, 3, 3] == 5
    assert segmentation[7, 7, 7] == 5
    assert segmentation[5, 5, 5] == 5


def test_draw_disk_3d_handles_overlap():
    segmentation = np.zeros((10, 10, 10), dtype=np.int32)
    positions = np.array([[4.0, 4.0, 4.0], [7.0, 7.0, 7.0]])
    radii = np.ones((2, 3)) * 5
    labels = np.arange(2, dtype=np.int32)

    draw_disk_3d(segmentation, positions, radii, labels, overwrite=True)

    # Center of disk is correct
    assert segmentation[4, 4, 4] == 1
    assert segmentation[7, 7, 7] == 2

    # At overlap
    assert segmentation[5, 5, 5] == 1
    assert segmentation[6, 6, 6] == 2

    # Still some background far away
    assert segmentation[0, 0, 0] == 0


## Then `as_detections`


def test_as_detections_passthrough_returns_same_object(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d)
    assert byotrack.as_detections(det) is det


def test_as_detections_float_n2_gives_point_detections(pos_2d: torch.Tensor):
    result = byotrack.as_detections(pos_2d)
    assert isinstance(result, byotrack.PointDetections)

    result = byotrack.as_detections(pos_2d.numpy())
    assert isinstance(result, byotrack.PointDetections)


def test_as_detections_float_n3_gives_point_detections(pos_3d: torch.Tensor):
    result = byotrack.as_detections(pos_3d)
    assert isinstance(result, byotrack.PointDetections)


def test_as_detections_int_n4_gives_bbox_detections(bbox_2d: torch.Tensor):
    result = byotrack.as_detections(bbox_2d)
    assert isinstance(result, byotrack.BBoxDetections)


def test_as_detections_int_n6_gives_bbox_detections(bbox_3d: torch.Tensor):
    result = byotrack.as_detections(bbox_3d)
    assert isinstance(result, byotrack.BBoxDetections)

    result = byotrack.as_detections(bbox_3d.numpy())
    assert isinstance(result, byotrack.BBoxDetections)


def test_as_detections_2d_int_gives_segmentation_detections(seg_2d: torch.Tensor):
    result = byotrack.as_detections(seg_2d)
    assert isinstance(result, byotrack.SegmentationDetections)

    result = byotrack.as_detections(seg_2d.numpy())
    assert isinstance(result, byotrack.SegmentationDetections)


def test_as_detections_3d_int_gives_segmentation_detections(seg_3d):
    result = byotrack.as_detections(seg_3d)
    assert isinstance(result, byotrack.SegmentationDetections)


def test_as_detections_list_are_converted_to_array():
    positions = [[1.0, 1.0], [2.0, 2.0], [5.0, 5.0]]
    result = byotrack.as_detections(positions)  # type: ignore[arg-type]
    assert isinstance(result, byotrack.PointDetections)


def test_as_detections_unrecognized_shape_raises():
    data = torch.ones(4, 5)  # (N, 5) float - not recognized
    with pytest.raises(ValueError, match="Cannot automatically determine Detections format"):
        byotrack.as_detections(data)


def test_as_detections_kwargs_forwarded():
    data = torch.tensor([[5.0, 5.0]])
    result = byotrack.as_detections(data, radius=3.0)
    assert isinstance(result, byotrack.PointDetections)
    assert result._radius.shape == data.shape
    assert (result._radius == 3.0).all()


## And finally the Detections class using PointDetections as a reference.


def test_detections_default_confidence_is_ones(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d)
    assert torch.allclose(det.confidence, torch.ones(len(pos_2d)))


def test_detections_default_labels_are_consecutive(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d)
    assert torch.equal(det.labels, torch.arange(len(pos_2d)))


def test_detections_explicit_confidence_stored(pos_2d: torch.Tensor) -> None:
    conf = torch.rand(len(pos_2d))
    det = byotrack.PointDetections(pos_2d, confidence=conf)
    assert torch.allclose(det.confidence, conf)


def test_detections_explicit_labels_stored(pos_2d: torch.Tensor) -> None:
    labels = torch.randint(0, 15, (len(pos_2d),))
    det = byotrack.PointDetections(pos_2d, labels=labels)
    assert torch.equal(det.labels, labels)


def test_detections_confidence_wrong_format_raises(pos_2d: torch.Tensor) -> None:
    conf = torch.ones(len(pos_2d), 3)  # Should be 1D
    with pytest.raises(ValueError, match="confidence is expected"):
        byotrack.PointDetections(pos_2d, confidence=conf)

    conf = torch.ones(len(pos_2d) + 1)  # Mismatch on length
    with pytest.raises(ValueError, match="confidence length"):
        byotrack.PointDetections(pos_2d, confidence=conf)

    conf = torch.ones(len(pos_2d)) * -1  # Negative values
    with pytest.raises(ValueError, match="non-negative"):
        byotrack.PointDetections(pos_2d, confidence=conf)


def test_detections_labels_wrong_format_raises(pos_2d: torch.Tensor) -> None:
    labels = torch.zeros(1, len(pos_2d), dtype=torch.int32)  # Should be 1D
    with pytest.raises(ValueError, match="labels is expected"):
        byotrack.PointDetections(pos_2d, labels=labels)

    labels = torch.ones(len(pos_2d) - 1, dtype=torch.int32)  # Mismatch on length
    with pytest.raises(ValueError, match="labels length"):
        byotrack.PointDetections(pos_2d, labels=labels)

    labels = torch.ones(len(pos_2d)) * -1  # Negative values
    with pytest.raises(ValueError, match="non-negative"):
        byotrack.PointDetections(pos_2d, labels=labels)


def test_detections_len_equals_length(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d)
    assert len(det) == det.length == len(pos_2d)


def test_detections_save_load_unknown_type_raises(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "det.pt"
    torch.save({"_type": "unknown_type"}, path)
    with pytest.raises(ValueError, match="Unknown Detections type"):
        byotrack.Detections.load(path)


def test_detections_save_load_multi_frames(tmp_path: pathlib.Path, pos_2d: torch.Tensor) -> None:
    d0 = byotrack.PointDetections(pos_2d)
    d1 = byotrack.PointDetections(pos_2d * 2)
    out_dir = tmp_path / "frames"
    byotrack.Detections.save_multi_frames_detections([d0, d1], out_dir)
    loaded = byotrack.Detections.load_multi_frames_detections(out_dir)
    assert len(loaded) == 2
    assert isinstance(loaded[0], byotrack.PointDetections)
    assert torch.allclose(loaded[0].position, d0.position)
    assert torch.allclose(loaded[1].position, d1.position)


def test_detections_save_load_multi_frames_fail_if_missing_file(tmp_path: pathlib.Path, pos_2d: torch.Tensor) -> None:
    d0 = byotrack.PointDetections(pos_2d)
    d1 = byotrack.PointDetections(pos_2d * 2)
    out_dir = tmp_path / "frames"
    byotrack.Detections.save_multi_frames_detections([d0, d1], out_dir)

    shutil.move(out_dir / "1.pt", out_dir / "2.pt")

    with pytest.raises(KeyError):
        byotrack.Detections.load_multi_frames_detections(out_dir)


def test_detections_no_cache_for_metadata(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d, cache=True, compress=False)

    assert len(det._cache) == 0

    # Test non-cached properties
    labels = det.labels
    confidence = det.confidence
    labeled_seg = det.labeled_segmentation

    # labeled_seg triggers segmentation
    assert set(det._cache) == {"segmentation"}

    assert det.labels is not labels
    assert det.confidence is not confidence

    # As we don't have labels, labeled_seg is simply the cached segmentation
    assert det.labeled_segmentation is labeled_seg


## relabel


def test_relabel_changes_labels(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d)
    new_labels = torch.tensor([5, 9], dtype=torch.int32)

    relabeled = det.relabel(new_labels)

    assert torch.equal(relabeled.labels, new_labels)
    assert torch.allclose(relabeled.position, det.position)  # Other data untouched


def test_relabel_original_unaffected(pos_2d: torch.Tensor) -> None:
    labels = torch.tensor([0, 1], dtype=torch.int32)
    det = byotrack.PointDetections(pos_2d, labels=labels)

    relabeled = det.relabel(torch.tensor([5, 9], dtype=torch.int32))

    assert relabeled is not det
    assert torch.equal(det.labels, labels)  # Original untouched


def test_relabel_metadata_isolated_between_clone_and_original(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d)
    relabeled = det.relabel(torch.tensor([5, 9], dtype=torch.int32))

    relabeled.metadata["features"] = torch.ones(2)

    assert "features" not in det.metadata


def test_relabel_cache_isolated_between_clone_and_original(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d, cache=True)
    det.segmentation  # noqa: B018  # Populate cache on the original

    relabeled = det.relabel(torch.tensor([5, 9], dtype=torch.int32))

    assert relabeled._cache is not det._cache
    assert "segmentation" in relabeled._cache  # Copied over, not lost


def test_relabel_wrong_length_raises(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d)
    with pytest.raises(ValueError, match="labels length"):
        det.relabel(torch.zeros(len(pos_2d) + 1, dtype=torch.int32))
