from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import byotrack
from byotrack.api.detections.detections import labels_of

if TYPE_CHECKING:
    import pathlib


## Construction


def test_point_detections_2d_construction() -> None:
    pos_2d = torch.tensor([[2.0, 4.0], [5.5, 2.0]])

    det = byotrack.PointDetections(pos_2d, radius=2.0)
    assert det.length == len(pos_2d)
    assert det.dim == len(det.shape) == 2
    assert det.position.shape == (len(pos_2d), 2)
    assert det.bbox.shape == (len(pos_2d), 4)
    assert det.segmentation.shape == det.shape
    assert det.labeled_segmentation.shape == det.shape
    assert det.mass.shape == (len(pos_2d),)
    assert det.labels.shape == (len(pos_2d),)
    assert det.confidence.shape == (len(pos_2d),)

    # Check inferred shape (ceil(max(pos + radius)) + 1)
    assert torch.allclose(det._radius, torch.full(pos_2d.shape, 2.0))
    assert det.shape == (9, 7)  # ceil(5.5 + 2.0) + 1, ceil(4.0 + 2.0) + 1


def test_point_detections_3d_construction(pos_3d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_3d)
    assert det.length == len(pos_3d)
    assert det.dim == len(det.shape) == 3
    assert det.position.shape == (len(pos_3d), 3)
    assert det.bbox.shape == (len(pos_3d), 6)
    assert det.segmentation.shape == det.shape
    assert det.labeled_segmentation.shape == det.shape
    assert det.mass.shape == (len(pos_3d),)
    assert det.labels.shape == (len(pos_3d),)
    assert det.confidence.shape == (len(pos_3d),)


def test_point_detections_wrong_format_raises():
    pos = torch.tensor([[float("nan"), 2.0]])  # NaN not allowed
    with pytest.raises(ValueError, match="Found ill-defined NaN position"):
        byotrack.PointDetections(pos)

    pos = torch.ones(3)  # Should be 2D
    with pytest.raises(ValueError, match="expected to be of shape"):
        byotrack.PointDetections(pos)

    pos = torch.ones(3, 4)  # dim should be 2 or 3
    with pytest.raises(ValueError, match="should have 2 or 3 values"):
        byotrack.PointDetections(pos)


def test_point_detections_negative_radius_raises(pos_2d: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="Radius should be non-negative"):
        byotrack.PointDetections(pos_2d, radius=-1.0)

    # However, without empty data, it does not raise:
    byotrack.PointDetections(torch.zeros(0, 3), radius=-1.0)


def test_point_detections_empty_2d():
    pos = torch.zeros(0, 2)
    det = byotrack.PointDetections(pos)
    assert det.length == 0
    assert det.shape == (1, 1)
    assert det.position.shape == (0, 2)
    assert det.bbox.shape == (0, 4)
    assert det.segmentation.shape == (1, 1)
    assert det.labeled_segmentation.shape == (1, 1)
    assert det.mass.shape == (0,)
    assert det.labels.shape == (0,)
    assert det.confidence.shape == (0,)


def test_point_detections_empty_3d():
    pos = torch.zeros(0, 3)
    det = byotrack.PointDetections(pos)
    assert det.length == 0
    assert det.shape == (1, 1, 1)
    assert det.position.shape == (0, 3)
    assert det.bbox.shape == (0, 6)
    assert det.segmentation.shape == (1, 1, 1)
    assert det.labeled_segmentation.shape == (1, 1, 1)
    assert det.mass.shape == (0,)
    assert det.labels.shape == (0,)
    assert det.confidence.shape == (0,)


def test_point_detections_uniform_radius_scalar(pos_3d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_3d, radius=3.0)
    assert det._radius.shape == pos_3d.shape
    assert (det._radius == 3.0).all()


def test_point_detections_per_detection_radius(pos_2d: torch.Tensor) -> None:
    radius = torch.rand(pos_2d.shape) * 5
    det = byotrack.PointDetections(pos_2d, radius=radius)
    assert torch.allclose(det._radius, radius)


def test_point_detections_asymmetric_radius() -> None:
    radius = torch.tensor([[2.0, 5.0]])
    pos = torch.tensor([[5.0, 2.0]])
    det = byotrack.PointDetections(pos, radius=radius)
    assert det._radius[0, 0] == 2.0
    assert det._radius[0, 1] == 5.0
    assert det.shape == (8, 8)


def test_point_detections_explicit_shape(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d, shape=(100, 100))
    assert det.shape == (100, 100)

    det = byotrack.PointDetections(pos_2d, radius=20.0, shape=(10, 10))
    assert det.shape == (10, 10)


def test_point_detections_shape_wrong_dim_raises(pos_2d: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="shape has 3 dimensions but"):
        byotrack.PointDetections(pos_2d, shape=(100, 100, 100))  # 3D shape for 2D positions


## Properties


def test_point_detections_position_identity_2d(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d)
    assert torch.allclose(det.position, pos_2d)


def test_point_detections_position_identity_3d(pos_3d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_3d)
    assert torch.allclose(det.position, pos_3d)


def test_point_detections_bbox_shape_2d(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d)
    assert det.bbox.shape == (len(pos_2d), 4)
    assert det.bbox.dtype == torch.int32


def test_point_detections_bbox_shape_3d(pos_3d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_3d)
    assert det.bbox.shape == (len(pos_3d), 6)
    assert det.bbox.dtype == torch.int32


def test_point_detections_bbox_contains_position(pos_2d: torch.Tensor):
    det = byotrack.PointDetections(pos_2d, radius=2.0)
    start = det.bbox[:, :2]
    end = start + det.bbox[:, 2:]

    assert (start <= pos_2d).all()
    assert (pos_2d <= end).all()


def test_point_detections_bbox_small_radius(pos_2d: torch.Tensor):
    det = byotrack.PointDetections(pos_2d, radius=0.01)
    assert det.bbox[:, det.dim :].min() >= 1  # Bbox size is at least 1


def test_point_detections_segmentation_shape_2d(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d)
    seg = det.segmentation
    assert seg.ndim == 2
    assert seg.shape == det.shape
    assert seg.dtype == torch.int32


def test_point_detections_segmentation_shape_3d(pos_3d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_3d)
    seg = det.segmentation
    assert seg.ndim == 3
    assert seg.shape == det.shape
    assert seg.dtype == torch.int32


def test_point_detections_segmentation_contains_labels() -> None:
    # True only if labels do not fully overlap or at not outside the shape
    pos = torch.tensor([[2.0, 5.0], [5.0, 2.0]])
    det = byotrack.PointDetections(pos)
    seg = det.segmentation
    assert len(labels_of(seg)) == len(pos)
    assert seg[2, 5] == 1
    assert seg[5, 2] == 2


def test_point_detections_mass_2d_formula():
    pos = torch.tensor([[5.0, 5.0]])
    det = byotrack.PointDetections(pos, radius=3.0)
    expected = round(3.14159 * 3.0 * 3.0)
    assert det.mass[0].item() == expected


def test_point_detections_mass_3d_formula():
    pos = torch.tensor([[5.0, 5.0, 5.0]])
    det = byotrack.PointDetections(pos, radius=2.0)
    expected = round((4 / 3) * 3.14159 * 2.0 * 2.0 * 2.0)
    assert det.mass[0].item() == expected


def test_point_detections_mass_min_one():
    pos = torch.tensor([[5.0, 5.0]])
    det = byotrack.PointDetections(pos, radius=0.01)  # tiny radius
    assert det.mass[0].item() >= 1


## Filtering


def test_point_detections_filter_2d(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d)
    kept = torch.full((len(pos_2d),), fill_value=False)
    kept[1] = True

    filtered = det.filter(kept)

    assert filtered.length == 1
    assert torch.allclose(filtered.position[0], pos_2d[1])


def test_point_detections_filter_3d(pos_3d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_3d)
    kept = torch.full((len(pos_3d),), fill_value=False)
    kept[0] = True
    kept[2] = True

    filtered = det.filter(kept)
    assert filtered.length == 2
    assert torch.allclose(filtered.position[0], pos_3d[0])
    assert torch.allclose(filtered.position[1], pos_3d[2])


def test_point_detections_filter_with_metadata(pos_3d: torch.Tensor):
    confidence = torch.rand(len(pos_3d))
    labels = torch.randint(0, 15, (len(pos_3d),), dtype=torch.int32)
    radius = torch.rand(pos_3d.shape) * 5
    det = byotrack.PointDetections(pos_3d, radius=radius, confidence=confidence, labels=labels)

    kept = torch.full((len(pos_3d),), fill_value=False)
    kept[2] = True  # Let's keep a single value

    filtered = det.filter(kept)
    assert len(filtered) == 1
    assert torch.allclose(filtered.position, pos_3d[2:3])
    assert torch.allclose(filtered.confidence, confidence[2:3])
    assert torch.allclose(filtered.labels, labels[2:3])
    assert torch.allclose(filtered._radius, radius[2:3])

    assert det.shape == filtered.shape  # Shape is preserved


def test_point_detections_filter_all_false_empty(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d)
    kept = torch.zeros(len(pos_2d), dtype=torch.bool)
    filtered = det.filter(kept)
    assert filtered.length == 0


## Save and Load


def test_point_detections_save_load_2d(tmp_path: pathlib.Path, pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d)
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)

    assert isinstance(loaded, byotrack.PointDetections)
    assert torch.allclose(loaded.position, pos_2d)


def test_point_detections_save_load_3d(tmp_path: pathlib.Path, pos_3d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_3d, compress=False)
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)

    assert isinstance(loaded, byotrack.PointDetections)
    assert torch.allclose(loaded.position, pos_3d)


def test_point_detections_save_load_with_metadata(tmp_path: pathlib.Path, pos_2d: torch.Tensor) -> None:
    confidence = torch.rand(len(pos_2d))
    labels = torch.randint(0, 15, (len(pos_2d),), dtype=torch.int32)
    radius = torch.rand(pos_2d.shape) * 5

    det = byotrack.PointDetections(pos_2d, radius=radius, confidence=confidence, labels=labels)
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)

    assert isinstance(loaded, byotrack.PointDetections)
    assert torch.allclose(loaded.confidence, confidence)
    assert torch.allclose(loaded.labels, labels)
    assert torch.allclose(loaded._radius, radius)
    assert det.shape == loaded.shape


def test_point_detections_save_load_explicit_shape(tmp_path: pathlib.Path, pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d, shape=(100, 100))
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)
    assert loaded.shape == (100, 100)


def test_point_detections_save_load_compress_and_cache(tmp_path: pathlib.Path, pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d, compress=not byotrack.ZSTD_SEG, cache=False)
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)
    assert loaded._use_cache != det._use_cache
    assert loaded._compress != det._compress

    loaded = byotrack.Detections.load(path, compress=not byotrack.ZSTD_SEG, cache=False)

    assert loaded._use_cache == det._use_cache
    assert loaded._compress == det._compress


## Cache & Compress


def test_point_detections_without_cache(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d, cache=False)

    assert len(det._cache) == 0

    pos = det.position
    bbox = det.bbox
    seg = det.segmentation
    mass = det.mass

    assert len(det._cache) == 0

    assert det.position is pos
    assert det.bbox is not bbox  # Always cached in det._position
    assert det.segmentation is not seg
    assert det.mass is not mass


def test_point_detections_cache_without_compress(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d, cache=True, compress=False)

    assert len(det._cache) == 0

    pos = det.position
    bbox = det.bbox
    seg = det.segmentation
    mass = det.mass

    assert set(det._cache) == {"bbox", "segmentation", "mass"}

    assert det._cache["segmentation"] is seg  # Uncompressed cache

    assert det.position is pos
    assert det.bbox is bbox
    assert det.segmentation is seg
    assert det.mass is mass


def test_point_detections_cache_with_compress(pos_2d: torch.Tensor) -> None:
    det = byotrack.PointDetections(pos_2d, cache=True, compress=True)

    assert len(det._cache) == 0

    pos = det.position
    bbox = det.bbox
    seg = det.segmentation
    mass = det.mass

    assert set(det._cache) == {"bbox", "segmentation", "mass"}

    assert det._cache["segmentation"].dtype == torch.uint8  # Cache is compressed for seg

    assert det.position is pos
    assert det.bbox is bbox
    assert det.segmentation is not seg  # Decompressed
    assert det.mass is mass
