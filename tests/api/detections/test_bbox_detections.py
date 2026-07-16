from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import byotrack
from byotrack.api.detections.detections import labels_of

if TYPE_CHECKING:
    import pathlib


## Construction


def test_bbox_detections_2d_construction() -> None:
    bbox_2d = torch.tensor([[1, 2, 3, 3], [5, 1, 2, 2]])
    det = byotrack.BBoxDetections(bbox_2d)
    assert det.length == len(bbox_2d)
    assert det.dim == len(det.shape) == 2
    assert det.position.shape == (len(bbox_2d), 2)
    assert det.bbox.shape == (len(bbox_2d), 4)
    assert det.segmentation.shape == det.shape
    assert det.labeled_segmentation.shape == det.shape
    assert det.mass.shape == (len(bbox_2d),)
    assert det.labels.shape == (len(bbox_2d),)
    assert det.confidence.shape == (len(bbox_2d),)
    assert det.shape == (7, 5)  # Check inferred shape


def test_bbox_detections_3d_construction(bbox_3d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_3d)
    assert det.length == len(bbox_3d)
    assert det.dim == len(det.shape) == 3
    assert det.position.shape == (len(bbox_3d), 3)
    assert det.bbox.shape == (len(bbox_3d), 6)
    assert det.segmentation.shape == det.shape
    assert det.labeled_segmentation.shape == det.shape
    assert det.mass.shape == (len(bbox_3d),)
    assert det.labels.shape == (len(bbox_3d),)
    assert det.confidence.shape == (len(bbox_3d),)


def test_bbox_detections_wrong_format_raises():
    bbox = torch.tensor([[0, 0, 0, 5]], dtype=torch.int32)  # Zero height should raise
    with pytest.raises(ValueError, match="bbox should only have positive sizes"):
        byotrack.BBoxDetections(bbox)

    bbox = torch.ones(4)  # Should be 2D
    with pytest.raises(ValueError, match="expected to be of shape"):
        byotrack.BBoxDetections(bbox)

    bbox = torch.ones(3, 3)  # dim should be 4 or 6
    with pytest.raises(ValueError, match="should have 4 or 6 values"):
        byotrack.BBoxDetections(bbox)


def test_bbox_detections_empty_2d():
    bbox = torch.zeros(0, 4)
    det = byotrack.BBoxDetections(bbox)
    assert det.length == 0
    assert det.shape == (1, 1)
    assert det.position.shape == (0, 2)
    assert det.bbox.shape == (0, 4)
    assert det.segmentation.shape == (1, 1)
    assert det.labeled_segmentation.shape == (1, 1)
    assert det.mass.shape == (0,)
    assert det.labels.shape == (0,)
    assert det.confidence.shape == (0,)


def test_bbox_detections_empty_3d():
    bbox = torch.zeros(0, 6)
    det = byotrack.BBoxDetections(bbox)
    assert det.length == 0
    assert det.shape == (1, 1, 1)
    assert det.position.shape == (0, 3)
    assert det.bbox.shape == (0, 6)
    assert det.segmentation.shape == (1, 1, 1)
    assert det.labeled_segmentation.shape == (1, 1, 1)
    assert det.mass.shape == (0,)
    assert det.labels.shape == (0,)
    assert det.confidence.shape == (0,)


def test_bbox_detections_explicit_shape(bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d, shape=(100, 100))
    assert det.shape == (100, 100)


def test_bbox_detections_shape_wrong_dim_raises(bbox_2d: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="shape has 3 dimensions but"):
        byotrack.BBoxDetections(bbox_2d, shape=(100, 100, 100))  # 3D shape for 2D bbox


## Properties


def test_bbox_detections_position_shape_2d(bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d)
    pos = det.position
    assert pos.shape == (len(bbox_2d), 2)
    assert pos.dtype == torch.float32


def test_bbox_detections_position_shape_3d(bbox_3d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_3d)
    pos = det.position
    assert pos.shape == (len(bbox_3d), 3)
    assert pos.dtype == torch.float32


def test_bbox_detections_position_inside_bbox(bbox_3d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_3d)
    pos = det.position

    start = det.bbox[:, :3]
    end = start + det.bbox[:, 3:]

    assert (start <= pos).all()
    assert (pos <= end).all()

    # It should even be the middle
    assert torch.allclose(pos, (start + end - 1) / 2)


def test_bbox_detections_bbox_identity_2d(bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d)
    assert torch.allclose(det.bbox, bbox_2d)


def test_bbox_detections_bbox_identity_3d(bbox_3d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_3d)
    assert torch.allclose(det.bbox, bbox_3d)


def test_bbox_detections_segmentation_shape_2d(bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d)
    seg = det.segmentation
    assert seg.ndim == 2
    assert seg.shape == det.shape
    assert seg.dtype == torch.int32


def test_bbox_detections_segmentation_shape_3d(bbox_3d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_3d)
    seg = det.segmentation
    assert seg.ndim == 3
    assert seg.shape == det.shape
    assert seg.dtype == torch.int32


def test_bbox_detections_segmentation_fills_region() -> None:
    bbox_2d = torch.tensor([[2.0, 5.0, 2.0, 2.0], [5.0, 2.0, 2.0, 2.0]])
    det = byotrack.BBoxDetections(bbox_2d)
    seg = det.segmentation

    assert len(labels_of(seg)) == len(bbox_2d)

    for i, bbox in enumerate(det.bbox):
        assert (seg[bbox[0] : bbox[0] + bbox[2], bbox[1] : bbox[1] + bbox[3]] == i + 1).all()


def test_bbox_detections_mass_product_of_sizes_2d(bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d)
    assert torch.allclose(det.mass, det.bbox[:, det.dim :].prod(dim=1))


def test_bbox_detections_mass_product_of_sizes_3d(bbox_3d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_3d)
    assert torch.allclose(det.mass, det.bbox[:, det.dim :].prod(dim=1))


## Filtering


def test_bbox_detections_filter(bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d)
    kept = torch.full((len(bbox_2d),), fill_value=False)
    kept[1] = True

    filtered = det.filter(kept)

    assert isinstance(filtered, byotrack.BBoxDetections)
    assert filtered.length == 1
    assert torch.allclose(filtered.bbox[0], bbox_2d[1])


def test_bbox_detections_filter_with_metadata(bbox_2d: torch.Tensor):
    confidence = torch.rand(len(bbox_2d))
    labels = torch.randint(0, 15, (len(bbox_2d),), dtype=torch.int32)
    det = byotrack.BBoxDetections(bbox_2d, confidence=confidence, labels=labels)

    kept = torch.full((len(bbox_2d),), fill_value=False)
    kept[0] = True  # Let's keep a single value

    filtered = det.filter(kept)
    assert len(filtered) == 1
    assert torch.allclose(filtered.bbox, bbox_2d[2:3])
    assert torch.allclose(filtered.confidence, confidence[2:3])
    assert torch.allclose(filtered.labels, labels[2:3])

    assert det.shape == filtered.shape  # Shape is preserved


def test_bbox_detections_filter_all_false_empty(bbox_3d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_3d)
    kept = torch.zeros(len(bbox_3d), dtype=torch.bool)

    filtered = det.filter(kept)

    assert filtered.length == 0


## add_disks


def test_bbox_detections_add_disks_basic(bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d)
    new_positions = torch.tensor([[20.0, 20.0]])

    updated = det.add_disks(new_positions, labels=torch.tensor([7]), confidence=torch.tensor([0.5]))

    assert updated.length == det.length + 1
    assert updated.labels.tolist()[-1] == 7
    assert updated.confidence.tolist()[-1] == 0.5
    assert torch.allclose(updated.position[-1], new_positions[0])


def test_bbox_detections_add_disks_basic_3d(bbox_3d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_3d)
    new_positions = torch.tensor([[20.0, 20.0, 10.0], [5.0, 5.0, 20.0]])

    updated = det.add_disks(new_positions, labels=torch.tensor([3, 4]))

    assert updated.length == det.length + 2
    assert updated.labels.tolist()[-2:] == [3, 4]
    assert torch.allclose(updated.position[-2:], new_positions)


def test_bbox_detections_add_disks_default_labels_and_confidence(bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d)
    updated = det.add_disks(torch.tensor([[20.0, 20.0]]), 2.0)

    assert updated.labels.tolist()[-1] == det.length
    assert updated.confidence.tolist()[-1] == 1.0

    det = byotrack.BBoxDetections(bbox_2d, labels=torch.tensor([2, 7]))
    updated = det.add_disks(torch.tensor([[20.0, 20.0]]), 2.0)

    assert updated.labels.tolist()[-1] == 8


def test_bbox_detections_add_disks_sizes_are_positive(bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d)
    # Even a zero radius must yield a strictly positive box size (BBoxDetections forbids zero-size boxes)
    updated = det.add_disks(torch.tensor([[20.0, 20.0]]), 0.0)

    assert (updated.bbox[:, det.dim :] > 0).all()


def test_bbox_detections_add_disks_original_unaffected(bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d)
    original_length = det.length

    det.add_disks(torch.tensor([[20.0, 20.0]]), 2.0)

    assert det.length == original_length


## Save & Load


def test_bbox_detections_save_load_2d(tmp_path: pathlib.Path, bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d)
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)

    assert isinstance(loaded, byotrack.BBoxDetections)
    assert torch.allclose(loaded.bbox, bbox_2d)


def test_bbox_detections_save_load_3d(tmp_path: pathlib.Path, bbox_3d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_3d)
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)

    assert isinstance(loaded, byotrack.BBoxDetections)
    assert torch.allclose(loaded.bbox, bbox_3d)


def test_bbox_detections_save_load_with_metadata(tmp_path: pathlib.Path, bbox_2d: torch.Tensor) -> None:
    confidence = torch.rand(len(bbox_2d))
    labels = torch.randint(0, 15, (len(bbox_2d),), dtype=torch.int32)

    det = byotrack.BBoxDetections(bbox_2d, confidence=confidence, labels=labels)
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)

    assert isinstance(loaded, byotrack.BBoxDetections)
    assert torch.allclose(loaded.confidence, confidence)
    assert torch.allclose(loaded.labels, labels)
    assert det.shape == loaded.shape


def test_bbox_detections_save_load_explicit_shape(tmp_path: pathlib.Path, bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d, shape=(100, 100))
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)
    assert loaded.shape == (100, 100)


def test_bbox_detections_save_load_compress_and_cache(tmp_path: pathlib.Path, bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d, compress=not byotrack.ZSTD_SEG, cache=False)
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)
    assert loaded._use_cache != det._use_cache
    assert loaded._compress != det._compress

    loaded = byotrack.Detections.load(path, compress=not byotrack.ZSTD_SEG, cache=False)

    assert loaded._use_cache == det._use_cache
    assert loaded._compress == det._compress


## Cache & Compress


def test_bbox_detections_without_cache(bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d, cache=False)

    assert len(det._cache) == 0

    pos = det.position
    bbox = det.bbox
    seg = det.segmentation
    mass = det.mass

    assert len(det._cache) == 0

    assert det.position is not pos
    assert det.bbox is bbox  # Always cached in det._bbox
    assert det.segmentation is not seg
    assert det.mass is not mass


def test_bbox_detections_cache_without_compress(bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d, cache=True, compress=False)

    assert len(det._cache) == 0

    pos = det.position
    bbox = det.bbox
    seg = det.segmentation
    mass = det.mass

    assert set(det._cache) == {"position", "segmentation", "mass"}

    assert det._cache["segmentation"] is seg  # Uncompressed cache

    assert det.position is pos
    assert det.bbox is bbox
    assert det.segmentation is seg
    assert det.mass is mass


def test_bbox_detections_cache_with_compress(bbox_2d: torch.Tensor) -> None:
    det = byotrack.BBoxDetections(bbox_2d, cache=True, compress=True)

    assert len(det._cache) == 0

    pos = det.position
    bbox = det.bbox
    seg = det.segmentation
    mass = det.mass

    assert set(det._cache) == {"position", "segmentation", "mass"}

    assert det._cache["segmentation"].dtype == torch.uint8  # Cache is compressed for seg

    assert det.position is pos
    assert det.bbox is bbox
    assert det.segmentation is not seg  # Decompressed
    assert det.mass is mass
