from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

import byotrack
from byotrack.api.detections.detections import fast_relabel, labels_of
from byotrack.api.detections.segmentation_detections import _median_from_segmentation

if TYPE_CHECKING:
    import pathlib


def test_seg_detections_2d_construction(seg_2d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_2d)
    assert det.length == 3
    assert det.dim == 2
    assert det.shape == seg_2d.shape
    assert det.position.shape == (det.length, 2)
    assert det.bbox.shape == (det.length, 4)
    assert det.segmentation.shape == det.shape
    assert det.labeled_segmentation.shape == det.shape
    assert det.mass.shape == (det.length,)
    assert det.labels.shape == (det.length,)
    assert det.confidence.shape == (det.length,)


def test_seg_detections_3d_construction(seg_3d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_3d)
    assert det.length == 2
    assert det.dim == 3
    assert det.shape == seg_3d.shape
    assert det.position.shape == (det.length, 3)
    assert det.bbox.shape == (det.length, 6)
    assert det.segmentation.shape == det.shape
    assert det.labeled_segmentation.shape == det.shape
    assert det.mass.shape == (det.length,)
    assert det.labels.shape == (det.length,)
    assert det.confidence.shape == (det.length,)


def test_seg_detections_wrong_format_raises():
    seg = torch.zeros(20, 20, dtype=torch.int32)
    seg[3:6, 3:6] = -5  # Labels should be positive
    with pytest.raises(ValueError, match="segmentation should be non-negative"):
        byotrack.SegmentationDetections(seg)

    seg = torch.ones(4, dtype=torch.int32)  # Should be 2D or 3D
    with pytest.raises(ValueError, match="expected to be 2D or 3D"):
        byotrack.SegmentationDetections(seg)

    seg = torch.ones(0, 5, 0, dtype=torch.int32)  # Should at least have one pixel.
    with pytest.raises(ValueError, match="at least one pixel"):
        byotrack.SegmentationDetections(seg)


def test_seg_detections_wrong_position_fn_raises():
    seg = torch.zeros(20, 20, dtype=torch.int32)
    with pytest.raises(ValueError, match="Unknown position_method 'medion'"):
        byotrack.SegmentationDetections(seg, position_method="medion")


def test_seg_detections_all_background_2d():
    seg = torch.zeros(10, 10, dtype=torch.int32)
    det = byotrack.SegmentationDetections(seg)
    assert det.length == 0
    assert det.shape == seg.shape
    assert det.position.shape == (0, 2)
    assert det.bbox.shape == (0, 4)
    assert det.segmentation.shape == seg.shape
    assert det.labeled_segmentation.shape == seg.shape
    assert det.mass.shape == (0,)
    assert det.labels.shape == (0,)
    assert det.confidence.shape == (0,)


def test_seg_detections_all_background_3d():
    seg = torch.zeros(5, 10, 10, dtype=torch.int32)
    det = byotrack.SegmentationDetections(seg)
    assert det.length == 0
    assert det.shape == seg.shape
    assert det.position.shape == (0, 3)
    assert det.bbox.shape == (0, 6)
    assert det.segmentation.shape == seg.shape
    assert det.labeled_segmentation.shape == seg.shape
    assert det.mass.shape == (0,)
    assert det.labels.shape == (0,)
    assert det.confidence.shape == (0,)


def test_seg_detections_non_consecutive_labels():
    seg = torch.zeros(10, 10, dtype=torch.int32)
    seg[0:3, 0:3] = 1
    seg[5:8, 5:8] = 5  # non-consecutive (gap at 2,3,4)
    det = byotrack.SegmentationDetections(seg.clone())
    assert det.length == 2

    # Internally relabeled to consecutive
    unique = det.segmentation.unique().tolist()
    assert sorted(unique) == [0, 1, 2]

    # Original labels should be stored (0-indexed: 0 and 4)
    assert sorted(det.labels.tolist()) == [0, 4]

    # And the labeled_segmentation should match the original seg
    assert torch.allclose(det.labeled_segmentation, seg)


## Properties


def test_seg_detections_position_when_round() -> None:
    seg = torch.zeros(5, 10, 10)
    seg[3, 5, 5] = 7
    seg[1, 2, 2] = 2
    seg[2, 8, 2] = 4
    det = byotrack.SegmentationDetections(seg)
    pos = det.position

    assert det.position.shape == (3, 3)
    assert det.position.dtype == torch.float32
    assert torch.allclose(pos, torch.tensor([[1.0, 2, 2], [2, 8, 2], [3, 5, 5]]))


def test_seg_detections_position_method_mean_vs_median() -> None:
    seg = torch.tensor(  # Typical example where the median will be better
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    det = byotrack.SegmentationDetections(seg, position_method="mean")
    pos = det.position
    assert pos.shape == (1, 2)
    assert det.position[0, 0] == 3
    assert det.position[0, 1] > 1.5

    det = byotrack.SegmentationDetections(seg, position_method="median")
    pos = det.position
    assert pos.shape == (1, 2)
    assert det.position[0, 0] == 3
    assert det.position[0, 1] == 1


def test_seg_detections_position_method_custom_callable(seg_2d: torch.Tensor) -> None:
    def custom_fn(seg: np.ndarray) -> np.ndarray:
        n = seg.max()
        return np.zeros((n, len(seg.shape)), dtype=np.float32)

    det = byotrack.SegmentationDetections(seg_2d, position_method=custom_fn)
    pos = det.position
    assert pos.shape == (3, 2)
    assert pos.dtype == torch.float32
    assert (pos == 0.0).all()


def test_seg_detections_bbox_2d(seg_2d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_2d)
    bbox = det.bbox
    assert bbox.shape == (3, 4)
    assert bbox.dtype == torch.int32


def test_seg_detections_bbox_3d(seg_3d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_3d)
    bbox = det.bbox
    assert bbox.shape == (2, 6)
    assert bbox.dtype == torch.int32


def test_seg_detections_bbox_contains_all_occurrences(seg_2d) -> None:
    det = byotrack.SegmentationDetections(seg_2d)

    # Let's erase label 1
    bbox = det.bbox[1]
    seg = det.segmentation.clone()
    seg[bbox[0] : bbox[0] + bbox[2], bbox[1] : bbox[1] + bbox[3]] = 0
    labels = labels_of(seg)  # Labels after erasing

    assert 1 not in labels  # Has been erased

    # Check that bbox is minimal
    seg = det.segmentation
    assert 1 in labels_of(seg[bbox[0] : bbox[0] + 1])
    assert 1 in labels_of(seg[bbox[1] : bbox[1] + 1])
    assert 1 in labels_of(seg[bbox[0] + bbox[2] - 1 : bbox[0] + bbox[2]])
    assert 1 in labels_of(seg[bbox[1] + bbox[3] - 1 : bbox[1] + bbox[3]])


def test_seg_detections_segmentation_identity_2d(seg_2d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_2d)
    assert torch.allclose(det.segmentation, seg_2d)


def test_seg_detections_segmentation_identity_3d(seg_3d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_3d)
    assert torch.allclose(det.segmentation, seg_3d)


def test_seg_detections_mass_2d() -> None:
    seg = torch.tensor(
        [
            [5, 0, 0, 3, 3, 0],
            [0, 0, 0, 0, 3, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 2, 2, 2, 2],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 4, 4, 0],
        ]
    )
    det = byotrack.SegmentationDetections(seg)
    mass = det.mass
    assert mass.shape == (5,)
    assert torch.allclose(mass, torch.tensor([6, 4, 3, 2, 1], dtype=torch.int32))


def test_seg_detections_mass_3d() -> None:
    seg = torch.zeros(10, 10, 10, dtype=torch.int32)

    # A cross
    seg[3:7, 2, 2] = 5  # 4px
    seg[5, :5, 2] = 5  # 4px (+1)
    seg[5, 2, :5] = 5  # 4px (+1)

    # A point
    seg[1, 1, 1] = 9  # 1px

    # A rectangle
    seg[6:9, 6:9, 6:9] = 13  # 27px

    det = byotrack.SegmentationDetections(seg)
    mass = det.mass
    assert mass.shape == (3,)

    assert torch.allclose(mass, torch.tensor([12, 1, 27], dtype=torch.int32))


## Filtering


def test_seg_detections_filter_2d(seg_2d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_2d)
    kept = torch.tensor([True, False, True])

    filtered = det.filter(kept)

    assert isinstance(filtered, byotrack.SegmentationDetections)
    assert filtered.length == 2
    assert torch.allclose(det.position[kept], filtered.position)
    assert (filtered.segmentation[seg_2d == 2] == 0).all()


def test_seg_detections_filter_3d(seg_3d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_3d)
    kept = torch.tensor([True, False])

    filtered = det.filter(kept)

    assert filtered.length == 1
    assert torch.allclose(det.position[kept], filtered.position)
    assert (filtered.segmentation[seg_3d == 2] == 0).all()


def test_seg_detections_filter_with_metadata(seg_2d: torch.Tensor) -> None:
    confidence = torch.rand(3)
    labels = torch.tensor([5, 7, 13], dtype=torch.int32)

    fast_relabel(seg_2d.numpy(), labels.numpy())

    det = byotrack.SegmentationDetections(seg_2d, confidence=confidence)
    kept = torch.tensor([True, False, True])

    filtered = det.filter(kept)

    # Sanity check
    assert det._labels is not None
    assert (det._labels == labels).all()

    assert len(filtered) == 2
    assert torch.allclose(filtered.confidence, confidence[[0, 2]])
    assert torch.allclose(filtered.labels, labels[[0, 2]])

    assert torch.allclose(labels_of(filtered.labeled_segmentation), labels[[0, 2]])


def test_seg_detections_filter_all_false_empty(seg_3d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_3d)
    kept = torch.tensor([False, False])

    filtered = det.filter(kept)

    assert filtered.length == 0
    assert (filtered.segmentation == 0).all()


def test_seg_detections_filter_with_compress(seg_2d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_2d, compress=True)
    kept = torch.tensor([True, False, True])

    filtered = det.filter(kept)

    assert isinstance(filtered, byotrack.SegmentationDetections)
    assert filtered.length == 2
    assert torch.allclose(det.position[kept], filtered.position)
    assert (filtered.segmentation[seg_2d == 2] == 0).all()


def test_seg_detections_filter_without_labels(seg_2d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_2d, compress=False)
    det._labels = None
    kept = torch.tensor([True, False, True])

    filtered = det.filter(kept)

    assert isinstance(filtered, byotrack.SegmentationDetections)
    assert filtered.length == 2
    assert torch.allclose(det.position[kept], filtered.position)
    assert (filtered.segmentation[seg_2d == 2] == 0).all()


## Save & Load


def test_seg_detections_save_load_2d(tmp_path: pathlib.Path, seg_2d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_2d)
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)

    assert isinstance(loaded, byotrack.SegmentationDetections)
    assert (loaded.segmentation == det.segmentation).all()


def test_seg_detections_save_load_3d(tmp_path: pathlib.Path, seg_3d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_3d)
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)

    assert isinstance(loaded, byotrack.SegmentationDetections)
    assert (loaded.segmentation == det.segmentation).all()


def test_seg_detections_save_load_with_metadata(tmp_path: pathlib.Path, seg_2d: torch.Tensor) -> None:
    confidence = torch.rand(3)
    labels = torch.tensor([5, 7, 13], dtype=torch.int32)

    fast_relabel(seg_2d.numpy(), labels.numpy())

    det = byotrack.SegmentationDetections(seg_2d, confidence=confidence)
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)

    assert isinstance(loaded, byotrack.SegmentationDetections)
    assert torch.allclose(loaded.labeled_segmentation, det.labeled_segmentation)
    assert torch.allclose(loaded.confidence, confidence)
    assert torch.allclose(loaded.labels, labels)
    assert det.shape == loaded.shape


def test_seg_detections_save_load_position_method_mean(tmp_path: pathlib.Path, seg_2d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_2d, position_method="mean")
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)

    assert isinstance(loaded, byotrack.SegmentationDetections)
    assert torch.allclose(loaded.position, det.position)
    assert loaded._position_fn == det._position_fn


def test_seg_detections_save_load_position_method_specific(tmp_path: pathlib.Path, seg_2d: torch.Tensor) -> None:
    def custom_fn(seg: np.ndarray) -> np.ndarray:
        n = seg.max()
        return np.zeros((n, len(seg.shape)), dtype=np.float32)

    det = byotrack.SegmentationDetections(seg_2d, position_method=custom_fn)
    path = tmp_path / "det.pt"

    with pytest.warns(UserWarning, match="Specific ``position_fn`` cannot be saved"):
        det.save(path)

    loaded = byotrack.Detections.load(path)

    assert isinstance(loaded, byotrack.SegmentationDetections)
    assert not torch.allclose(loaded.position, det.position)
    assert loaded._position_fn.__name__ == "_median_from_segmentation"


def test_bbox_detections_save_load_compress_and_cache(tmp_path: pathlib.Path, seg_3d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_3d, compress=not byotrack.ZSTD_SEG, cache=False)
    path = tmp_path / "det.pt"
    det.save(path)
    loaded = byotrack.Detections.load(path)
    assert loaded._use_cache != det._use_cache
    assert loaded._compress != det._compress

    loaded = byotrack.Detections.load(path, compress=not byotrack.ZSTD_SEG, cache=False)

    assert loaded._use_cache == det._use_cache
    assert loaded._compress == det._compress


## Cache & Compress


def test_seg_detections_without_cache_and_compress(seg_2d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_2d, cache=False, compress=False)

    assert len(det._cache) == 0

    pos = det.position
    bbox = det.bbox
    seg = det.segmentation
    mass = det.mass

    assert len(det._cache) == 0

    assert det.position is not pos
    assert det.bbox is not bbox
    assert det.segmentation is seg  # Always cached in det._segmentation
    assert det.mass is not mass


def test_seg_detections_without_cache_with_compress(seg_2d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_2d, cache=False, compress=True)

    assert len(det._cache) == 0

    pos = det.position
    bbox = det.bbox
    seg = det.segmentation
    mass = det.mass

    assert len(det._cache) == 0

    assert det._segmentation.dtype == torch.uint8  # Compressed

    assert det.position is not pos
    assert det.bbox is not bbox
    assert det.segmentation is not seg  # Decompressed
    assert det.mass is not mass


def test_seg_detections_cache_without_compress(seg_2d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_2d, cache=True, compress=False)

    assert len(det._cache) == 0

    pos = det.position
    bbox = det.bbox
    seg = det.segmentation
    mass = det.mass

    assert set(det._cache) == {"position", "bbox", "mass"}

    assert det._segmentation is seg  # Uncompressed

    assert det.position is pos
    assert det.bbox is bbox
    assert det.segmentation is seg
    assert det.mass is mass


def test_bbox_detections_cache_with_compress(seg_2d: torch.Tensor) -> None:
    det = byotrack.SegmentationDetections(seg_2d, cache=True, compress=True)

    assert len(det._cache) == 0

    pos = det.position
    bbox = det.bbox
    seg = det.segmentation
    mass = det.mass

    assert set(det._cache) == {"position", "bbox", "mass"}

    assert det._segmentation.dtype == torch.uint8  # Compressed

    assert det.position is pos
    assert det.bbox is bbox
    assert det.segmentation is not seg  # Decompressed
    assert det.mass is mass


## Helpers behaviors that are not yet checked


def test_median_from_segmentation_non_consecutive():
    seg = np.zeros((10, 10), dtype=np.int32)
    seg[2:4, 2:4] = 1
    seg[6:9, 6:8] = 4

    median = _median_from_segmentation(seg)

    assert np.allclose(median[0], 2.5)
    assert np.isnan(median[1:3]).all()
    assert np.allclose(median[3, 0], 7)
    assert np.allclose(median[3, 1], 6.5)
