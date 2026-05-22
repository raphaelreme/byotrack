from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

import byotrack

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class MinimalBatchDetector(byotrack.BatchDetector):
    """BatchDetector stub that returns one PointDetections per frame and records batch sizes."""

    def __init__(self, batch_size: int = 3) -> None:
        super().__init__(batch_size=batch_size)
        self.batches: list[int] = []

    @override
    def detect(self, batch: np.ndarray) -> list[byotrack.PointDetections]:
        self.batches.append(batch.shape[0])
        return [byotrack.PointDetections(torch.tensor([[1.0, 2.0]])) for _ in range(batch.shape[0])]


class MinimalDetectionsRefiner(byotrack.DetectionsRefiner):
    """DetectionsRefiner stub that records each frame and detections."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray | None] = []
        self.detections_sequence: list[byotrack.DetectionsLike] = []

    @override
    def apply(self, detections: byotrack.DetectionsLike, frame: np.ndarray | None = None) -> byotrack.Detections:
        self.frames.append(frame)
        self.detections_sequence.append(detections)

        return byotrack.as_detections(detections)


## Detector


def test_batch_detector_empty_video_returns_empty():
    detector = MinimalBatchDetector(batch_size=3)
    result = detector.run(np.zeros((0, 10, 10, 1)))
    assert result == []


def test_batch_detector_single_frame():
    detector = MinimalBatchDetector(batch_size=3)
    video = np.zeros((1, 10, 10, 1))
    result = detector.run(video)
    assert len(result) == 1
    assert detector.batches == [1]


def test_batch_detector_exact_multiple_of_batch():
    detector = MinimalBatchDetector(batch_size=3)
    video = np.zeros((6, 10, 10, 1))
    result = detector.run(video)
    assert len(result) == 6
    assert detector.batches == [3, 3]


def test_batch_detector_non_multiple_batch():
    detector = MinimalBatchDetector(batch_size=3)
    video = np.zeros((7, 10, 10, 1))
    result = detector.run(video)
    assert len(result) == 7
    assert detector.batches == [3, 3, 1]


## DetectionsRefiner


def test_detections_refiner_empty_sequence():
    refiner = MinimalDetectionsRefiner()
    result = refiner.run([])

    assert len(result) == len(refiner.frames) == len(refiner.detections_sequence) == 0


def test_detections_refiner_each_frame_refined():
    refiner = MinimalDetectionsRefiner()
    dets = [byotrack.PointDetections(torch.tensor([[float(i), float(i)]])) for i in range(4)]
    result = refiner.run(dets)

    assert len(result) == len(refiner.frames) == len(refiner.detections_sequence) == 4

    assert refiner.frames[0] is None
    assert refiner.detections_sequence[0] is dets[0]


def test_detections_refiner_video_frame_passed_to_apply():
    refiner = MinimalDetectionsRefiner()
    dets = [byotrack.PointDetections(torch.tensor([[1.0, 2.0]]))] * 3
    video = [np.ones((10, 10, 1)) * i for i in range(3)]
    result = refiner.run(dets, video)

    assert len(result) == len(refiner.frames) == len(refiner.detections_sequence) == 3

    assert refiner.frames[0] is video[0]
    assert refiner.detections_sequence[0] is dets[0]


## GroundTruthDetector


def test_ground_truth_detector_detect_2d():
    batch = np.zeros((2, 5, 5, 1), dtype=np.int32)
    batch[0, 2, 2, 0] = 1  # frame 0: 1 object
    batch[1, 1, 1, 0] = 2  # frame 1: 2 objects
    batch[1, 3, 3, 0] = 3

    detector = byotrack.GroundTruthDetector()
    result = detector.detect(batch)

    assert len(result) == 2
    assert isinstance(result[0], byotrack.SegmentationDetections)
    assert isinstance(result[1], byotrack.SegmentationDetections)
    assert result[0].length == 1
    assert result[1].length == 2


def test_ground_truth_detector_detect_3d():
    batch = np.zeros((1, 3, 5, 5, 1), dtype=np.int32)
    batch[0, 0, 1, 1, 0] = 1
    batch[0, 2, 3, 3, 0] = 2

    detector = byotrack.GroundTruthDetector()
    result = detector.detect(batch)

    assert len(result) == 1
    assert isinstance(result[0], byotrack.SegmentationDetections)
    assert result[0].length == 2
    assert result[0].dim == 3


def test_ground_truth_detector_detect_rejects_multichannel():
    batch = np.zeros((1, 5, 5, 2), dtype=np.int32)
    detector = byotrack.GroundTruthDetector()
    with pytest.raises(ValueError, match="Multichannel"):
        detector.detect(batch)


def test_ground_truth_detector_detect_rejects_float():
    batch = np.zeros((1, 5, 5, 1), dtype=np.float32)
    detector = byotrack.GroundTruthDetector()
    with pytest.raises(ValueError, match="integer"):
        detector.detect(batch)


def test_ground_truth_detector_run_without_segmentations_uses_video():
    video = np.zeros((3, 5, 5, 1), dtype=np.int32)
    video[0, 2, 2, 0] = 1

    detector = byotrack.GroundTruthDetector()
    result = detector.run(video)

    assert len(result) == 3
    assert result[0].length == 1
    assert result[1].length == 0


def test_ground_truth_detector_run_with_segmentations_uses_them():
    video = np.zeros((2, 5, 5, 3), dtype=np.float32)  # no labels
    segmentations = np.zeros((2, 5, 5, 1), dtype=np.int32)
    segmentations[0, 2, 2, 0] = 7  # 1 object in frame 0

    detector = byotrack.GroundTruthDetector(segmentations)
    result = detector.run(video)

    assert len(result) == 2
    assert result[0].length == 1  # came from segmentations, not video
    assert result[1].length == 0


def test_ground_truth_detector_run_shape_mismatch_raises():
    video = np.zeros((3, 5, 5, 1), dtype=np.int32)
    segmentations = np.zeros((3, 6, 5, 1), dtype=np.int32)

    detector = byotrack.GroundTruthDetector(segmentations)
    with pytest.raises(ValueError, match="shape"):
        detector.run(video)

    detector = byotrack.GroundTruthDetector(list(segmentations))
    with pytest.raises(ValueError, match="shape"):
        detector.run(list(video))


def test_ground_truth_detector_check_shape_noop_without_segmentation():
    video = np.zeros((3, 5, 5, 1), dtype=np.int32)

    detector = byotrack.GroundTruthDetector()
    detector._check_shape(video)
