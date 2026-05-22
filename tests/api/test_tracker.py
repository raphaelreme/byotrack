from __future__ import annotations

import io
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

import byotrack
from byotrack.api.tracker import PauseableTQDM

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

from tests.api.test_detector import MinimalBatchDetector
from tests.api.test_linker import MinimalOnlineLinker


class MinimalDetector(byotrack.Detector):
    """Detector stub returning one PointDetections per video frame."""

    @override
    def run(self, video: Sequence[np.ndarray] | np.ndarray) -> list[byotrack.PointDetections]:
        return [byotrack.PointDetections(torch.tensor([[1.0, 2.0]])) for _ in video]


class MinimalLinker(byotrack.Linker):
    """Linker stub that always returns an empty track list."""

    @override
    def run(
        self, video: Sequence[np.ndarray] | np.ndarray | None, detections_sequence: Sequence[byotrack.DetectionsLike]
    ) -> list[byotrack.Track]:

        if len(detections_sequence) == 0:
            return []

        points = torch.stack(
            [
                detections.position[0] if detections.length else torch.full((detections.dim,), torch.nan)
                for detections in map(byotrack.as_detections, detections_sequence)
            ]
        )

        return [byotrack.Track(0, points)]


class MinimalRefiner(byotrack.Refiner):
    """Refiner stub that returns tracks unchanged."""

    def __init__(self):
        super().__init__()
        self.called = False

    @override
    def run(
        self, video: Sequence[np.ndarray] | np.ndarray | None, tracks: Collection[byotrack.Track]
    ) -> Collection[byotrack.Track]:
        self.called = True
        return tracks


def test_multi_step_tracker_empty_returns_empty():
    detector = MinimalDetector()
    linker = MinimalLinker()
    tracker = byotrack.MultiStepTracker(detector, linker)

    result = tracker.run([])

    assert len(result) == 0


def test_multi_step_tracker_basic_run():
    detector = MinimalDetector()
    linker = MinimalLinker()
    tracker = byotrack.MultiStepTracker(detector, linker)

    video = [np.zeros((10, 10, 1)) for _ in range(3)]
    result = tracker.run(video)

    assert len(result) == 1
    assert len(next(iter(result))) == 3


def test_multi_step_tracker_with_refiner():
    detector = MinimalDetector()
    linker = MinimalLinker()
    refiner = MinimalRefiner()
    tracker = byotrack.MultiStepTracker(detector, linker, refiners=[refiner])

    video = [np.zeros((10, 10, 1)) for _ in range(3)]
    result = tracker.run(video)

    assert len(result) == 1
    assert len(next(iter(result))) == 3
    assert refiner.called


def test_multi_step_tracker_check_tracks_warns_on_bad_result():
    """Linker returning tracks with duplicate identifiers should trigger a warning."""

    class DuplicateIdLinker(MinimalLinker):
        @override
        def run(self, video, detections_sequence):
            return [
                byotrack.Track(0, torch.ones(2, 2), identifier=666),
                byotrack.Track(0, torch.ones(2, 2), identifier=666),  # duplicate
            ]

    detector = MinimalDetector()
    tracker = byotrack.MultiStepTracker(detector, DuplicateIdLinker())
    video = [np.zeros((10, 10, 1)) for _ in range(2)]

    with pytest.warns(UserWarning, match="identifier"):
        tracker.run(video)


## BatchMultiStepTracker


def test_batch_multi_step_tracker_empty_video():
    detector = MinimalBatchDetector(batch_size=3)
    linker = MinimalOnlineLinker()
    tracker = byotrack.BatchMultiStepTracker(detector, linker)
    result = tracker.run(np.zeros((0, 10, 10, 1)))
    assert list(result) == []


def test_batch_multi_step_tracker_2d_basics():
    detector = MinimalBatchDetector(batch_size=3)
    linker = MinimalOnlineLinker()
    tracker = byotrack.BatchMultiStepTracker(detector, linker)

    # 2 full batches
    video = [np.zeros((10, 10, 1)) for _ in range(6)]

    tracks = tracker.run(video)

    assert detector.batches == [3, 3]
    assert linker.calls[0] == ("reset", 2)  # Reset with dim=2
    assert set(linker.calls[1:7]).pop() == ("update", False, 1)  # update 6 times
    assert linker.calls[-1] == "collect"
    assert len(linker.calls) == 8

    assert tracks is linker.tracks


def test_batch_multi_step_tracker_3d_basics():
    detector = MinimalBatchDetector(batch_size=5)
    linker = MinimalOnlineLinker()
    tracker = byotrack.BatchMultiStepTracker(detector, linker)

    # Single batch
    video = [np.zeros((10, 10, 10, 1)) for _ in range(5)]

    tracks = tracker.run(video)

    assert detector.batches == [5]
    assert linker.calls[0] == ("reset", 3)  # Reset with dim=2
    assert set(linker.calls[1:6]).pop() == ("update", False, 1)  # update 5 times
    assert linker.calls[-1] == "collect"
    assert len(linker.calls) == 7

    assert tracks is linker.tracks


def test_batch_multi_step_tracker_batch_1():
    detector = MinimalBatchDetector(batch_size=1)
    linker = MinimalOnlineLinker()
    tracker = byotrack.BatchMultiStepTracker(detector, linker)

    video = np.zeros((3, 10, 10, 1))
    tracks = tracker.run(video)

    assert detector.batches == [1, 1, 1]
    assert linker.calls[0] == ("reset", 2)
    assert set(linker.calls[1:4]).pop() == ("update", False, 1)  # update 5 times
    assert linker.calls[-1] == "collect"
    assert len(linker.calls) == 5

    assert tracks is linker.tracks


def test_batch_multi_step_tracker_smaller_batch():
    detector = MinimalBatchDetector(batch_size=7)
    linker = MinimalOnlineLinker()
    tracker = byotrack.BatchMultiStepTracker(detector, linker)

    video = np.zeros((5, 10, 10, 1))
    tracks = tracker.run(video)

    assert detector.batches == [5]
    assert linker.calls[0] == ("reset", 2)
    assert set(linker.calls[1:6]).pop() == ("update", False, 1)  # update 5 times
    assert linker.calls[-1] == "collect"
    assert len(linker.calls) == 7

    assert tracks is linker.tracks


def test_batch_multi_step_tracker_partial_batch():
    detector = MinimalBatchDetector(batch_size=3)
    linker = MinimalOnlineLinker()
    tracker = byotrack.BatchMultiStepTracker(detector, linker)

    video = np.zeros((7, 10, 10, 1))
    tracks = tracker.run(video)

    assert detector.batches == [3, 3, 1]
    assert linker.calls[0] == ("reset", 2)
    assert set(linker.calls[1:8]).pop() == ("update", False, 1)  # update 5 times
    assert linker.calls[-1] == "collect"
    assert len(linker.calls) == 9

    assert tracks is linker.tracks


def test_batch_multi_step_tracker_refiner_applied():
    detector = MinimalBatchDetector(batch_size=3)
    linker = MinimalOnlineLinker()
    refiner = MinimalRefiner()
    tracker = byotrack.BatchMultiStepTracker(detector, linker, refiners=[refiner])

    video = np.zeros((5, 10, 10, 1))
    tracker.run(video)

    assert refiner.called


## Pauseable TQDM


def _make_bar(total: int = 10) -> PauseableTQDM:
    return PauseableTQDM(total=total, file=io.StringIO())


def test_pauseable_tqdm_initial_state():
    bar = _make_bar()
    assert bar.last_pause_t == 0.0


def test_pauseable_tqdm_pause_sets_last_pause_t():
    bar = _make_bar()
    bar.pause()
    assert bar.last_pause_t != 0.0


def test_pauseable_tqdm_unpause_resets_last_pause_t():
    bar = _make_bar()
    bar.pause()
    bar.unpause()
    assert bar.last_pause_t == 0.0


def test_pauseable_tqdm_double_pause_warns():
    bar = _make_bar()
    bar.pause()
    with pytest.warns(UserWarning, match="already paused"):
        bar.pause()


def test_pauseable_tqdm_unpause_without_pause_warns():
    bar = _make_bar()
    with pytest.warns(UserWarning, match="not paused"):
        bar.unpause()


def test_pauseable_tqdm_update_while_paused_warns_and_unpauses():
    bar = _make_bar()
    bar.pause()
    with pytest.warns(UserWarning, match="paused"):
        bar.update(1)
    assert bar.last_pause_t == 0.0


def test_pauseable_tqdm_pause_no_refresh():
    bar = _make_bar()
    bar.pause(refresh=False)
    assert bar.last_pause_t != 0.0


def test_pauseable_tqdm_unpause_shifts_start_t():
    bar = _make_bar()
    original_start_t = bar.start_t
    bar.pause()
    bar.unpause()
    assert bar.start_t >= original_start_t


def test_pauseable_tqdm_disabled_pause_is_noop():
    bar = PauseableTQDM(total=10, disable=True)
    bar.pause()
    assert bar.last_pause_t == 0.0


def test_pauseable_tqdm_disabled_unpause_is_noop():
    bar = PauseableTQDM(total=10, disable=True)
    bar.unpause()  # no warning, no crash
    assert bar.last_pause_t == 0.0


def test_pauseable_tqdm_negative_delta_t_warns_and_skips_adjustment():
    bar = _make_bar()
    original_start_t = bar.start_t
    # Force a future pause timestamp so delta_t = _time() - last_pause_t < 0
    bar.last_pause_t = bar.start_t + 1e9
    with pytest.warns(UserWarning, match="negative pause time"):
        bar.unpause()
    assert bar.last_pause_t == 0.0  # still reset
    assert bar.start_t == original_start_t  # not adjusted


## GroundTruthDetector integration in BatchMultiStepTracker


def test_batch_multi_step_tracker_gt_without_segmentations_uses_normal_batch_flow():
    video = np.zeros((3, 5, 5, 1), dtype=np.int32)
    detector = byotrack.GroundTruthDetector(batch_size=2)
    linker = MinimalOnlineLinker()
    tracker = byotrack.BatchMultiStepTracker(detector, linker)
    tracker.run(video)

    assert linker.calls[0] == ("reset", 2)
    update_calls = [c for c in linker.calls if isinstance(c, tuple) and c[0] == "update"]
    assert len(update_calls) == 3
    assert linker.calls[-1] == "collect"


def test_batch_multi_step_tracker_gt_with_segmentations_dispatches_to_online_tracking():
    video = np.zeros((3, 5, 5, 1), dtype=np.float32)
    segmentations = np.zeros((3, 5, 5, 1), dtype=np.int32)
    segmentations[0, 2, 2, 0] = 1  # frame 0: 1 object
    segmentations[1, 1, 1, 0] = 2  # frame 1: 2 objects
    segmentations[1, 3, 3, 0] = 3
    # frame 2: 0 objects

    detector = byotrack.GroundTruthDetector(segmentations)
    linker = MinimalOnlineLinker()
    tracker = byotrack.BatchMultiStepTracker(detector, linker)
    tracker.run(video)

    update_calls = [c for c in linker.calls if isinstance(c, tuple) and c[0] == "update"]
    assert len(update_calls) == 3
    # frame is not None (from video), detection counts come from segmentations
    assert update_calls[0] == ("update", False, 1)
    assert update_calls[1] == ("update", False, 2)
    assert update_calls[2] == ("update", False, 0)


def test_batch_multi_step_tracker_gt_online_tracking_applies_refiner():
    video = np.zeros((3, 5, 5, 1), dtype=np.float32)
    segmentations = np.zeros((3, 5, 5, 1), dtype=np.int32)

    detector = byotrack.GroundTruthDetector(segmentations)
    linker = MinimalOnlineLinker()
    refiner = MinimalRefiner()
    tracker = byotrack.BatchMultiStepTracker(detector, linker, refiners=[refiner])
    tracker.run(video)

    assert refiner.called


def test_batch_multi_step_tracker_gt_online_tracking_shape_mismatch_raises():
    video = np.zeros((3, 5, 5, 1), dtype=np.float32)
    segmentations = np.zeros((3, 6, 5, 1), dtype=np.int32)

    detector = byotrack.GroundTruthDetector(segmentations)
    linker = MinimalOnlineLinker()
    tracker = byotrack.BatchMultiStepTracker(detector, linker)

    with pytest.raises(ValueError, match="shape"):
        tracker.run(video)


def test_batch_multi_step_tracker_gt_online_tracking_guard_raises_on_wrong_detector():
    video = np.zeros((3, 5, 5, 1), dtype=np.float32)
    detector = MinimalBatchDetector(batch_size=2)
    linker = MinimalOnlineLinker()
    tracker = byotrack.BatchMultiStepTracker(detector, linker)

    with pytest.raises(RuntimeError):
        tracker._online_tracking_with_ground_truth_detector(video)
