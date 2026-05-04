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


class MinimalOnlineLinker(byotrack.OnlineLinker):
    """OnlineLinker stub that records all calls for test assertions."""

    def __init__(self) -> None:
        self.calls: list = []
        self.tracks: list[byotrack.Track] = []

    @override
    def reset(self, dim: int = 2) -> None:
        self.calls.append(("reset", dim))

    @override
    def update(self, frame: np.ndarray | None, detections: byotrack.Detections) -> None:
        self.calls.append(("update", frame is None, detections.length))

    @override
    def collect(self) -> list[byotrack.Track]:
        self.calls.append("collect")
        return self.tracks


def test_online_linker_empty_detections_pass():
    linker = MinimalOnlineLinker()
    result = linker.run(None, [])
    assert result == []
    assert linker.calls == []  # reset/update/collect never called


def test_online_linker_on_2d_seq():
    linker = MinimalOnlineLinker()
    dets = [byotrack.PointDetections(torch.tensor([[float(i), float(i)]])) for i in range(5)]
    linker.run(None, dets)

    assert linker.calls[0] == ("reset", 2)  # Reset with dim=2
    assert set(linker.calls[1:6]).pop() == ("update", True, 1)  # update 5 times
    assert linker.calls[-1] == "collect"
    assert len(linker.calls) == 7


def test_online_linker_on_3d_seq():
    linker = MinimalOnlineLinker()
    dets = [
        byotrack.PointDetections(torch.tensor([[10, float(i), float(i)], [float(i), 10, float(i)]])) for i in range(3)
    ]
    linker.run(None, dets)

    assert linker.calls[0] == ("reset", 3)  # Reset with dim=3
    assert set(linker.calls[1:4]).pop() == ("update", True, 2)  # update 3 times
    assert linker.calls[-1] == "collect"
    assert len(linker.calls) == 5


def test_online_linker_video_none_passes_none_to_update():
    linker = MinimalOnlineLinker()
    det = byotrack.PointDetections(torch.tensor([[1.0, 2.0]]))
    linker.run(None, [det])
    update_calls = [c for c in linker.calls if isinstance(c, tuple) and c[0] == "update"]
    # frame_is_none is the second element
    assert all(c[1] is True for c in update_calls)


def test_online_linker_video_frames_passed_to_update():
    linker = MinimalOnlineLinker()
    video = np.zeros((3, 10, 10, 1))
    dets = [byotrack.PointDetections(torch.tensor([[1.0, 2.0]]))] * 3
    linker.run(video, dets)
    update_calls = [c for c in linker.calls if isinstance(c, tuple) and c[0] == "update"]
    assert all(c[1] is False for c in update_calls)


def test_online_linker_raw_tensor_converted():
    linker = MinimalOnlineLinker()
    raw = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (N,2) float => PointDetections
    linker.run(None, [raw])
    update_calls = [c for c in linker.calls if isinstance(c, tuple) and c[0] == "update"]
    assert update_calls[0][2] == 2  # detection length


def test_online_linker_returns_collect_result():
    linker = MinimalOnlineLinker()
    t = byotrack.Track(0, torch.ones(3, 2))
    linker.tracks = [t]
    det = byotrack.PointDetections(torch.tensor([[1.0, 2.0]]))
    result = linker.run(None, [det])
    assert result == [t]


def test_online_linker_fewer_video_frames_warns():
    linker = MinimalOnlineLinker()
    video = np.zeros((2, 10, 10, 1))  # 2 frames
    dets = [byotrack.PointDetections(torch.tensor([[1.0, 2.0]]))] * 5  # 5 detections

    with pytest.warns(UserWarning, match="Found less frames"):
        linker.run(video, dets)

    update_calls = [c for c in linker.calls if isinstance(c, tuple) and c[0] == "update"]
    assert len(update_calls) == 2  # truncated to video length


def test_online_linker_more_video_frames_warns():
    linker = MinimalOnlineLinker()
    video = np.zeros((5, 10, 10, 1))  # 5 frames
    dets = [byotrack.PointDetections(torch.tensor([[1.0, 2.0]]))] * 2  # 2 detections

    with pytest.warns(UserWarning, match="Found more frames"):
        linker.run(video, dets)

    update_calls = [c for c in linker.calls if isinstance(c, tuple) and c[0] == "update"]
    assert len(update_calls) == 2  # truncated to detection count


def test_online_linker_check_tracks_warns_on_bad_ids():
    linker = MinimalOnlineLinker()
    # Two tracks with duplicate identifiers => check_tracks warns
    linker.tracks = [
        byotrack.Track(0, torch.ones(2, 2), identifier=1001),
        byotrack.Track(0, torch.ones(2, 2), identifier=1001),  # duplicate
    ]
    det = byotrack.PointDetections(torch.tensor([[1.0, 2.0]]))
    with pytest.warns(UserWarning, match="identifier"):
        linker.run(None, [det])
