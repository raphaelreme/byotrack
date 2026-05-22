from __future__ import annotations

import pytest
import torch

import byotrack
from byotrack.api.detections import SegmentationDetections


@pytest.fixture
def tracks_2d() -> list[byotrack.Track]:
    return [
        byotrack.Track(0, torch.tensor([[20.0, 10.0], [21.0, 11.0]]), 0),
        byotrack.Track(1, torch.tensor([[30.0, 15.0], [31.0, 16.0]]), 1),
    ]


@pytest.fixture
def tracks_3d() -> list[byotrack.Track]:
    return [byotrack.Track(0, torch.tensor([[2.0, 20.0, 10.0], [2.5, 21.0, 11.0]]), 0)]


@pytest.fixture
def seg_2d_det() -> SegmentationDetections:
    seg = torch.zeros(10, 12, dtype=torch.int32)
    seg[2:5, 3:7] = 1
    seg[6:9, 8:11] = 2
    return SegmentationDetections(seg)


@pytest.fixture
def seg_3d_det() -> SegmentationDetections:
    seg = torch.zeros(4, 10, 12, dtype=torch.int32)
    seg[1:3, 2:5, 3:7] = 1
    return SegmentationDetections(seg)
