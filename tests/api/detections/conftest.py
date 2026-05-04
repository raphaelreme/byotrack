from __future__ import annotations

import pytest
import torch


@pytest.fixture
def pos_2d() -> torch.Tensor:
    return torch.rand(2, 2)


@pytest.fixture
def pos_3d() -> torch.Tensor:
    return torch.rand(4, 3)


@pytest.fixture
def seg_2d() -> torch.Tensor:
    seg = torch.zeros(30, 30, dtype=torch.int32)
    seg[2:6, 2:6] = 1
    seg[10:16, 10:16] = 2
    seg[20:25, 20:25] = 3
    return seg


@pytest.fixture
def seg_3d() -> torch.Tensor:
    seg = torch.zeros(10, 20, 20, dtype=torch.int32)
    seg[1:4, 2:6, 2:6] = 1
    seg[5:8, 10:15, 10:15] = 2
    return seg


@pytest.fixture
def bbox_2d() -> torch.Tensor:
    return torch.randint(1, 10, (2, 4), dtype=torch.int32)


@pytest.fixture
def bbox_3d() -> torch.Tensor:
    return torch.randint(1, 10, (1, 6), dtype=torch.int32)
