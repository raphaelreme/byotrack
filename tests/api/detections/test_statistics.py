from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

import byotrack
from byotrack.api.detections import statistics

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

## average_mass


def test_average_mass_empty_sequence() -> None:
    assert statistics.average_mass([]) == 0.0


def test_average_mass_no_detections() -> None:
    empty_det = byotrack.PointDetections(torch.zeros(0, 2), shape=(10, 10))
    assert statistics.average_mass([empty_det, empty_det]) == 0.0


def test_average_mass_single_frame() -> None:
    # BBox masses: 3*4=12, 5*5=25; average = 18.5
    bbox = torch.tensor([[0, 0, 3, 4], [0, 0, 5, 5]], dtype=torch.int32)
    det = byotrack.BBoxDetections(bbox, shape=(30, 30))
    assert statistics.average_mass([det]) == pytest.approx(18.5)


def test_average_mass_multiple_frames() -> None:
    # Frame 1: mass=4, frame 2: mass=16; total count=2, average=10.0
    det1 = byotrack.BBoxDetections(torch.tensor([[0, 0, 2, 2]], dtype=torch.int32), shape=(10, 10))
    det2 = byotrack.BBoxDetections(torch.tensor([[0, 0, 4, 4]], dtype=torch.int32), shape=(10, 10))
    assert statistics.average_mass([det1, det2]) == pytest.approx(10.0)


def test_average_mass_ignores_empty_frames() -> None:
    det = byotrack.BBoxDetections(torch.tensor([[0, 0, 4, 4]], dtype=torch.int32), shape=(10, 10))
    empty_det = byotrack.PointDetections(torch.zeros(0, 2), shape=(10, 10))
    # Only det contributes: average = 16.0
    assert statistics.average_mass([empty_det, det, empty_det]) == pytest.approx(16.0)


## average_radius


def test_average_radius_empty_sequence() -> None:
    assert statistics.average_radius([]) == 0.0


def test_average_radius_2d() -> None:
    # mass = 4*4 = 16; R = sqrt(16 / π)
    det = byotrack.BBoxDetections(torch.tensor([[0, 0, 4, 4]], dtype=torch.int32), shape=(10, 10))
    expected = math.sqrt(16.0 / math.pi)
    assert statistics.average_radius([det]) == pytest.approx(expected)


def test_average_radius_3d() -> None:
    # mass = 3*3*3 = 27; R = (27 * 3 / 4 / π) ** (1/3)
    det = byotrack.BBoxDetections(torch.tensor([[0, 0, 0, 3, 3, 3]], dtype=torch.int32), shape=(10, 10, 10))
    expected = (27.0 * 3 / 4 / math.pi) ** (1 / 3)
    assert statistics.average_radius([det]) == pytest.approx(expected)


def test_average_radius_anisotropy() -> None:
    # mass = 4*4*4 = 64; scaled mass = 64 * prod((2, 1, 1)) = 128; R = (128 * 3 / 4 / π) ** (1/3)
    det = byotrack.BBoxDetections(torch.tensor([[0, 0, 0, 4, 4, 4]], dtype=torch.int32), shape=(10, 10, 10))
    anisotropy = (2.0, 1.0, 1.0)
    expected = (128.0 * 3 / 4 / math.pi) ** (1 / 3)
    assert statistics.average_radius([det], anisotropy=anisotropy) == pytest.approx(expected)


## average_min_dist


def test_average_min_dist_empty_sequence() -> None:
    assert statistics.average_min_dist([]) == 0.0


def test_average_min_dist_single_detection() -> None:
    det = byotrack.PointDetections(torch.tensor([[5.0, 5.0]]), shape=(20, 20))
    assert statistics.average_min_dist([det]) == 0.0


def test_average_min_dist_2d() -> None:
    # Two detections at (0, 0) and (3, 4): distance = 5.0
    pos = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    det = byotrack.PointDetections(pos, shape=(10, 10))
    assert statistics.average_min_dist([det]) == pytest.approx(5.0)


def test_average_min_dist_3d() -> None:
    # Two detections at (0, 0, 0) and (1, 0, 0): distance = 1.0
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    det = byotrack.PointDetections(pos, shape=(10, 10, 10))
    assert statistics.average_min_dist([det]) == pytest.approx(1.0)


def test_average_min_dist_anisotropy() -> None:
    # Two 3D detections at (0, 0, 0) and (1, 0, 0): without anisotropy = 1.0
    # With anisotropy (2, 1, 1): scaled Z distance = 2.0
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    det = byotrack.PointDetections(pos, shape=(10, 10, 10))
    assert statistics.average_min_dist([det]) == pytest.approx(1.0)
    assert statistics.average_min_dist([det], anisotropy=(2.0, 1.0, 1.0)) == pytest.approx(2.0)


def test_average_min_dist_multiple_frames() -> None:
    # Frame 1: distance 3.0, frame 2: distance 4.0; average = 3.5
    det1 = byotrack.PointDetections(torch.tensor([[0.0, 0.0], [3.0, 0.0]]), shape=(10, 10))
    det2 = byotrack.PointDetections(torch.tensor([[0.0, 0.0], [4.0, 0.0]]), shape=(10, 10))
    assert statistics.average_min_dist([det1, det2]) == pytest.approx(3.5)


## anisotropy


def test_anisotropy_empty_sequence() -> None:
    assert statistics.anisotropy([]) == (1.0, 1.0, 1.0)


def test_anisotropy_2d_only_depth() -> None:
    # 2D with only_depth=True always returns (1.0, 1.0, 1.0)
    bbox = torch.tensor([[0, 0, 3, 6]], dtype=torch.int32)
    det = byotrack.BBoxDetections(bbox, shape=(20, 20))
    assert statistics.anisotropy([det], only_depth=True) == (1.0, 1.0, 1.0)


def test_anisotropy_2d_full() -> None:
    # 2D bbox sizes (3, 6): sizes[-1]/sizes[0] = 6/3 = 2.0 → (1.0, 2.0, 1.0)
    bbox = torch.tensor([[0, 0, 3, 6]], dtype=torch.int32)
    det = byotrack.BBoxDetections(bbox, shape=(20, 20))
    result = statistics.anisotropy([det], only_depth=False)
    assert result == pytest.approx((1.0, 2.0, 1.0))


def test_anisotropy_3d_only_depth() -> None:
    # 3D bbox sizes (2, 4, 4): depth_anisotropy = mean(4, 4) / 2 = 2.0 → (2.0, 1.0, 1.0)
    bbox = torch.tensor([[0, 0, 0, 2, 4, 4]], dtype=torch.int32)
    det = byotrack.BBoxDetections(bbox, shape=(20, 20, 20))
    result = statistics.anisotropy([det], only_depth=True)
    assert result == pytest.approx((2.0, 1.0, 1.0))


def test_anisotropy_3d_full() -> None:
    # 3D bbox sizes (2, 4, 8): sizes[-1]/sizes = (8/2, 8/4, 8/8) = (4.0, 2.0, 1.0)
    bbox = torch.tensor([[0, 0, 0, 2, 4, 8]], dtype=torch.int32)
    det = byotrack.BBoxDetections(bbox, shape=(20, 20, 20))
    result = statistics.anisotropy([det], only_depth=False)
    assert result == pytest.approx((4.0, 2.0, 1.0))


def test_anisotropy_isotropic() -> None:
    # Isotropic 3D detections: all bbox sizes equal → anisotropy = (1.0, 1.0, 1.0)
    bbox = torch.tensor([[0, 0, 0, 5, 5, 5]], dtype=torch.int32)
    det = byotrack.BBoxDetections(bbox, shape=(20, 20, 20))
    assert statistics.anisotropy([det]) == pytest.approx((1.0, 1.0, 1.0))


def test_anisotropy_zero_sizes(mocker: MockerFixture) -> None:
    # Defensive guard: if mean bbox sizes are zero, return (1.0, 1.0, 1.0).
    # Standard Detections classes enforce non-zero bbox sizes, so we use a mock.
    mock_det = mocker.MagicMock()
    mock_det.dim = 2
    mock_det.bbox = torch.zeros(1, 4, dtype=torch.int32)  # zero sizes in last 2 columns
    assert statistics.anisotropy([mock_det]) == (1.0, 1.0, 1.0)


def test_anisotropy_warns_when_yx_anisotropic() -> None:
    # Warning is emitted when only_depth=True but Y and X axes are significantly anisotropic.
    # Bbox sizes (z=4, y=2, x=8): Y/X anisotropy ratio = 4.0 > 1.5 → warning.
    bbox = torch.tensor([[0, 0, 0, 4, 2, 8]], dtype=torch.int32)
    det = byotrack.BBoxDetections(bbox, shape=(20, 20, 20))
    with pytest.warns(UserWarning, match="but X and Y axes seems to be anistrope"):
        result = statistics.anisotropy([det], only_depth=True)

    assert result == pytest.approx((1.25, 1.0, 1.0))
