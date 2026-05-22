from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
import torch

import byotrack
from byotrack.fiji.io import load_tracks, save_detections

from .conftest import _write_minimal_trackmate_xml

if TYPE_CHECKING:
    import pathlib

    from byotrack.api.detections import SegmentationDetections


## save_detections


def test_save_detections_2d(tmp_path: pathlib.Path, seg_2d_det: SegmentationDetections):
    path = tmp_path / "out.tiff"
    save_detections([seg_2d_det] * 3, path)

    with tifffile.TiffFile(path) as tif:
        assert tif.is_imagej

    data = tifffile.imread(path)

    assert data.dtype == np.uint16
    assert data.shape == (3, 10, 12)
    assert (data[0] == seg_2d_det.segmentation.numpy().astype(np.uint16)).all()


def test_save_detections_3d(tmp_path: pathlib.Path, seg_3d_det: SegmentationDetections):
    path = tmp_path / "out.tiff"
    save_detections([seg_3d_det] * 2, path)
    # XXX: Loading these tiff with T=1 squeeze T. Is this a problem for ImageJ ?

    with tifffile.TiffFile(path) as tif:
        assert tif.is_imagej

    data = tifffile.imread(path)

    assert data.dtype == np.uint16
    assert data.shape == (2, 4, 10, 12)
    assert (data[0] == seg_3d_det.segmentation.numpy().astype(np.uint16)).all()


def test_save_detections_empty_raises(tmp_path: pathlib.Path):
    path = tmp_path / "out.tiff"

    with pytest.raises(ValueError, match="No detections to save"):
        save_detections([], path)


def test_save_detections_with_outside_labels_ignores(tmp_path: pathlib.Path):
    path = tmp_path / "rois.xml"

    # 3 detections as circle, where the second detections is fully outside.
    point_det = byotrack.PointDetections(
        torch.tensor([[10.0, 10.0], [-10.0, -10.0], [-2.0, 5.0]]), radius=4, shape=(20, 20)
    )

    save_detections([point_det], path)
    data = tifffile.imread(path)

    assert byotrack.labels_of(data).shape == (2,)


## load_tracks


def test_load_tracks_2d_basic(trackmate_2d_path: pathlib.Path):
    tracks = load_tracks(trackmate_2d_path)

    assert len(tracks) == 2
    for track in tracks:
        assert track.start == 0
        assert len(track) == 2
        assert track.dim == 2

    tracks.sort(key=lambda t: t.identifier)

    # Track 0: spots (x=10, y=20) => stored as (y, x) = (20, 10)
    assert torch.allclose(tracks[0].points[0], torch.tensor([20.0, 10.0]))
    assert torch.allclose(tracks[0].points[1], torch.tensor([21.0, 11.0]))

    identifiers = {t.identifier for t in tracks}
    assert identifiers == {0, 1}


def test_load_tracks_3d_z_preserved(trackmate_3d_path: pathlib.Path):
    tracks = load_tracks(trackmate_3d_path)

    assert len(tracks) == 1
    assert tracks[0].dim == 3
    # spot (x=10, y=20, z=2) => stored as (z, y, x) = (2, 20, 10)
    assert torch.allclose(tracks[0].points[0], torch.tensor([2.0, 20.0, 10.0]))
    assert torch.allclose(tracks[0].points[1], torch.tensor([3.0, 21.0, 11.0]))


def test_load_tracks_temporal_gap(tmp_path: pathlib.Path):
    spots = {
        1: (0, 10.0, 20.0, 0.0),
        2: (2, 11.0, 21.0, 0.0),  # frame 1 is missing
    }
    path = tmp_path / "gap.xml"
    _write_minimal_trackmate_xml(path, spots, [(0, [(1, 2)])])

    tracks = load_tracks(path)
    assert len(tracks) == 1
    assert len(tracks[0]) == 3
    assert torch.isnan(tracks[0].points[1]).all()


def test_load_tracks_splitting_raises(tmp_path: pathlib.Path):
    # Spots 2 and 3 are both at frame 1 => after sorting they appear consecutively
    # causing frame <= old_frame on the second one
    spots = {
        1: (0, 10.0, 20.0, 0.0),
        2: (1, 11.0, 21.0, 0.0),
        3: (1, 12.0, 22.0, 0.0),
    }
    path = tmp_path / "split.xml"
    _write_minimal_trackmate_xml(path, spots, [(0, [(1, 2), (1, 3)])])

    with pytest.raises(NotImplementedError):  # TODO: support splitting?
        load_tracks(path)


def test_load_tracks_no_model_raises(tmp_path: pathlib.Path):
    path = tmp_path / "bad.xml"
    path.write_text("<TrackMate></TrackMate>")

    with pytest.raises(ValueError, match="No tracks"):
        load_tracks(path)
