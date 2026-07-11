"""Tests for byotrack.dataset.ctc."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
import torch

import byotrack
from byotrack.dataset.ctc import (
    _build_radii,
    _parse_meta_data,
    _save_metadata,
    load_detections,
    load_tracks,
    save_detections,
    save_tracks,
)

if TYPE_CHECKING:
    import pathlib

T = 5
SHAPE_2D = (20, 20)
SHAPE_3D = (5, 20, 20)
RADIUS = 3.0


def _write_tiff_2d(path: pathlib.Path, label: int, frame_id: int, n_digit: int = 4) -> None:
    """Write a minimal 20x20 CTC 2D tiff with one labeled region."""
    seg = np.zeros(SHAPE_2D, dtype=np.uint16)
    seg[3:6, 3:6] = label
    seg_imagej = seg[None, None, None, ..., None]  # TZCYXS
    name = f"mask{frame_id:0{n_digit}}.tif"
    tifffile.imwrite(path / name, seg_imagej, imagej=True, compression="zlib")


@pytest.fixture
def detections_2d() -> list[byotrack.Detections]:
    seg = torch.zeros(*SHAPE_2D, dtype=torch.int32)
    seg[3:6, 3:6] = 1
    seg[10:13, 10:13] = 2
    det = byotrack.SegmentationDetections(seg)
    return [det] * T


@pytest.fixture
def detections_3d() -> list[byotrack.Detections]:
    seg = torch.zeros(*SHAPE_3D, dtype=torch.int32)
    seg[1:4, 3:6, 3:6] = 1
    seg[1:4, 10:13, 10:13] = 2
    det = byotrack.SegmentationDetections(seg)
    return [det] * T


def _make_tracks_2d() -> list[byotrack.Track]:
    """Two 2D tracks with integer positions, identifiers 1 and 2."""
    points_1 = torch.full((T, 2), 5.0)
    points_2 = torch.full((T, 2), 14.0)
    return [
        byotrack.Track(0, points_1, identifier=1),
        byotrack.Track(0, points_2, identifier=3),
    ]


def _make_tracks_3d() -> list[byotrack.Track]:
    """Two 3D tracks with integer positions, identifiers 1 and 2."""
    points_1 = torch.tensor([[2.0, 5.0, 5.0]] * T)
    points_2 = torch.tensor([[2.0, 14.0, 14.0]] * T)
    return [
        byotrack.Track(0, points_1, identifier=1),
        byotrack.Track(0, points_2, identifier=2),
    ]


def _write_tiff_multichannel(path: pathlib.Path, frame_id: int, n_digit: int = 4) -> None:
    """Write a 20x20x3 multichannel tiff (not CTC-compatible)."""
    seg = np.zeros((*SHAPE_2D, 3), dtype=np.uint16)
    name = f"mask{frame_id:0{n_digit}}.tif"
    tifffile.imwrite(path / name, seg, compression="zlib")


def _write_seg_tiff(path: pathlib.Path, name: str, label: int) -> None:
    """Write a minimal 20x20 tiff with one labeled region under an arbitrary file name."""
    seg = np.zeros(SHAPE_2D, dtype=np.uint16)
    seg[3:6, 3:6] = label
    seg_imagej = seg[None, None, None, ..., None]  # TZCYXS
    tifffile.imwrite(path / name, seg_imagej, imagej=True, compression="zlib")


# --- _parse_meta_data ---


def test_parse_meta_data_basic(tmp_path: pathlib.Path) -> None:
    txt = tmp_path / "track.txt"
    txt.write_text("1 0 4 0\n2 2 6 0", encoding="utf-8")
    meta = _parse_meta_data(txt)
    assert meta == {0: (0, 4, 0), 1: (2, 6, 0)}


def test_parse_meta_data_with_parent(tmp_path: pathlib.Path) -> None:
    txt = tmp_path / "track.txt"
    txt.write_text("2 2 6 1", encoding="utf-8")
    meta = _parse_meta_data(txt)
    # Parent is stored raw (value from file, not offset)
    assert meta[1] == (2, 6, 1)


def test_parse_meta_data_identifier_offset(tmp_path: pathlib.Path) -> None:
    txt = tmp_path / "track.txt"
    txt.write_text("3 0 2 0", encoding="utf-8")
    meta = _parse_meta_data(txt)
    assert 2 in meta
    assert 3 not in meta


# --- load_detections ---


def test_load_detections_standard_naming(tmp_path: pathlib.Path) -> None:
    for i in range(T):
        _write_tiff_2d(tmp_path, label=1, frame_id=i)  # mask0000.tif, mask0001.tif, ...

    detections = load_detections(tmp_path)

    assert len(detections) == T
    for detections_frame in detections:
        assert detections_frame.shape == SHAPE_2D
        assert (detections_frame.segmentation[3:6, 3:6] == 1).all()


def test_load_detections_non_standard_naming(tmp_path: pathlib.Path) -> None:
    for i in range(T):
        _write_seg_tiff(tmp_path, f"frame{i}.tiff", label=i + 1)

    detections = load_detections(tmp_path)

    assert len(detections) == T
    for i, detections_frame in enumerate(detections):
        assert (detections_frame.segmentation[3:6, 3:6] == 1).all()
        assert detections_frame.labels.tolist() == [i]  # Offset of 1 with the raw label i + 1


def test_load_detections_roundtrip(tmp_path: pathlib.Path, detections_2d) -> None:
    save_detections(detections_2d, tmp_path, as_res=True)

    loaded = load_detections(tmp_path)

    assert len(loaded) == T
    for orig, got in zip(detections_2d, loaded, strict=True):
        assert torch.equal(got.segmentation, orig.segmentation)


# --- load_tracks ---


def test_load_tracks_res_mode(tmp_path: pathlib.Path) -> None:
    tracks = _make_tracks_2d()
    path = tmp_path / "res"
    save_tracks(tracks, path, shape=SHAPE_2D, as_res=True, default_radius=RADIUS)
    loaded = load_tracks(path)
    assert len(loaded) == 2
    identifiers = {t.identifier for t in loaded}
    assert identifiers == {1, 3}


def test_load_tracks_man_track_mode(tmp_path: pathlib.Path) -> None:
    tracks = _make_tracks_2d()
    path = tmp_path / "man"
    save_tracks(tracks, path, shape=SHAPE_2D, as_res=False, as_seg=False, default_radius=RADIUS)
    loaded = load_tracks(path)
    assert len(loaded) == 2
    identifiers = {t.identifier for t in loaded}
    assert identifiers == {1, 3}


def test_load_tracks_man_seg_mode(tmp_path: pathlib.Path) -> None:
    tracks = _make_tracks_2d()
    # Directory name must contain "seg" to trigger the man_seg naming branch
    path = tmp_path / "man_seg"
    save_tracks(tracks, path, shape=SHAPE_2D, as_res=False, as_seg=True, default_radius=RADIUS)
    loaded = load_tracks(path)
    assert len(loaded) == 2
    identifiers = {t.identifier for t in loaded}
    assert identifiers == {1, 3}


def test_load_tracks_no_metadata_warns(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "man"
    path.mkdir()

    # No metadata file => is_res=False => code globs man_*.tif
    seg = np.zeros(SHAPE_2D, dtype=np.uint16)
    seg[3:6, 3:6] = 1
    seg_imagej = seg[None, None, None, ..., None]
    tifffile.imwrite(path / "man_track0000.tif", seg_imagej, imagej=True, compression="zlib")

    with pytest.warns(UserWarning, match="res_track.txt or man_track.txt not found"):
        loaded = load_tracks(path)

    assert len(loaded) == 1
    assert loaded[0].parent_id == -1


def test_load_tracks_missing_id_in_metadata_warns(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "man"
    path.mkdir()

    # Metadata only mentions track CTC-ID=1 (byotrack id=0), but tiff has labels 1 and 2
    (path / "man_track.txt").write_text("1 0 0 0", encoding="utf-8")
    seg = np.zeros(SHAPE_2D, dtype=np.uint16)
    seg[3:6, 3:6] = 1
    seg[10:13, 10:13] = 2
    seg_imagej = seg[None, None, None, ..., None]
    tifffile.imwrite(path / "man_track0000.tif", seg_imagej, imagej=True, compression="zlib")

    with pytest.warns(UserWarning, match="Missing identifier"):
        load_tracks(path)


def test_load_tracks_frame_before_start_raises(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "res"
    path.mkdir()
    # Metadata says track starts at frame 2, but mask0000.tif has label 1 at frame 0
    (path / "res_track.txt").write_text("1 2 4 0", encoding="utf-8")
    _write_tiff_2d(path, label=1, frame_id=0)

    with pytest.raises(ValueError, match="before for it started"):
        load_tracks(path)


def test_load_tracks_frame_after_last_raises(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "res"
    path.mkdir()
    # Metadata says track ends at frame 0, but mask0004.tif has label 1
    (path / "res_track.txt").write_text("1 0 0 0", encoding="utf-8")
    _write_tiff_2d(path, label=1, frame_id=4)

    with pytest.raises(ValueError, match="after for it ended"):
        load_tracks(path)


def test_load_tracks_multichannel_raises(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "res"
    path.mkdir()
    (path / "res_track.txt").write_text("1 0 0 0", encoding="utf-8")
    _write_tiff_multichannel(path, frame_id=0)

    with pytest.raises(ValueError, match="Multichannel"):
        load_tracks(path)


def test_load_tracks_with_parent(tmp_path: pathlib.Path) -> None:
    tracks = [
        byotrack.Track(0, torch.full((2, 2), 5.0), identifier=1),
        byotrack.Track(2, torch.full((2, 2), 14.0), identifier=2, parent_id=1),
        byotrack.Track(2, torch.full((2, 2), 5.0), identifier=3, parent_id=1),
    ]
    path = tmp_path / "res"
    save_tracks(tracks, path, shape=SHAPE_2D, as_res=True, default_radius=RADIUS)
    loaded = sorted(load_tracks(path), key=lambda t: t.identifier)

    assert loaded[0].identifier == 1
    assert loaded[0].parent_id == -1

    assert loaded[1].identifier == 2
    assert loaded[1].parent_id == 1
    assert loaded[1].start == 2

    assert loaded[2].identifier == 3
    assert loaded[2].parent_id == 1
    assert loaded[2].start == 2


# --- save_detections ---


def test_save_detections_as_res_naming(tmp_path: pathlib.Path, detections_2d) -> None:
    save_detections(detections_2d, tmp_path, as_res=True)

    for i in range(T):
        assert (tmp_path / f"mask{i:04d}.tif").exists()
    assert not (tmp_path / "man_track0000.tif").exists()


def test_save_detections_as_man_track_naming(tmp_path: pathlib.Path, detections_2d) -> None:
    save_detections(detections_2d, tmp_path, as_res=False, as_seg=False)

    for i in range(T):
        assert (tmp_path / f"man_track{i:04d}.tif").exists()
    assert not (tmp_path / "mask0000.tif").exists()


def test_save_detections_as_seg_naming(tmp_path: pathlib.Path, detections_2d) -> None:
    save_detections(detections_2d, tmp_path, as_res=False, as_seg=True)

    for i in range(T):
        assert (tmp_path / f"man_seg{i:04d}.tif").exists()


def test_save_detections_n_digit(tmp_path: pathlib.Path, detections_2d) -> None:
    save_detections([detections_2d[0]], tmp_path, as_res=True, n_digit=3)

    assert (tmp_path / "mask000.tif").exists()
    assert not (tmp_path / "mask0000.tif").exists()


def test_save_detections_creates_directory(tmp_path: pathlib.Path, detections_2d) -> None:
    nested = tmp_path / "a" / "b" / "c"
    save_detections([detections_2d[0]], nested)
    assert (nested / "mask0000.tif").exists()


def test_save_detections_2d_roundtrip(tmp_path: pathlib.Path) -> None:
    seg = torch.zeros(*SHAPE_2D, dtype=torch.int32)
    seg[3:6, 3:6] = 1
    det = byotrack.SegmentationDetections(seg)
    save_detections([det], tmp_path, as_res=True)

    loaded = tifffile.imread(tmp_path / "mask0000.tif")
    assert loaded.shape == SHAPE_2D
    assert (loaded[3:6, 3:6] == 1).all()
    assert loaded[0, 0] == 0


def test_save_detections_3d_roundtrip(tmp_path: pathlib.Path) -> None:
    seg = torch.zeros(*SHAPE_3D, dtype=torch.int32)
    seg[1:4, 3:6, 3:6] = 1
    det = byotrack.SegmentationDetections(seg)
    save_detections([det], tmp_path, as_res=True)

    loaded = tifffile.imread(tmp_path / "mask0000.tif")
    assert loaded.shape == SHAPE_3D
    assert (loaded[1:4, 3:6, 3:6] == 1).all()
    assert loaded[0, 0, 0] == 0


# --- _save_metadata ---


def test_save_metadata_content(tmp_path: pathlib.Path) -> None:
    track = byotrack.Track(0, torch.ones(3, 2), identifier=1)
    txt = tmp_path / "res_track.txt"
    _save_metadata([track], txt)
    # identifier + 1 = 2, start = 0, end = 2, parent_id + 1 = 0
    assert txt.read_text(encoding="utf-8").strip() == "2 0 2 0"


def test_save_metadata_with_parent(tmp_path: pathlib.Path) -> None:
    track = byotrack.Track(0, torch.ones(3, 2), identifier=2, parent_id=1)
    txt = tmp_path / "res_track.txt"
    _save_metadata([track], txt)
    parts = txt.read_text(encoding="utf-8").strip().split()
    # Last column = parent_id + 1 = 2
    assert parts[-1] == "2"


def test_save_metadata_merge_warns(tmp_path: pathlib.Path) -> None:
    track = byotrack.Track(0, torch.ones(3, 2), identifier=1, merge_id=5)
    txt = tmp_path / "res_track.txt"
    with pytest.warns(UserWarning, match="merge events"):
        _save_metadata([track], txt)
    # Metadata line is still written despite the warning
    assert txt.exists()


# --- _build_radii ---


def test_build_radii_2d() -> None:
    radii = _build_radii(3, 2, 5.0, anisotropy=1.0)
    assert radii.shape == (3, 2)
    assert np.allclose(radii, 5.0)


def test_build_radii_2d_anisotropy_ignored() -> None:
    # Anisotropy only applies to dim==3; 2D result must be unchanged
    radii = _build_radii(3, 2, 5.0, anisotropy=2.0)
    assert np.allclose(radii, 5.0)


def test_build_radii_3d_isotropic() -> None:
    radii = _build_radii(2, 3, 4.0, anisotropy=1.0)
    assert radii.shape == (2, 3)
    assert np.allclose(radii, 4.0)


def test_build_radii_3d_anisotropic() -> None:
    radii = _build_radii(2, 3, 4.0, anisotropy=2.0)
    assert np.allclose(radii[:, :2], 4.0)
    assert np.allclose(radii[:, 2], 2.0)  # 4.0 / 2.0


# --- save_tracks: structure and naming ---


def test_save_tracks_as_res_creates_files(tmp_path: pathlib.Path) -> None:
    tracks = _make_tracks_2d()
    save_tracks(tracks, tmp_path, shape=SHAPE_2D, as_res=True, default_radius=RADIUS)
    assert (tmp_path / "res_track.txt").exists()
    for i in range(T):
        assert (tmp_path / f"mask{i:04d}.tif").exists()


def test_save_tracks_as_man_track_creates_files(tmp_path: pathlib.Path) -> None:
    tracks = _make_tracks_2d()
    save_tracks(tracks, tmp_path, shape=SHAPE_2D, as_res=False, as_seg=False, default_radius=RADIUS)
    assert (tmp_path / "man_track.txt").exists()
    for i in range(T):
        assert (tmp_path / f"man_track{i:04d}.tif").exists()


def test_save_tracks_as_seg_creates_files(tmp_path: pathlib.Path) -> None:
    tracks = _make_tracks_2d()
    save_tracks(tracks, tmp_path, shape=SHAPE_2D, as_res=False, as_seg=True, default_radius=RADIUS)
    assert (tmp_path / "man_track.txt").exists()
    for i in range(T):
        assert (tmp_path / f"man_seg{i:04d}.tif").exists()


def test_save_tracks_no_shape_raises(tmp_path: pathlib.Path) -> None:
    tracks = _make_tracks_2d()
    with pytest.raises(ValueError, match="shape"):
        save_tracks(tracks, tmp_path)


def test_save_tracks_shape_mismatch_raises(tmp_path: pathlib.Path, detections_2d) -> None:
    tracks = _make_tracks_2d()
    with pytest.raises(ValueError, match="not compatible"):
        save_tracks(tracks, tmp_path, detections_sequence=detections_2d, shape=(30, 30))


def test_save_tracks_last_overwrite(tmp_path: pathlib.Path) -> None:
    tracks = _make_tracks_2d()
    save_tracks(tracks, tmp_path, shape=SHAPE_2D, as_res=True, last=7, default_radius=RADIUS)
    for i in range(8):
        assert (tmp_path / f"mask{i:04d}.tif").exists()
    assert not (tmp_path / "mask0008.tif").exists()


# --- save_tracks: disk drawing warnings ---


def test_save_tracks_missing_position_in_segment_warns(tmp_path: pathlib.Path) -> None:
    points = torch.full((T, 2), 5.0)
    points[2] = float("nan")  # NaN inside segment [0, T)
    track = byotrack.Track(0, points, identifier=1)
    with pytest.warns(UserWarning, match="missing position inside a track segment"):
        save_tracks([track], tmp_path, shape=SHAPE_2D, as_res=True, default_radius=RADIUS)


def test_save_tracks_disk_outside_image_warns(tmp_path: pathlib.Path) -> None:
    points = torch.full((T, 2), 50.0)  # Outside 20x20 image
    track = byotrack.Track(0, points, identifier=1)
    with pytest.warns(UserWarning, match="outside of image or occluded"):
        save_tracks([track], tmp_path, shape=SHAPE_2D, as_res=True, default_radius=RADIUS)


def test_save_tracks_occluded_track_warns(tmp_path: pathlib.Path) -> None:
    # Track 1 linked to a detection; Track 2 draws a disk over the same area with overwrite=True
    seg = torch.zeros(*SHAPE_2D, dtype=torch.int32)
    seg[5:12, 5:12] = 1  # Large blob for detection
    det = byotrack.SegmentationDetections(seg)
    detections_seq = [det] * T

    det_ids = torch.zeros(T, dtype=torch.int32)  # Track 1 always links to detection 0
    track_linked = byotrack.Track(0, torch.full((T, 2), 8.0), identifier=1, detection_ids=det_ids)

    # Track 2 has no detection link => draws a disk right at the same position, overwriting the linked track
    track_disk = byotrack.Track(0, torch.full((T, 2), 8.0), identifier=2)

    with pytest.warns(UserWarning, match="fully occluded"):
        save_tracks(
            [track_linked, track_disk],
            tmp_path,
            detections_sequence=detections_seq,
            as_res=True,
            default_radius=8.0,
            overwrite_detections=True,
        )


# --- save_tracks: with detections ---


def test_save_tracks_with_detections_uses_segmentation(tmp_path: pathlib.Path) -> None:
    seg = torch.zeros(*SHAPE_2D, dtype=torch.int32)
    seg[3:6, 3:6] = 1
    det = byotrack.SegmentationDetections(seg)
    detections_seq = [det] * T

    det_ids = torch.zeros(T, dtype=torch.int32)
    track = byotrack.Track(0, torch.full((T, 2), 4.0), identifier=1, detection_ids=det_ids)

    save_tracks([track], tmp_path, detections_sequence=detections_seq, as_res=True)

    raw = tifffile.imread(tmp_path / "mask0000.tif").squeeze()
    # Track 1 => label 2 in file (identifier + 1)
    assert (raw[3:6, 3:6] == 2).all()


def test_save_tracks_unlinked_track_draws_disk(tmp_path: pathlib.Path) -> None:
    seg = torch.zeros(*SHAPE_2D, dtype=torch.int32)
    seg[3:6, 3:6] = 1
    det = byotrack.SegmentationDetections(seg)
    detections_seq = [det] * T

    # Track 1 linked to detection 0
    det_ids = torch.zeros(T, dtype=torch.int32)
    track_linked = byotrack.Track(0, torch.full((T, 2), 4.0), identifier=1, detection_ids=det_ids)

    # Track 2 not linked (detection_ids all -1) => disk drawn at position (14, 14)
    track_disk = byotrack.Track(0, torch.full((T, 2), 14.0), identifier=2)

    save_tracks(
        [track_linked, track_disk], tmp_path, detections_sequence=detections_seq, as_res=True, default_radius=RADIUS
    )

    raw = tifffile.imread(tmp_path / "mask0000.tif").squeeze()
    # Label 3 = track_disk.identifier + 1 = 3 should appear around (14, 14)
    assert (raw[11:17, 11:17] > 0).any()


# --- Round-trip tests ---


def _sorted_tracks(tracks: list[byotrack.Track]) -> list[byotrack.Track]:
    return sorted(tracks, key=lambda t: t.identifier)


def test_round_trip_2d_res(tmp_path: pathlib.Path) -> None:
    tracks = _make_tracks_2d()
    path = tmp_path / "res"
    save_tracks(tracks, path, shape=SHAPE_2D, as_res=True, default_radius=RADIUS)
    loaded = _sorted_tracks(load_tracks(path))

    assert len(loaded) == 2
    for orig, got in zip(tracks, loaded, strict=True):
        assert got.identifier == orig.identifier
        assert got.start == orig.start
        assert got.parent_id == -1
        assert torch.allclose(got.points, orig.points, atol=1.5)


def test_round_trip_2d_man_track(tmp_path: pathlib.Path) -> None:
    tracks = _make_tracks_2d()
    path = tmp_path / "man"
    save_tracks(tracks, path, shape=SHAPE_2D, as_res=False, as_seg=False, default_radius=RADIUS)
    loaded = _sorted_tracks(load_tracks(path))

    assert len(loaded) == 2
    for orig, got in zip(tracks, loaded, strict=True):
        assert got.identifier == orig.identifier
        assert torch.allclose(got.points, orig.points, atol=1.5)


def test_round_trip_2d_man_seg(tmp_path: pathlib.Path) -> None:
    tracks = _make_tracks_2d()
    path = tmp_path / "man_seg"
    save_tracks(tracks, path, shape=SHAPE_2D, as_res=False, as_seg=True, default_radius=RADIUS)
    loaded = _sorted_tracks(load_tracks(path))

    assert len(loaded) == 2
    for orig, got in zip(tracks, loaded, strict=True):
        assert got.identifier == orig.identifier
        assert torch.allclose(got.points, orig.points, atol=1.5)


def test_round_trip_3d_res(tmp_path: pathlib.Path) -> None:
    tracks = _make_tracks_3d()
    path = tmp_path / "res"
    save_tracks(tracks, path, shape=SHAPE_3D, as_res=True, default_radius=RADIUS)
    loaded = _sorted_tracks(load_tracks(path))

    assert len(loaded) == 2
    for orig, got in zip(tracks, loaded, strict=True):
        assert got.identifier == orig.identifier
        assert torch.allclose(got.points, orig.points, atol=1.5)


def test_round_trip_with_detections(tmp_path: pathlib.Path) -> None:
    seg = torch.zeros(*SHAPE_2D, dtype=torch.int32)
    seg[3:6, 3:6] = 1  # detection index 0 at ~(4, 4)
    seg[13:16, 13:16] = 2  # detection index 1 at ~(14, 14)
    det = byotrack.SegmentationDetections(seg)
    detections_seq = [det] * T

    det_ids_1 = torch.zeros(T, dtype=torch.int32)  # always links to detection 0
    det_ids_2 = torch.ones(T, dtype=torch.int32)  # always links to detection 1
    tracks = [
        byotrack.Track(0, torch.full((T, 2), 3.0), identifier=1, detection_ids=det_ids_1),
        byotrack.Track(0, torch.full((T, 2), 13.0), identifier=2, detection_ids=det_ids_2),
    ]

    path = tmp_path / "res"
    save_tracks(tracks, path, detections_sequence=detections_seq, as_res=True)
    loaded = _sorted_tracks(load_tracks(path))

    assert len(loaded) == 2
    assert loaded[0].identifier == 1
    assert loaded[1].identifier == 2
    # Positions recovered from segmentation blobs (3 => 4, 13 => 14)
    assert torch.allclose(loaded[0].points, torch.full((T, 2), 4.0))
    assert torch.allclose(loaded[1].points, torch.full((T, 2), 14.0))
