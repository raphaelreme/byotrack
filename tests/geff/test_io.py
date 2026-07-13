from __future__ import annotations

from typing import TYPE_CHECKING

import geff
import geff_spec  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch

import byotrack
from byotrack.geff.io import (
    load_detections_from_geff,
    load_detections_from_zarr,
    load_tracks_from_geff,
    load_video_from_geff,
    load_video_from_zarr,
    save_detections_to_zarr,
    save_tracks_to_geff,
    save_video_to_zarr,
)

if TYPE_CHECKING:
    import pathlib

    from byotrack.api.detections import SegmentationDetections


## save_video_to_zarr


def test_save_video_to_zarr_2d(tmp_path: pathlib.Path, video_2d: np.ndarray):
    store = save_video_to_zarr(video_2d, tmp_path / "video.zarr")

    assert store.shape == video_2d.shape
    assert store.dtype == video_2d.dtype
    assert store.chunks == (1, *video_2d.shape[1:])
    assert (np.asarray(store) == video_2d).all()


def test_save_video_to_zarr_3d(tmp_path: pathlib.Path, video_3d: np.ndarray):
    store = save_video_to_zarr(video_3d, tmp_path / "video.zarr")

    assert store.shape == video_3d.shape
    assert store.dtype == video_3d.dtype
    assert (np.asarray(store) == video_3d).all()


def test_save_video_to_zarr_channel_selection(tmp_path: pathlib.Path, video_2d: np.ndarray):
    store = save_video_to_zarr(video_2d, tmp_path / "video.zarr", channel=1)

    assert store.shape == video_2d.shape[:-1]
    assert (np.asarray(store) == video_2d[..., 1]).all()


def test_save_video_to_zarr_custom_chunks(tmp_path: pathlib.Path, video_2d: np.ndarray):
    store = save_video_to_zarr(video_2d, tmp_path / "video.zarr", chunks=(2, 5, 6, 2))

    assert store.chunks == (2, 5, 6, 2)


def test_save_video_to_zarr_empty_list_raises(tmp_path: pathlib.Path):
    with pytest.raises(ValueError, match="empty"):
        save_video_to_zarr([], tmp_path / "video.zarr")


## save_detections_to_zarr


def test_save_detections_to_zarr_2d(tmp_path: pathlib.Path, seg_2d_det: SegmentationDetections):
    store = save_detections_to_zarr([seg_2d_det] * 3, tmp_path / "seg.zarr")

    assert store.shape == (3, *seg_2d_det.shape)
    assert store.dtype == np.uint16
    assert (np.asarray(store[0]) == seg_2d_det.segmentation.numpy().astype(np.uint16)).all()
    assert (np.asarray(store[1]) == store[0]).all()


def test_save_detections_to_zarr_3d(tmp_path: pathlib.Path, seg_3d_det: SegmentationDetections):
    store = save_detections_to_zarr([seg_3d_det] * 2, tmp_path / "seg.zarr")

    assert store.shape == (2, *seg_3d_det.shape)
    assert store.dtype == np.uint16
    assert (np.asarray(store[0]) == seg_3d_det.segmentation.numpy().astype(np.uint16)).all()


def test_save_detections_to_zarr_custom_chunks(tmp_path: pathlib.Path, seg_2d_det: np.ndarray):
    store = save_detections_to_zarr([seg_2d_det] * 5, tmp_path / "seg.zarr", chunks=(2, 5, 6))

    assert store.chunks == (2, 5, 6)


def test_save_detections_to_zarr_empty_raises(tmp_path: pathlib.Path):
    with pytest.raises(ValueError, match="empty"):
        save_detections_to_zarr([], tmp_path / "seg.zarr")


## save_tracks_to_geff / load_tracks_from_geff


def test_tracks_round_trip_2d(tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track]):
    path = tmp_path / "tracks.geff"
    save_tracks_to_geff(tracks_2d, path)

    loaded = sorted(load_tracks_from_geff(path), key=lambda t: t.identifier)

    assert len(loaded) == len(tracks_2d)
    for orig, got in zip(sorted(tracks_2d, key=lambda t: t.identifier), loaded, strict=True):
        assert got.identifier == orig.identifier
        assert got.start == orig.start
        assert got.dim == 2
        assert torch.allclose(got.points, orig.points)


def test_tracks_round_trip_3d(tmp_path: pathlib.Path, tracks_3d: list[byotrack.Track]):
    path = tmp_path / "tracks.geff"
    save_tracks_to_geff(tracks_3d, path)

    loaded = load_tracks_from_geff(path)

    assert len(loaded) == len(tracks_3d)
    assert loaded[0].dim == 3
    assert torch.allclose(loaded[0].points, tracks_3d[0].points)


def test_tracks_round_trip_empty(tmp_path: pathlib.Path):
    path = tmp_path / "tracks.geff"
    save_tracks_to_geff([], path)

    loaded = load_tracks_from_geff(path)

    assert loaded == []


def test_tracks_round_trip_split(tmp_path: pathlib.Path):
    parent = byotrack.Track(0, torch.tensor([[0.0, 0.0], [1.0, 1.0]]), identifier=0)
    child_a = byotrack.Track(2, torch.tensor([[2.0, 2.0]]), identifier=1, parent_id=0)
    child_b = byotrack.Track(2, torch.tensor([[3.0, 3.0]]), identifier=2, parent_id=0)

    path = tmp_path / "tracks.geff"
    save_tracks_to_geff([parent, child_a, child_b], path)

    loaded = {track.identifier: track for track in load_tracks_from_geff(path)}

    assert loaded[0].parent_id == -1
    assert loaded[1].parent_id == 0
    assert loaded[2].parent_id == 0


def test_tracks_round_trip_merge(tmp_path: pathlib.Path):
    parent_a = byotrack.Track(0, torch.tensor([[0.0, 0.0], [1.0, 1.0]]), identifier=0, merge_id=2)
    parent_b = byotrack.Track(0, torch.tensor([[5.0, 5.0], [4.0, 4.0]]), identifier=1, merge_id=2)
    child = byotrack.Track(2, torch.tensor([[2.0, 2.0]]), identifier=2)

    path = tmp_path / "tracks.geff"
    save_tracks_to_geff([parent_a, parent_b, child], path)

    loaded = {track.identifier: track for track in load_tracks_from_geff(path)}

    assert loaded[0].merge_id == 2
    assert loaded[1].merge_id == 2
    assert loaded[2].merge_id == -1


def test_save_tracks_drop_nan_round_trip_only_removes_outer_nan(tmp_path: pathlib.Path):
    inner_nan = torch.tensor([[0.0, 0.0], [torch.nan, torch.nan], [2.0, 2.0]])
    outer_nan = torch.tensor([[0.0, 0.0], [2.0, 3.0], [torch.nan, torch.nan]])
    tracks = [byotrack.Track(0, inner_nan, 0), byotrack.Track(0, outer_nan, 1)]

    keep_path = tmp_path / "keep_nan.geff"
    drop_path = tmp_path / "drop_nan.geff"
    save_tracks_to_geff(tracks, keep_path, drop_nan=False)
    save_tracks_to_geff(tracks, drop_path, drop_nan=True)

    keep_tracks = load_tracks_from_geff(keep_path)
    drop_tracks = load_tracks_from_geff(drop_path)

    assert len(tracks[0]) == len(keep_tracks[0]) == len(drop_tracks[0])
    assert len(tracks[1]) == len(keep_tracks[1]) == len(drop_tracks[1]) + 1


def test_tracks_anisotropy_sets_axis_scale(tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track]):
    path = tmp_path / "tracks.geff"
    save_tracks_to_geff(tracks_2d, path, anisotropy=(2.0, 1.5, 1.25))

    metadata = geff.GeffReader(path).metadata

    axes = {axis.name: axis for axis in metadata.axes}
    assert axes["y"].scale == pytest.approx(1.5)
    assert axes["x"].scale == pytest.approx(1.25)


def test_tracks_with_video_writes_video_without_split(
    tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track], video_2d: np.ndarray
):
    path = tmp_path / "tracks.geff"
    save_tracks_to_geff(tracks_2d, path, video=video_2d, split_channels=False)

    metadata = geff.GeffReader(path).metadata

    related_types = [obj.type for obj in metadata.related_objects]
    related_paths = [obj.path for obj in metadata.related_objects]
    assert related_types == ["image"]
    assert related_paths == ["video"]

    loaded_video = load_video_from_zarr(path / "video")
    assert loaded_video.shape == video_2d.shape
    assert (np.asarray(loaded_video) == video_2d).all()


def test_tracks_with_video_splits_channel(
    tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track], video_2d: np.ndarray
):
    path = tmp_path / "tracks.geff"
    save_tracks_to_geff(tracks_2d, path, video=video_2d, split_channels=True)

    n_channels = video_2d.shape[-1]
    metadata = geff.GeffReader(path).metadata

    related_types = [obj.type for obj in metadata.related_objects]
    related_paths = [obj.path for obj in metadata.related_objects]
    assert related_types == ["image"] * n_channels
    assert related_paths == [f"video-{c}" for c in range(n_channels)]

    for channel in range(n_channels):
        with pytest.warns(UserWarning, match="Channel dimension not found."):
            loaded_video = load_video_from_zarr(path / f"video-{channel}")

        assert loaded_video.shape[:-1] == video_2d.shape[:-1]
        assert (np.asarray(loaded_video)[..., 0] == video_2d[..., channel]).all()


def test_tracks_with_detections_writes_segmentation(
    tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track], seg_2d_det: SegmentationDetections
):
    path = tmp_path / "tracks.geff"
    save_tracks_to_geff(tracks_2d, path, detections_sequence=[seg_2d_det] * 2)

    metadata = geff.GeffReader(path).metadata

    assert len(metadata.related_objects) == 1
    assert metadata.related_objects[0].type == "labels"
    assert metadata.related_objects[0].path == "segmentation"

    loaded_dets = load_detections_from_zarr(path / "segmentation")
    assert len(loaded_dets) == 2
    assert torch.equal(loaded_dets[0].segmentation, seg_2d_det.segmentation)


## load_video_from_zarr


def test_load_video_from_zarr_round_trip(tmp_path: pathlib.Path, video_2d: np.ndarray):
    path = tmp_path / "video.zarr"
    save_video_to_zarr(video_2d, path)

    video = load_video_from_zarr(path)

    assert video.shape == video_2d.shape
    assert video.dtype == video_2d.dtype
    assert (np.asarray(video) == video_2d).all()


## load_detections_from_zarr


def test_load_detections_from_zarr_round_trip(tmp_path: pathlib.Path, seg_2d_det: SegmentationDetections):
    path = tmp_path / "seg.zarr"
    save_detections_to_zarr([seg_2d_det] * 3, path)

    detections = load_detections_from_zarr(path)

    assert len(detections) == 3
    for det in detections:
        assert torch.equal(det.segmentation, seg_2d_det.segmentation)


## load_video_from_geff


def test_load_video_from_geff_round_trip(tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track], video_2d: np.ndarray):
    path = tmp_path / "tracks.geff"
    save_tracks_to_geff(tracks_2d, path, video=video_2d, split_channels=False)

    video = load_video_from_geff(path)

    assert video.shape == video_2d.shape
    assert (np.asarray(video) == video_2d).all()


def test_load_video_from_geff_only_loads_first_channel(
    tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track], video_2d: np.ndarray
):
    # save_tracks_to_geff splits a multi-channel video into one related object per channel;
    # load_video_from_geff only loads the first one it finds (channel 0), not all of them.
    path = tmp_path / "tracks.geff"
    save_tracks_to_geff(tracks_2d, path, video=video_2d, split_channels=True)

    with pytest.warns(UserWarning, match="Channel dimension not found."):
        video = load_video_from_geff(path)

    assert video.shape[-1] == 1
    assert (np.asarray(video)[..., 0] == video_2d[..., 0]).all()
    assert not (np.asarray(video)[..., 0] == video_2d[..., 1]).all()


def test_load_video_from_geff_no_related_object_raises(tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track]):
    path = tmp_path / "tracks.geff"
    save_tracks_to_geff(tracks_2d, path)

    with pytest.raises(FileNotFoundError, match="image"):
        load_video_from_geff(path)


def test_load_video_from_geff_do_not_load_labels(
    tmp_path: pathlib.Path, video_2d: np.ndarray, tracks_2d: list[byotrack.Track]
):
    path = tmp_path / "tracks.geff"
    geff.write(
        byotrack.TrackingGraph.from_tracks(tracks_2d),
        path,
        geff_spec.GeffMetadata(
            directed=True,
            node_props_metadata={},
            edge_props_metadata={},
            related_objects=[
                geff_spec.RelatedObject(type="labels", path="segmentation"),  # Labels first
                geff_spec.RelatedObject(type="image", path="video"),
            ],
        ),
    )
    save_video_to_zarr(video_2d, path / "video")

    video = load_video_from_geff(path)
    assert video.shape == video_2d.shape
    assert (np.asarray(video) == video_2d).all()


## load_detections_from_geff


def test_load_detections_from_geff_round_trip(
    tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track], seg_2d_det: SegmentationDetections
):
    path = tmp_path / "tracks.geff"
    save_tracks_to_geff(tracks_2d, path, detections_sequence=[seg_2d_det] * 2)

    detections_sequence = load_detections_from_geff(path)

    assert len(detections_sequence) == 2
    assert torch.equal(detections_sequence[0].segmentation, seg_2d_det.segmentation)


def test_load_detections_from_geff_no_related_object_raises(tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track]):
    path = tmp_path / "tracks.geff"
    save_tracks_to_geff(tracks_2d, path)

    with pytest.raises(FileNotFoundError, match="labels"):
        load_detections_from_geff(path)


def test_load_detections_from_geff_do_not_load_image(
    tmp_path: pathlib.Path, seg_2d_det: SegmentationDetections, tracks_2d: list[byotrack.Track]
):
    path = tmp_path / "tracks.geff"
    geff.write(
        byotrack.TrackingGraph.from_tracks(tracks_2d),
        path,
        geff_spec.GeffMetadata(
            directed=True,
            node_props_metadata={},
            edge_props_metadata={},
            related_objects=[
                geff_spec.RelatedObject(type="image", path="video"),  # Image first
                geff_spec.RelatedObject(type="labels", path="segmentation"),
            ],
        ),
    )
    save_detections_to_zarr([seg_2d_det], path / "segmentation")

    detections_sequence = load_detections_from_geff(path)
    assert len(detections_sequence) == 1
    assert torch.equal(detections_sequence[0].segmentation, seg_2d_det.segmentation)
