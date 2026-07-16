from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import byotrack
from byotrack.api.tracks import _resolve_disk_radii, update_detection_ids, update_detections_from_tracks

if TYPE_CHECKING:
    import pathlib


## Construction


def test_track_2d_construction():
    points = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    track = byotrack.Track(0, points, identifier=100)
    assert track.start == 0
    assert track.identifier == 100
    assert track.dim == 2
    assert len(track) == 2


def test_track_3d_construction():
    points = torch.ones(4, 3)
    track = byotrack.Track(2, points, identifier=101)
    assert track.dim == 3
    assert len(track) == 4
    assert track.start == 2
    assert track.identifier == 101


def test_track_auto_identifier_increments():
    before = byotrack.Track._next_identifier
    track = byotrack.Track(0, torch.ones(2, 2))
    assert track.identifier == before
    assert byotrack.Track._next_identifier == before + 1


def test_track_negative_identifier_raises():
    with pytest.raises(ValueError, match="Track identifiers cannot be negative"):
        byotrack.Track(0, torch.ones(2, 2), identifier=-1)


def test_track_wrong_points_shape_raises():
    msg = "expected to be \\(T, 2\\) or \\(T, 3\\) with T>0"
    with pytest.raises(ValueError, match=msg):
        byotrack.Track(0, torch.ones(5))
    with pytest.raises(ValueError, match=msg):
        byotrack.Track(0, torch.ones(5, 1))
    with pytest.raises(ValueError, match=msg):
        byotrack.Track(0, torch.ones(5, 4))
    with pytest.raises(ValueError, match=msg):
        byotrack.Track(0, torch.zeros(0, 2))


def test_track_float64_dtype_raises():
    with pytest.raises(ValueError, match="have a torch\\.float32 dtype"):
        byotrack.Track(0, torch.ones(3, 2, dtype=torch.float64))


def test_track_default_detection_ids_minus_one():
    track = byotrack.Track(0, torch.ones(3, 2))
    assert (track.detection_ids == -1).all()
    assert track.detection_ids.dtype == torch.int32


def test_track_explicit_detection_ids():
    det_ids = torch.tensor([0, 1, 2], dtype=torch.int32)
    track = byotrack.Track(0, torch.ones(3, 2), detection_ids=det_ids)
    assert torch.equal(track.detection_ids, det_ids)


def test_track_wrong_detection_ids_shape_raises():
    with pytest.raises(ValueError, match="expected to match the size \\(T,\\)"):
        byotrack.Track(0, torch.ones(3, 2), detection_ids=torch.zeros(2, dtype=torch.int32))

    with pytest.raises(ValueError, match="expected to match the size \\(T,\\)"):
        byotrack.Track(0, torch.ones(3, 2), detection_ids=torch.zeros(2, 1, dtype=torch.int32))


def test_track_wrong_detection_ids_dtype_raises():
    with pytest.raises(ValueError, match="expected to have a torch\\.int32 dtype"):
        byotrack.Track(0, torch.ones(3, 2), detection_ids=torch.zeros(3, dtype=torch.int64))


def test_track_merge_id_stored():
    track = byotrack.Track(0, torch.ones(3, 2), merge_id=999)
    assert track.merge_id == 999


def test_track_parent_id_stored():
    track = byotrack.Track(0, torch.ones(3, 2), parent_id=888)
    assert track.parent_id == 888


## Length and getitem


def test_track_len_equals_t():
    track = byotrack.Track(0, torch.ones(5, 2))
    assert len(track) == 5


def test_track_len_3d():
    track = byotrack.Track(5, torch.ones(7, 3))
    assert len(track) == 7


def test_track_getitem_inside_range():
    points = torch.rand(5, 3)
    track = byotrack.Track(10, points)

    # Start
    result = track[10]
    assert torch.allclose(result, points[0])

    # Middle
    result = track[12]
    assert torch.allclose(result, points[2])

    # End
    result = track[14]
    assert torch.allclose(result, points[-1])


def test_track_getitem_out_range_returns_nan():
    track = byotrack.Track(5, torch.ones(3, 2))
    result = track[4]
    assert result.shape == (2,)
    assert torch.isnan(result).all()

    result = track[8]  # last valid frame is 7
    assert result.shape == (2,)
    assert torch.isnan(result).all()


def test_track_getitem_shape_2d():
    track = byotrack.Track(0, torch.ones(3, 2))
    assert track[0].shape == (2,)


def test_track_getitem_shape_3d():
    track = byotrack.Track(0, torch.ones(3, 3))
    assert track[0].shape == (3,)


## overlaps_with


def test_track_overlaps_same_frames():
    track_1 = byotrack.Track(0, torch.ones(3, 2))
    track_2 = byotrack.Track(0, torch.ones(3, 2))
    assert track_1.overlaps_with(track_2)
    assert track_2.overlaps_with(track_1)


def test_track_no_overlap_adjacent():
    track_1 = byotrack.Track(0, torch.ones(3, 2))  # frames 0,1,2
    track_2 = byotrack.Track(3, torch.ones(2, 2))  # frames 3,4
    assert not track_1.overlaps_with(track_2)
    assert not track_2.overlaps_with(track_1)


def test_track_no_overlap_with_gap():
    track_1 = byotrack.Track(0, torch.ones(2, 3))  # frames 0,1
    track_2 = byotrack.Track(5, torch.ones(2, 3))  # frames 5,6
    assert not track_1.overlaps_with(track_2)
    assert not track_2.overlaps_with(track_1)


def test_track_partial_overlap():
    track_1 = byotrack.Track(0, torch.ones(5, 2))  # frames 0-4
    track_2 = byotrack.Track(3, torch.ones(5, 2))  # frames 3-7
    assert track_1.overlaps_with(track_2)
    assert track_2.overlaps_with(track_1)


def test_track_overlaps_single_shared_frame():
    track_1 = byotrack.Track(0, torch.ones(3, 2))  # frames 0,1,2
    track_2 = byotrack.Track(2, torch.ones(3, 2))  # frames 2,3,4
    assert track_1.overlaps_with(track_2)
    assert track_2.overlaps_with(track_1)


def test_track_overlaps_tolerance_extends_overlap():
    track_1 = byotrack.Track(0, torch.ones(3, 2))  # frames 0-2
    track_2 = byotrack.Track(3, torch.ones(3, 2))  # frames 3-5
    assert not track_1.overlaps_with(track_2, tolerance=0)
    # Negative tolerance allows gap
    assert track_1.overlaps_with(track_2, tolerance=-1)


def test_track_overlaps_tolerance_reduces_overlap():
    track_1 = byotrack.Track(0, torch.ones(4, 2))  # frames 0-3
    track_2 = byotrack.Track(2, torch.ones(4, 2))  # frames 2-5, overlap of 2 frames
    assert track_1.overlaps_with(track_2, tolerance=0)
    assert track_1.overlaps_with(track_2, tolerance=1)
    assert not track_1.overlaps_with(track_2, tolerance=2)


## Tensorize


def test_track_tensorize_single():
    track = byotrack.Track(0, torch.rand(3, 2))
    tensor = byotrack.Track.tensorize([track])

    assert tensor.shape == (3, 1, 2)
    assert torch.allclose(tensor[:, 0, :], track.points)


def test_track_tensorize_multiple() -> None:
    tracks = [
        byotrack.Track(0, torch.rand(3, 3)),
        byotrack.Track(2, torch.rand(7, 3)),
        byotrack.Track(3, torch.rand(3, 3)),
    ]
    tensor = byotrack.Track.tensorize(tracks)

    assert tensor.shape == (9, 3, 3)

    # NaN before 2nd track starts
    assert torch.isnan(tensor[0:2, 1]).all()

    # NaN after 3rd track ends
    assert torch.isnan(tensor[6:, 2]).all()

    # First track is in first line
    assert torch.allclose(tensor[:3, 0], tracks[0].points)

    expected = [len(track) for track in tracks]
    num_elements = (~torch.isnan(tensor).any(dim=-1)).sum(dim=0)

    assert num_elements.tolist() == expected


def test_track_tensorize_inferred_frame_range() -> None:
    tracks = [
        byotrack.Track(7, torch.rand(3, 2)),  # 7, 8, 9
        byotrack.Track(2, torch.rand(7, 2)),  # 2, ..., 8
        byotrack.Track(3, torch.rand(3, 2)),  # 3, 4, 5
    ]
    tensor = byotrack.Track.tensorize(tracks)

    assert tensor.shape == (8, 3, 2)

    print(byotrack.Track.tensorize(tracks, (2, 10)).shape)

    # The range can also be explicitly given.
    assert torch.allclose(tensor, byotrack.Track.tensorize(tracks, (2, 10)), equal_nan=True)


def test_track_tensorize_explicit_extended_frame_range():
    tracks = [
        byotrack.Track(7, torch.rand(3, 2)),  # 7, 8, 9
        byotrack.Track(2, torch.rand(7, 2)),  # 2, ..., 8
        byotrack.Track(3, torch.rand(3, 2)),  # 3, 4, 5
    ]
    tensor = byotrack.Track.tensorize(tracks, frame_range=(0, 15))

    assert tensor.shape == (15, 3, 2)

    # Tracks start on frame 2
    assert torch.isnan(tensor[:2]).all()
    assert not torch.isnan(tensor[3]).all()

    # Tracks end on frame 9
    assert torch.isnan(tensor[10:]).all()
    assert not torch.isnan(tensor[9]).all()

    # Tracks should be fully included in the range
    expected = [len(track) for track in tracks]
    num_elements = (~torch.isnan(tensor).any(dim=-1)).sum(dim=0)

    assert num_elements.tolist() == expected


def test_track_tensorize_explicit_restricted_frame_range():
    tracks = [
        byotrack.Track(7, torch.rand(3, 2)),  # 7, 8, 9
        byotrack.Track(2, torch.rand(7, 2)),  # 2, ..., 8
        byotrack.Track(3, torch.rand(3, 2)),  # 3, 4, 5
    ]
    tensor = byotrack.Track.tensorize(tracks, frame_range=(4, 6))

    assert tensor.shape == (2, 3, 2)
    assert not torch.isnan(tensor[:, 1:]).any()  # 1 and 2 defined in frames 4 and 5
    assert torch.isnan(tensor[:, 0]).all()  # 0 starts on frame 7

    assert torch.allclose(tensor[:, 2], tracks[2].points[1:3])


def test_track_tensorize_empty_raises():
    with pytest.raises(ValueError, match="Cannot tensorize an empty collection of Tracks"):
        byotrack.Track.tensorize([])


def test_track_tensorize_different_dim_raises():
    tracks = [
        byotrack.Track(7, torch.rand(3, 2)),
        byotrack.Track(2, torch.rand(7, 3)),  # 3D
        byotrack.Track(3, torch.rand(3, 2)),
    ]
    with pytest.raises(ValueError, match="share the same spatial dimension"):
        byotrack.Track.tensorize(tracks)


## tensorize_det_ids


def test_track_tensorize_det_ids_single():
    det_ids = torch.randint(0, 15, (3,), dtype=torch.int32)
    track = byotrack.Track(0, torch.ones(3, 2), detection_ids=det_ids)
    ids_tensor = byotrack.Track._tensorize_det_ids([track])
    assert ids_tensor.shape == (3, 1)
    assert ids_tensor.dtype == torch.int32
    assert torch.equal(ids_tensor[:, 0], det_ids)


def test_track_tensorize_det_ids_multiple():
    tracks = [
        byotrack.Track(7, torch.rand(2, 3), detection_ids=torch.randint(0, 15, (2,), dtype=torch.int32)),
        byotrack.Track(2, torch.rand(7, 3), detection_ids=torch.randint(0, 15, (7,), dtype=torch.int32)),
        byotrack.Track(3, torch.rand(3, 3), detection_ids=torch.randint(0, 15, (3,), dtype=torch.int32)),
    ]
    ids_tensor = byotrack.Track._tensorize_det_ids(tracks)
    assert ids_tensor.shape == (7, 3)

    # -1 before 1st track starts
    assert (ids_tensor[:5, 0] == -1).all()

    # -1 after 3rd track ends
    assert (ids_tensor[6:, 2] == -1).all()

    # Second track is in the second line
    assert torch.allclose(ids_tensor[:, 1], tracks[1].detection_ids)

    expected = [len(track) for track in tracks]
    num_elements = (ids_tensor != -1).sum(dim=0)

    assert num_elements.tolist() == expected


def test_track_tensorize_det_ids_empty_raises():
    with pytest.raises(ValueError, match="Cannot tensorize an empty collection of Tracks"):
        byotrack.Track._tensorize_det_ids([])


## Check tracks


def test_check_tracks_valid_independent() -> None:
    tracks = [
        byotrack.Track(7, torch.rand(3, 2)),
        byotrack.Track(2, torch.rand(7, 2)),
        byotrack.Track(3, torch.rand(3, 2)),
    ]
    byotrack.Track.check_tracks(tracks, warn=False)


def test_check_tracks_detect_shared_identifiers() -> None:
    tracks = [
        byotrack.Track(7, torch.rand(3, 2), identifier=5),
        byotrack.Track(2, torch.rand(7, 2), identifier=7),
        byotrack.Track(3, torch.rand(3, 2), identifier=5),
    ]
    with pytest.raises(ValueError, match="Invalid tracks"), pytest.warns(UserWarning, match="duplicated identifiers"):
        byotrack.Track.check_tracks(tracks, warn=False)

    with pytest.warns(UserWarning, match="duplicated identifiers"):
        byotrack.Track.check_tracks(tracks, warn=True)


def test_check_tracks_valid_merge() -> None:
    tracks = [
        byotrack.Track(0, torch.rand(10, 2), identifier=5, merge_id=12),
        byotrack.Track(5, torch.rand(5, 2), identifier=7, merge_id=12),
        byotrack.Track(10, torch.rand(3, 2), identifier=12),
    ]
    byotrack.Track.check_tracks(tracks, warn=False)


def test_check_tracks_invalid_merge() -> None:
    tracks = [
        byotrack.Track(0, torch.rand(10, 2), identifier=5, merge_id=12),
        byotrack.Track(5, torch.rand(5, 2), identifier=7, merge_id=12),
        byotrack.Track(8, torch.rand(3, 2), identifier=12),
    ]

    with (
        pytest.raises(ValueError, match="Invalid tracks"),
        pytest.warns(UserWarning, match="merges into track 12\\. But"),
    ):
        byotrack.Track.check_tracks(tracks, warn=False)


def test_check_tracks_merge_only_warns_if_gap() -> None:
    tracks = [
        byotrack.Track(0, torch.rand(10, 2), identifier=5, merge_id=12),
        byotrack.Track(5, torch.rand(5, 2), identifier=7, merge_id=12),
        byotrack.Track(12, torch.rand(3, 2), identifier=12),
    ]

    with pytest.warns(UserWarning, match="merges into track 12\\. But"):
        byotrack.Track.check_tracks(tracks, warn=False)


def test_check_tracks_merge_warns_if_not_couple() -> None:
    tracks = [
        byotrack.Track(0, torch.rand(10, 2), identifier=5, merge_id=12),
        byotrack.Track(10, torch.rand(3, 2), identifier=12),
    ]

    with pytest.warns(UserWarning, match="Track 12 is the results of 1 ! = 2 tracks"):
        byotrack.Track.check_tracks(tracks, warn=False)

    tracks = [
        byotrack.Track(0, torch.rand(10, 2), identifier=5, merge_id=12),
        byotrack.Track(5, torch.rand(5, 2), identifier=7, merge_id=12),
        byotrack.Track(3, torch.rand(7, 2), identifier=6, merge_id=12),
        byotrack.Track(10, torch.rand(3, 2), identifier=12),
    ]

    with pytest.warns(UserWarning, match="Track 12 is the results of 3 ! = 2 tracks"):
        byotrack.Track.check_tracks(tracks, warn=False)


def test_check_tracks_valid_split() -> None:
    tracks = [
        byotrack.Track(3, torch.rand(5, 3), identifier=5),
        byotrack.Track(8, torch.rand(5, 3), identifier=7, parent_id=5),
        byotrack.Track(8, torch.rand(7, 3), identifier=12, parent_id=5),
    ]
    byotrack.Track.check_tracks(tracks, warn=False)


def test_check_tracks_invalid_split() -> None:
    tracks = [
        byotrack.Track(3, torch.rand(5, 3), identifier=5),
        byotrack.Track(7, torch.rand(5, 3), identifier=7, parent_id=5),
        byotrack.Track(8, torch.rand(7, 3), identifier=12, parent_id=5),
    ]

    with (
        pytest.raises(ValueError, match="Invalid tracks"),
        pytest.warns(UserWarning, match="splits into track 7\\. But"),
    ):
        byotrack.Track.check_tracks(tracks, warn=False)


def test_check_tracks_split_only_warns_if_gap() -> None:
    tracks = [
        byotrack.Track(3, torch.rand(5, 3), identifier=5),
        byotrack.Track(9, torch.rand(5, 3), identifier=7, parent_id=5),
        byotrack.Track(8, torch.rand(7, 3), identifier=12, parent_id=5),
    ]

    with pytest.warns(UserWarning, match="splits into track 7\\. But"):
        byotrack.Track.check_tracks(tracks, warn=False)


def test_check_tracks_split_warns_if_not_couple() -> None:
    tracks = [
        byotrack.Track(3, torch.rand(5, 3), identifier=5),
        byotrack.Track(8, torch.rand(5, 3), identifier=7, parent_id=5),
    ]

    with pytest.warns(UserWarning, match="5 splits into 1 ! = 2 tracks"):
        byotrack.Track.check_tracks(tracks, warn=False)

    tracks = [
        byotrack.Track(3, torch.rand(5, 3), identifier=5),
        byotrack.Track(8, torch.rand(5, 3), identifier=7, parent_id=5),
        byotrack.Track(8, torch.rand(3, 3), identifier=9, parent_id=5),
        byotrack.Track(8, torch.rand(7, 3), identifier=12, parent_id=5),
    ]

    with pytest.warns(UserWarning, match="5 splits into 3 ! = 2 tracks"):
        byotrack.Track.check_tracks(tracks, warn=False)


def test_check_valid_split_and_merge() -> None:
    tracks = [
        byotrack.Track(3, torch.rand(5, 3), identifier=5),
        byotrack.Track(8, torch.rand(7, 3), identifier=7, parent_id=5, merge_id=1),
        byotrack.Track(8, torch.rand(7, 3), identifier=12, parent_id=5, merge_id=1),
        byotrack.Track(15, torch.rand(3, 3), identifier=1),
        byotrack.Track(0, torch.rand(15, 3), identifier=0),  # Indep track
    ]
    byotrack.Track.check_tracks(tracks, warn=False)


## Save & Load


def test_track_save_load_single(tmp_path: pathlib.Path) -> None:
    points = torch.rand(3, 2)
    track = byotrack.Track(0, points, identifier=800)
    path = tmp_path / "tracks.pt"
    byotrack.Track.save([track], path)
    loaded = byotrack.Track.load(path)
    assert len(loaded) == 1
    assert loaded[0].identifier == 800
    assert torch.allclose(loaded[0].points, points)
    assert loaded[0].start == 0


def test_track_save_load_single_with_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    points = torch.rand(5, 3)
    det_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    track = byotrack.Track(0, points, identifier=800, detection_ids=det_ids, merge_id=700, parent_id=700)

    # Let's patch check_tracks as clearly id 700 does not exists.
    def _pass(*args, **kwargs):
        pass

    monkeypatch.setattr(byotrack.Track, "check_tracks", _pass)

    byotrack.Track.save([track], tmp_path / "tracks.pt")
    loaded = byotrack.Track.load(tmp_path / "tracks.pt")

    assert len(loaded) == 1
    assert loaded[0].identifier == 800
    assert torch.allclose(loaded[0].points, points)
    assert loaded[0].start == 0
    assert torch.allclose(loaded[0].detection_ids, det_ids)
    assert loaded[0].merge_id == track.merge_id
    assert loaded[0].parent_id == track.parent_id


def test_track_save_load_preserves_start_with_offset(tmp_path: pathlib.Path) -> None:
    track_1 = byotrack.Track(10, torch.ones(3, 2))
    track_2 = byotrack.Track(20, torch.ones(7, 2))
    path = tmp_path / "tracks.pt"
    byotrack.Track.save([track_1, track_2], path)
    loaded = byotrack.Track.load(path)

    assert loaded[0].start == 10
    assert loaded[1].start == 20


def test_track_save_empty_collection_raises(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "tracks.pt"
    with pytest.raises(ValueError, match="No tracks to save"):
        byotrack.Track.save([], path)


## Reverse


def test_track_reverse_points_reversed_single():
    points = torch.rand(5, 2)
    track = byotrack.Track(0, points)

    reversed_tracks = byotrack.Track.reverse([track])

    assert len(reversed_tracks) == 1
    assert torch.allclose(reversed_tracks[0].points, torch.flip(points, (0,)))
    assert reversed_tracks[0].start == 0  # End of the sequence


def test_track_reverse_points_reversed_multiple():
    tracks = [
        byotrack.Track(7, torch.rand(3, 3)),  # 7, 8, 9
        byotrack.Track(2, torch.rand(7, 3)),  # 2, ..., 8
        byotrack.Track(3, torch.rand(3, 3)),  # 3, 4, 5
    ]
    reversed_tracks = byotrack.Track.reverse(tracks)

    for track, rtrack in zip(tracks, reversed_tracks, strict=True):
        assert rtrack.identifier == track.identifier
        assert torch.allclose(rtrack.points, track.points.flip((0,)))
        assert rtrack.start == 9 - track.start - len(track) + 1


def test_track_reverse_start_time_adjusted():
    track = byotrack.Track(0, torch.ones(3, 2))  # 0, 1, 2
    reversed_tracks = byotrack.Track.reverse([track], video_length=10)
    # reversed start = video_length - (start + len) = 10 - 3 = 7
    assert reversed_tracks[0].start == 7


def test_track_reverse_merge_becomes_parent():
    track = byotrack.Track(0, torch.ones(3, 2), merge_id=666)
    reversed_tracks = byotrack.Track.reverse([track])
    assert reversed_tracks[0].parent_id == 666
    assert reversed_tracks[0].merge_id == -1


def test_track_reverse_parent_becomes_merge():
    track = byotrack.Track(0, torch.ones(3, 2), parent_id=777)
    reversed_tracks = byotrack.Track.reverse([track])
    assert reversed_tracks[0].merge_id == 777
    assert reversed_tracks[0].parent_id == -1


def test_track_reverse_detection_ids_reversed():
    det_ids = torch.tensor([0, 1, 2], dtype=torch.int32)
    track = byotrack.Track(0, torch.ones(3, 2), detection_ids=det_ids)
    reversed_tracks = byotrack.Track.reverse([track])
    expected = torch.tensor([2, 1, 0], dtype=torch.int32)
    assert torch.equal(reversed_tracks[0].detection_ids, expected)


# update_detection_ids


def test_update_detection_ids_basic():
    track = byotrack.Track(0, torch.rand(2, 2))
    detections_sequence = [byotrack.PointDetections(track.points[0:1]), byotrack.PointDetections(track.points[1:2])]

    assert (track.detection_ids == -1).all()

    update_detection_ids([track], detections_sequence, use_segmentation=False)

    assert (track.detection_ids == 0).all()


def test_update_detection_ids_empty_pass():
    update_detection_ids([], [])


def test_update_detection_ids_with_seg():
    track = byotrack.Track(0, torch.tensor([[1.2, 1.2], [3.2, 3.2]]))
    detections_sequence = [byotrack.PointDetections(track.points[0:1]), byotrack.PointDetections(track.points[1:2])]

    assert (track.detection_ids == -1).all()

    update_detection_ids([track], detections_sequence, use_segmentation=True)

    # Seg will not match perfectly => will not set det_ids
    assert (track.detection_ids == -1).all()

    update_detection_ids([track], detections_sequence, threshold=1.0, use_segmentation=True)

    assert (track.detection_ids == 0).all()


def test_update_detection_ids_with_empty_seg():
    track = byotrack.Track(0, torch.rand(2, 2))
    detections_sequence = [byotrack.PointDetections(track.points[:0]), byotrack.PointDetections(track.points[:0])]

    assert (track.detection_ids == -1).all()

    update_detection_ids([track], detections_sequence, use_segmentation=False)

    # No seg to update
    assert (track.detection_ids == -1).all()


def test_update_detection_ids_with_larger_sequence():
    track = byotrack.Track(1, torch.rand(2, 2))
    detections_sequence = [
        byotrack.PointDetections(track.points[:1]),
        byotrack.PointDetections(track.points[:1]),
        byotrack.PointDetections(track.points[1:2]),
        byotrack.PointDetections(track.points[:1]),
    ]

    assert (track.detection_ids == -1).all()

    update_detection_ids([track], detections_sequence, use_segmentation=False)

    assert (track.detection_ids == 0).all()


def test_update_detection_ids_with_nan_in_track():
    track = byotrack.Track(0, torch.rand(2, 2))
    detections_sequence = [
        byotrack.PointDetections(track.points[:1]),
        byotrack.PointDetections(track.points[1:2]),
    ]

    track.points[0] = torch.nan

    assert (track.detection_ids == -1).all()

    update_detection_ids([track], detections_sequence, use_segmentation=False)

    assert track.detection_ids[0] == -1
    assert track.detection_ids[1] == 0


# _resolve_disk_radii


def test_resolve_disk_radii_scalar() -> None:
    radii = _resolve_disk_radii(5.0, n_tracks=3, n_frames=2, dim=2, anisotropy=(1.0, 1.0, 1.0))
    assert radii.shape == (2, 3, 2)
    assert torch.allclose(radii, torch.full((2, 3, 2), 5.0))


def test_resolve_disk_radii_anisotropy_2d_uses_last_two_axes() -> None:
    # anisotropy[-2:] = (ani_y, ani_x) = (2.0, 4.0); depth (first) factor is ignored in 2D
    radii = _resolve_disk_radii(5.0, n_tracks=1, n_frames=1, dim=2, anisotropy=(1.0, 2.0, 4.0))
    assert torch.allclose(radii[0, 0], torch.tensor([2.5, 1.25]))


def test_resolve_disk_radii_anisotropy_3d_scales_depth_axis() -> None:
    # The depth (k) axis is the FIRST one (positions are ([k, ]i, j)), and must be scaled by ani_z
    radii = _resolve_disk_radii(4.0, n_tracks=1, n_frames=1, dim=3, anisotropy=(2.0, 1.0, 1.0))
    assert torch.allclose(radii[0, 0], torch.tensor([2.0, 4.0, 4.0]))


def test_resolve_disk_radii_per_track_tensor() -> None:
    radius = torch.tensor([[1.0], [5.0]])  # Per-track radius, shape (N, 1)
    radii = _resolve_disk_radii(radius, n_tracks=2, n_frames=3, dim=2, anisotropy=(1.0, 1.0, 1.0))
    assert torch.allclose(radii[:, 0], torch.full((3, 2), 1.0))
    assert torch.allclose(radii[:, 1], torch.full((3, 2), 5.0))


# update_detections_from_tracks


def test_update_detections_from_tracks_empty_tracks_passthrough() -> None:
    det = byotrack.PointDetections(torch.rand(2, 2))
    updated = update_detections_from_tracks([det], [])
    assert updated == [det]


def test_update_detections_from_tracks_empty_sequence_passthrough() -> None:
    updated = update_detections_from_tracks([], [byotrack.Track(0, torch.randn((3, 2)), identifier=0)])
    assert updated == []


def test_update_detections_from_tracks_relabels_matched() -> None:
    seg = torch.zeros(20, 20, dtype=torch.int32)
    seg[3:6, 3:6] = 1
    det = byotrack.SegmentationDetections(seg)

    det_ids = torch.zeros(1, dtype=torch.int32)
    track = byotrack.Track(0, torch.tensor([[4.0, 4.0]]), identifier=7, detection_ids=det_ids)

    updated = update_detections_from_tracks([det], [track])

    assert updated[0].labels.tolist() == [7]
    assert updated[0].labeled_segmentation[4, 4] == 8


def test_update_detections_from_tracks_drops_false_positives() -> None:
    seg = torch.zeros(20, 20, dtype=torch.int32)
    seg[3:6, 3:6] = 1
    seg[10:13, 10:13] = 2
    det = byotrack.SegmentationDetections(seg)

    det_ids = torch.zeros(1, dtype=torch.int32)
    track = byotrack.Track(0, torch.tensor([[4.0, 4.0]]), identifier=7, detection_ids=det_ids)

    updated = update_detections_from_tracks([det], [track])

    assert updated[0].length == 1  # The unmatched blob is dropped
    assert updated[0].labels.tolist() == [7]
    assert updated[0].labeled_segmentation[11, 11] == 0


def test_update_detections_from_tracks_keeps_false_positives() -> None:
    seg = torch.zeros(20, 20, dtype=torch.int32)
    seg[3:6, 3:6] = 1
    seg[10:13, 10:13] = 2
    det = byotrack.SegmentationDetections(seg)

    det_ids = torch.zeros(1, dtype=torch.int32)
    track = byotrack.Track(0, torch.tensor([[4.0, 4.0]]), identifier=7, detection_ids=det_ids)

    updated = update_detections_from_tracks([det], [track], drop_false_positives=False)

    assert updated[0].length == 2
    assert set(updated[0].labels.tolist()) == {7, 1}  # Unmatched detection (index 1) keeps its original label
    assert updated[0].labeled_segmentation[11, 11] == 2


def test_update_detections_from_tracks_draws_false_negatives() -> None:
    seg = torch.zeros(20, 20, dtype=torch.int32)
    seg[3:6, 3:6] = 1
    det = byotrack.SegmentationDetections(seg)

    det_ids = torch.zeros(1, dtype=torch.int32)
    track_linked = byotrack.Track(0, torch.tensor([[4.0, 4.0]]), identifier=1, detection_ids=det_ids)
    track_missing = byotrack.Track(0, torch.tensor([[14.0, 14.0]]), identifier=2)  # No matching detection

    updated = update_detections_from_tracks([det], [track_linked, track_missing], radius=2.0)

    assert set(updated[0].labels.tolist()) == {1, 2}
    assert updated[0].labeled_segmentation[14, 14] == 3


def test_update_detections_from_tracks_no_false_negatives_when_disabled() -> None:
    seg = torch.zeros(20, 20, dtype=torch.int32)
    seg[3:6, 3:6] = 1
    det = byotrack.SegmentationDetections(seg)

    det_ids = torch.zeros(1, dtype=torch.int32)
    track_linked = byotrack.Track(0, torch.tensor([[4.0, 4.0]]), identifier=1, detection_ids=det_ids)
    track_missing = byotrack.Track(0, torch.tensor([[14.0, 14.0]]), identifier=2)

    updated = update_detections_from_tracks([det], [track_linked, track_missing], draw_false_negatives=False)

    assert updated[0].labels.tolist() == [1]
    assert updated[0].labeled_segmentation[14, 14] == 0


def test_update_detections_from_tracks_per_track_radius() -> None:
    seg = torch.zeros(20, 20, dtype=torch.int32)
    det = byotrack.SegmentationDetections(seg)

    track_small = byotrack.Track(0, torch.tensor([[4.0, 4.0]]), identifier=1)
    track_large = byotrack.Track(0, torch.tensor([[14.0, 14.0]]), identifier=2)
    radius = torch.tensor([[1.0], [4.0]])  # Per-track radius, shape (N, 1)

    updated = update_detections_from_tracks([det], [track_large, track_small], radius=radius)

    small_mass = updated[0].mass[0]  # Labels are sorted so track_small should be first
    large_mass = updated[0].mass[1]
    assert large_mass > small_mass


def test_update_detections_from_tracks_complete_sequence() -> None:
    segs = [torch.zeros(20, 20, dtype=torch.int32) for _ in range(3)]
    segs[0][3:6, 3:6] = 1
    segs[1][2:5, 2:5] = 1
    segs[1][11:14, 11:14] = 2
    segs[2][10:13, 10:13] = 1

    tracks = [
        byotrack.Track(
            0,
            torch.tensor([[4.0, 4.0], [3.0, 3.0]]),
            identifier=1,
            detection_ids=torch.tensor([0, 0], dtype=torch.int32),
        ),
        byotrack.Track(
            1,
            torch.tensor([[12.0, 12.0], [11.0, 11.0]]),
            identifier=0,
            detection_ids=torch.tensor([1, 0], dtype=torch.int32),
        ),
    ]

    updated = update_detections_from_tracks([byotrack.SegmentationDetections(seg) for seg in segs], tracks)

    assert updated[0].labels.tolist() == [1]
    assert set(updated[1].labels.tolist()) == {0, 1}
    assert updated[2].labels.tolist() == [0]

    assert updated[0].labeled_segmentation[4, 4] == 2
    assert updated[1].labeled_segmentation[3, 3] == 2
    assert updated[1].labeled_segmentation[12, 12] == 1
    assert updated[2].labeled_segmentation[11, 11] == 1


def test_update_detections_from_tracks_warns_on_nan_gap() -> None:
    points = torch.tensor([[4.0, 4.0], [float("nan"), float("nan")], [4.0, 4.0]])
    track = byotrack.Track(0, points, identifier=1)
    det = byotrack.PointDetections(torch.empty((0, 2)))

    with pytest.warns(UserWarning, match="undefined position"):
        update_detections_from_tracks([det, det, det], [track])


def test_update_detections_from_tracks_warns_when_disk_not_found() -> None:
    seg = torch.zeros(20, 20, dtype=torch.int32)
    det = byotrack.SegmentationDetections(seg)
    track = byotrack.Track(0, torch.tensor([[1000.0, 1000.0]]), identifier=1)  # Outside the frame

    with pytest.warns(UserWarning, match="could not be represented"):
        update_detections_from_tracks([det], [track], radius=2.0)
