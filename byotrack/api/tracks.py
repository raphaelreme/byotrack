from __future__ import annotations

import os
from typing import cast, Collection, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import torch

import byotrack  # pylint: disable=cyclic-import

from .detector.detections import _position_from_segmentation


def _check_points(points: torch.Tensor) -> None:
    """Check points validity (type, shape)"""
    assert len(points.shape) == 2
    assert points.shape[0] > 0
    assert points.shape[1] in (2, 3)  # Only 2d and 3d supported
    assert points.dtype is torch.float32


def _check_detection_ids(detection_ids: torch.Tensor, size: int) -> None:
    assert detection_ids.shape == (size,)
    assert detection_ids.dtype is torch.int32


def _check_tracks(tracks: Collection[Track]):
    """Check consistency of a Collection of tracks. See `Track.check_tracks`"""
    id_to_track = {track.identifier: track for track in tracks}
    merge_count: Dict[int, int] = {}
    parent_count: Dict[int, int] = {}

    assert len(id_to_track) == len(tracks), "Found duplicated identifiers"

    for track in tracks:
        if track.parent_id != -1:
            parent = id_to_track[track.parent_id]
            end = parent.start + len(parent)
            assert track.start == end, (
                f"Track {parent.identifier} (Last frame: {end - 1} splits into track {track.identifier}. "
                f"But track {track.identifier} starts at {track.start} != {end}."
            )

            parent_count[track.parent_id] = parent_count.get(track.parent_id, 0) + 1
        if track.merge_id != -1:
            child = id_to_track[track.merge_id]
            end = track.start + len(track)
            assert end == child.start, (
                f"Track {track.identifier} (Last frame: {end - 1}) merges into track {child.identifier}. "
                f"But track {child.identifier} starts at {child.start} != {end}."
            )
            merge_count[track.merge_id] = merge_count.get(track.merge_id, 0) + 1

    for merge_id, count in merge_count.items():
        assert count == 2, f"Track {merge_id} is the results of {count} ! = 2 tracks."

    for parent_id, count in parent_count.items():
        assert count == 2, f"{parent_id} splits into {count} ! = 2 tracks."


class Track:
    """Track for a given particle

    A track is defined by an (non-unique) identifier, a starting frame and a succession of positions.
    In a detect-then-track context, a track can optionally contains the detection identifiers
    for each time frame (-1 if non-linked to any particular detection at this time frame)

    Attributes:
        identifier (int): Identifier of the track (non-unique)
        start (int): Starting frame of the track
        points (torch.Tensor): Positions (i, j) of the particle (from starting frame to ending frame)
            Shape: (T, dim), dtype: float32
        detection_ids (torch.Tensor): Detection id for each time frame (-1 if unknown or non-linked
            to a particular detection at this time frame)
            Shape: (T,), dtype: int32
        merge_id (int): Optional identifier to the merged track. This allows to handle 2-way merges
            (such as cell divisions in a reversed temporal order). The target track
            should start on the frame following the end of this track. This should not be used
            to do tracklet stitching (See `DistStitcher` for such use cases)
            Default: -1 (Merged to no one)
        parent_id (int) Optional identifier to a parent track. This allows to handle 2-way splits (such as
            cell divisions). The parent track should end one frame before the start of this track.
            This should not be used to do tracklet stitching (See `DistStitcher` for such use cases)
            Default: -1

    """

    _next_identifier = 0

    def __init__(
        self,
        start: int,
        points: torch.Tensor,
        identifier: Optional[int] = None,
        detection_ids: Optional[torch.Tensor] = None,
        *,
        merge_id: int = -1,
        parent_id: int = -1,
    ) -> None:
        if identifier is None:
            self.identifier = Track._next_identifier
            Track._next_identifier += 1
        else:
            assert identifier >= 0, "Track identifiers cannot be negative"
            self.identifier = identifier

        self.merge_id = merge_id
        self.parent_id = parent_id

        if detection_ids is None:  # All are unknown (-1)
            detection_ids = torch.full((len(points),), -1, dtype=torch.int32)

        _check_points(points)
        _check_detection_ids(detection_ids, len(points))

        self.start = start
        self.points = points
        self.detection_ids = detection_ids

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, frame_id: int) -> torch.Tensor:
        """Return the position of tracked particle for the given frame_id

        Args:
            frame_id (int): id of the frame in the video

        Returns:
            torch.Tensor: Position ([k, ]i, j) on the given frame (NaN if unknown)
                Shape: (dim, ), dtype: float32

        """
        if self.start <= frame_id < self.start + len(self.points):
            return self.points[frame_id - self.start]

        return torch.full((self.points.shape[1],), torch.nan)

    def overlaps_with(self, other: Track, tolerance=0) -> bool:
        """Test if this track overlaps with another one in time.

        Args:
            other (Track): The other track
            tolerance (int): Time tolerance before detecting an overlap.
                Default: 0 (no tolerance)

        """
        return self.start < other.start + len(other) - tolerance and self.start + len(self) - tolerance > other.start

    @staticmethod
    def tensorize(tracks: Collection[Track], frame_range: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Convert a collection of tracks into a tensor on a given frame_range

        Useful view of the data to speedup some mathematical operations

        Args:
            tracks (Collection[Track]): A collection of tracks (usually from the same video)
            frame_range (Optional[Tuple[int, int]]): Frame range (start included, end excluded)
                If None, the minimal range to hold all points is computed

        Returns:
            torch.Tensor: Tracks data in a single tensor
                Shape: (T, N, dim), dtype: float32

        """
        if not tracks:
            raise ValueError("Cannot tensorize an empty collection of Tracks")

        # Find the spatial dimension (all tracks should share the same one)
        dim = next(iter(tracks)).points.shape[1]

        # Compute once start and end of each track
        starts = [track.start for track in tracks]
        ends = [track.start + len(track) for track in tracks]

        if frame_range is None:
            start = min(starts)
            end = max(ends)
        else:
            start, end = frame_range

        points = torch.full((end - start, len(tracks), dim), torch.nan)

        for i, (track_start, track_end, track) in enumerate(zip(starts, ends, tracks)):
            if track_end <= start or track_start >= end:
                continue

            track_end = min(end, track_end)

            points[max(0, track_start - start) : track_end - start, i, :] = track.points[
                max(0, start - track_start) : track_end - track_start
            ]

        return points

    @staticmethod
    def _tensorize_det_ids(tracks: Collection[Track]) -> torch.Tensor:
        """Build a detection identifiers tensor for multiple tracks

        Args:
            tracks (Collection[Track]): A collection of tracks (usually from the same video)

        Returns:
            torch.Tensor: Detection identifiers in a single tensor
                Shape: (T, N), dtype: int32
        """
        if not tracks:
            raise ValueError("Cannot tensorize an empty collection of Tracks")

        # Compute once start and end of each track
        starts = [track.start for track in tracks]
        ends = [track.start + len(track) for track in tracks]

        start = min(starts)
        end = max(ends)
        ids = torch.full((end - start, len(tracks)), -1, dtype=torch.int32)

        for i, (track_start, track_end, track) in enumerate(zip(starts, ends, tracks)):
            ids[track_start - start : track_end - start, i] = track.detection_ids

        return ids

    @staticmethod
    def save(tracks: Collection[Track], path: Union[str, os.PathLike]) -> None:
        """Save a collection of tracks to path

        Format: pt (pytorch)

        .. code-block::

            {
                "offset": int
                "ids": Tensor (N, ), int64
                "points": Tensor (T, N, dim), float32
                "det_ids": Tensor (T, N), int32
                "merge_ids": Tensor (N, ), int32
                "parent_ids": Tensor (N, ), int32
            }

        Args:
            tracks (Collection[Track]): Tracks to save
            path (str | os.PathLike): Output path

        """
        ids = torch.tensor([track.identifier for track in tracks])
        merge_ids = torch.tensor([track.merge_id for track in tracks])
        parent_ids = torch.tensor([track.parent_id for track in tracks])
        offset = min(track.start for track in tracks)
        points = Track.tensorize(tracks)
        det_ids = Track._tensorize_det_ids(tracks)
        torch.save(
            {
                "offset": offset,
                "ids": ids,
                "points": points,
                "det_ids": det_ids,
                "merge_ids": merge_ids,
                "parent_ids": parent_ids,
            },
            path,
        )

    @staticmethod
    def load(path: Union[str, os.PathLike]) -> List[Track]:  # pylint: disable=too-many-locals
        """Load a collection of tracks from path

        Args:
            path (str | os.PathLike): Input path

        """
        data: dict = torch.load(path, map_location="cpu", weights_only=True)
        offset: int = data.get("offset", 0)
        points: torch.Tensor = data["points"]
        ids: torch.Tensor = data["ids"]
        det_ids: torch.Tensor = data.get("det_ids", torch.full(points.shape[:2], -1, dtype=torch.int32))
        merge_ids: torch.Tensor = data.get("merge_ids", torch.full(points.shape[1:2], -1, dtype=torch.int32))
        parent_ids: torch.Tensor = data.get("parent_ids", torch.full(points.shape[1:2], -1, dtype=torch.int32))

        frames = torch.arange(points.shape[0])
        defined = ~torch.isnan(points).all(dim=-1)

        tracks = []

        for i, identifier in enumerate(ids.tolist()):
            start = cast(int, frames[defined[:, i]].min().item())
            end = cast(int, frames[defined[:, i]].max().item())
            merge_id = cast(int, merge_ids[i].item())
            parent_id = cast(int, parent_ids[i].item())
            tracks.append(
                Track(
                    start + offset,
                    points[start : end + 1, i],
                    identifier,
                    det_ids[start : end + 1, i],
                    merge_id=merge_id,
                    parent_id=parent_id,
                )
            )

        Track.check_tracks(tracks, warn=True)

        return tracks

    @staticmethod
    def check_tracks(tracks: Collection[Track], warn=False):
        """Performs additional consistency checks that cannot be done at the Track level

        It will check that each track in the Collection has a different identifier.
        And it will check that merge_ids and parent_ids are correctly defined:

        1. 2 tracks should have the same merge id to a third one starting right after the two ended.
        2. 2 tracks should have the same parent id to a third one finishing right before the two started.

        Args:
            tracks (Collection[Track]): Collection of tracks to check
            warn (bool): Will only raise a warning instead of an Exception

        """
        if warn:
            try:
                _check_tracks(tracks)
            except (IndexError, AssertionError) as err:
                warnings.warn(f"Inconsistent tracks: {err}")
        else:
            _check_tracks(tracks)

    @staticmethod
    def reverse(tracks: Collection[Track], video_length=-1) -> List[Track]:
        """Reverse tracks in time.

        It will keep the same order for tracks and the same identifiers. Points are in reversed orders.
        A merge id becomes a parent id (and vice versa).

        Args:
            tracks (Collection[Track]): Collection of tracks to reverse
            video_length (int): Optional length of the video. If not provided, it is inferred
                from the last tracked object.
                Default: -1 (inferred from tracks)

        """
        if video_length == -1:
            video_length = max(track.start + len(track) for track in tracks)

        reversed_tracks = []
        for track in tracks:
            reversed_tracks.append(
                Track(
                    video_length - (track.start + len(track)),
                    torch.flip(track.points, (0,)),
                    track.identifier,
                    torch.flip(track.detection_ids, (0,)),
                    merge_id=track.parent_id,
                    parent_id=track.merge_id,
                )
            )

        return reversed_tracks


def update_detection_ids(  # pylint: disable=too-many-locals
    tracks: Collection[Track], detections_sequence: Sequence[byotrack.Detections], using_segmentation=True
) -> None:
    """Update the `detections_ids` attribute of each track inplace

    For each frame and each track, a perfectly matching detection is searched (the track position should be equal
    to the detection position). If a match is found, it is registered in the `detections_ids` attribute.

    This is useful to fill the `detection_ids` attributes after a wrapping linking code (See EMHT or TrackMate).
    For this code to work, the linking algorithm that produces tracks should use the detection position
    as the track position without using any temporal/spatial smoothing.

    Args:
        tracks (Collection[Track]): The tracks to update inplace
        detections_sequence (Sequence[byotrack.Detections]): Detections for the different frames
            It should directly be the detections used in the linking algorithm
        using_segmentation (bool): Whether to use the segmentation to compute position of detections
            or use position if available. (Icy and Fiji are only given the segmentation)
            It uses the mean position (not the median).

    """
    if not tracks:
        return

    frame_range = (min(track.start for track in tracks), max(track.start + len(track) for track in tracks))
    points = Track.tensorize(tracks, frame_range=frame_range)

    # Store tracks as an array of objects to access advance slicing
    # And reset detection_ids
    tracks_array = np.empty((len(tracks),), dtype=np.object_)
    for i, track in enumerate(tracks):
        tracks_array[i] = track
        track.detection_ids[:] = -1

    for frame_id, detections in enumerate(detections_sequence):
        if detections.length == 0:
            continue
        if frame_id < frame_range[0] or frame_id >= frame_range[1]:
            continue

        valid_tracks = ~torch.isnan(points[frame_id - frame_range[0]]).any(dim=1)
        if valid_tracks.sum() == 0:
            continue

        valid_points = points[frame_id - frame_range[0]][valid_tracks]

        # If using seg, we rely solely on the segmentation
        if using_segmentation and "position" in detections.data:
            position = torch.tensor(_position_from_segmentation(detections.segmentation.numpy()))
        else:
            position = detections.position

        # Compute dist between tracks and detections at time t (Shape: (N_det, N_track))
        dist = (valid_points[None] - position[:, None]).abs().sum(dim=-1)
        mini, argmin = torch.min(dist, dim=0)
        match = mini < 1e-5  # Keep only perfect matches
        argmin = argmin[match]

        for i, track in enumerate(tracks_array[valid_tracks.numpy()][match.numpy()]):
            track.detection_ids[frame_id - track.start] = argmin[i]
