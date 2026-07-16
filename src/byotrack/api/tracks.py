from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import tqdm.auto as tqdm

from byotrack.api.detections.segmentation_detections import _position_from_segmentation

if TYPE_CHECKING:
    import os
    from collections.abc import Collection, Sequence

    import byotrack


def _check_points(points: torch.Tensor) -> None:
    if len(points.shape) != 2 or points.shape[0] == 0 or points.shape[1] not in (2, 3):  # noqa: PLR2004
        raise ValueError("Points tensor is expected to be (T, 2) or (T, 3) with T>0")
    if points.dtype is not torch.float32:
        raise ValueError("Points tensor is expected to have a torch.float32 dtype")


def _check_detection_ids(detection_ids: torch.Tensor, size: int) -> None:
    if detection_ids.shape != (size,):
        raise ValueError("`detection_ids` tensor is expected to match the size (T,)")
    if detection_ids.dtype is not torch.int32:
        raise ValueError("`detection_ids` tensor is expected to have a torch.int32 dtype")


def _check_tracks(tracks: Collection[Track]) -> bool:  # noqa: C901
    """Check consistency of a Collection of tracks. See `Track.check_tracks`."""
    id_to_track = {track.identifier: track for track in tracks}
    merge_count: dict[int, int] = {}
    parent_count: dict[int, int] = {}
    valid = True

    if len(id_to_track) != len(tracks):
        warnings.warn("Found duplicated identifiers", stacklevel=2)
        valid = False

    for track in tracks:
        if track.parent_id != -1:
            parent = id_to_track[track.parent_id]
            end = parent.start + len(parent)

            if track.start != end:
                warnings.warn(  # Warn also for gap in splitting (though theoretically valid)
                    f"Track {parent.identifier} (Last frame: {end - 1}) splits into track {track.identifier}. "
                    f"But track {track.identifier} starts at {track.start} != {end}.",
                    stacklevel=2,
                )
                valid = track.start - end > 0

            parent_count[track.parent_id] = parent_count.get(track.parent_id, 0) + 1
        if track.merge_id != -1:
            child = id_to_track[track.merge_id]
            end = track.start + len(track)

            if end != child.start:
                warnings.warn(  # Warn also for gap in merging (though theoretically valid)
                    f"Track {track.identifier} (Last frame: {end - 1}) merges into track {child.identifier}. "
                    f"But track {child.identifier} starts at {child.start} != {end}.",
                    stacklevel=2,
                )
                valid = child.start - end > 0

            merge_count[track.merge_id] = merge_count.get(track.merge_id, 0) + 1

    for merge_id, count in merge_count.items():
        if count != 2:  # noqa: PLR2004
            warnings.warn(f"Track {merge_id} is the results of {count} ! = 2 tracks.", stacklevel=2)

    for parent_id, count in parent_count.items():
        if count != 2:  # noqa: PLR2004
            warnings.warn(f"{parent_id} splits into {count} ! = 2 tracks.", stacklevel=2)

    return valid


class Track:
    """Track for a given target.

    A track is defined by a positive identifier, a starting frame and a succession of positions.
    In a detect-then-track context, a track can optionally contains the detection identifiers
    for each time frame (-1 if non-linked to any particular detection at this time frame)

    It supports target splitting and merging through a mapping between tracks with `parent_id` and
    `merge_id` attributes. By construction, when a track splits, it terminates (and the children are born).

    Attributes:
        identifier (int): Identifier of the track (positive)
        start (int): Starting frame of the track
        points (torch.Tensor): Positions (i, j) of the target (from starting frame to ending frame)
            Shape: (T, dim), dtype: float32
        detection_ids (torch.Tensor): Detection id for each time frame (-1 if unknown or non-linked
            to a particular detection at this time frame)
            Shape: (T,), dtype: int32
        merge_id (int): Optional identifier to the merged track. This allows to handle merges (such as cell divisions
            in a reversed temporal order). At least one other track should share the same merge_id.
            This should not be used to do tracklet stitching (See `DistStitcher` for such use cases)
            Default: -1 (Merged to no one)
        parent_id (int) Optional identifier to a parent track. This allows to handle splits / target spawning (such as
            cell divisions). At least one other track should share the same parent_id.
            This should not be used to do tracklet stitching (See `DistStitcher` for such use cases)
            Default: -1 (No parent)

    """

    _next_identifier = 0

    def __init__(
        self,
        start: int,
        points: torch.Tensor,
        identifier: int | None = None,
        detection_ids: torch.Tensor | None = None,
        *,
        merge_id: int = -1,
        parent_id: int = -1,
    ) -> None:
        if identifier is None:
            self.identifier = Track._next_identifier
            Track._next_identifier += 1
        else:
            if identifier < 0:
                raise ValueError("Track identifiers cannot be negative")
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

    def __len__(self) -> int:  # noqa: D105
        return self.points.shape[0]

    @property
    def dim(self) -> int:
        """Return the dimension (2D or 3D) of the track."""
        return self.points.shape[1]

    def __getitem__(self, frame_id: int) -> torch.Tensor:
        """Return the position of tracked target for the given frame_id.

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
    def tensorize(tracks: Collection[Track], frame_range: tuple[int, int] | None = None) -> torch.Tensor:
        """Convert a collection of tracks into a tensor on a given frame_range.

        Useful view of the data to speedup some mathematical operations

        Args:
            tracks (Collection[Track]): A collection of tracks (usually from the same video)
            frame_range (tuple[int, int] | None): Frame range (start included, end excluded)
                If None, the minimal range to hold all points is computed

        Returns:
            torch.Tensor: Tracks data in a single tensor
                Shape: (T, N, dim), dtype: float32

        """
        if not tracks:
            raise ValueError("Cannot tensorize an empty collection of Tracks")

        # Find the spatial dimension (all tracks should share the same one)
        dim = max(track.dim for track in tracks)
        if dim != min(track.dim for track in tracks):
            raise ValueError("Tracks should share the same spatial dimension.")

        # Compute once start and end of each track
        starts = [track.start for track in tracks]
        ends = [track.start + len(track) for track in tracks]

        if frame_range is None:
            start = min(starts)
            end = max(ends)
        else:
            start, end = frame_range

        points = torch.full((end - start, len(tracks), dim), torch.nan)

        for i, (track_start, track_end, track) in enumerate(zip(starts, ends, tracks, strict=True)):
            if track_end <= start or track_start >= end:
                continue

            track_end = min(end, track_end)  # noqa: PLW2901

            points[max(0, track_start - start) : track_end - start, i, :] = track.points[
                max(0, start - track_start) : track_end - track_start
            ]

        return points

    @staticmethod
    def _tensorize_det_ids(tracks: Collection[Track]) -> torch.Tensor:
        """Build a detection identifiers tensor for multiple tracks.

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

        for i, (track_start, track_end, track) in enumerate(zip(starts, ends, tracks, strict=True)):
            ids[track_start - start : track_end - start, i] = track.detection_ids

        return ids

    @staticmethod
    def save(tracks: Collection[Track], path: str | os.PathLike) -> None:
        """Save a collection of tracks to path.

        Format: pt (pytorch)

        .. code-block::

            {
                "offset": int
                "ids": Tensor (N, ), int32
                "points": Tensor (T, N, dim), float32
                "det_ids": Tensor (T, N), int32
                "merge_ids": Tensor (N, ), int32
                "parent_ids": Tensor (N, ), int32
            }

        Args:
            tracks (Collection[Track]): Tracks to save
            path (str | os.PathLike): Output path

        """
        if not tracks:
            raise ValueError("No tracks to save.")
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
    def load(path: str | os.PathLike) -> list[Track]:
        """Load a collection of tracks from path.

        Args:
            path (str | os.PathLike): Input path

        Returns:
            list[Track]: Loaded tracks

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
            start = cast("int", frames[defined[:, i]].min().item())
            end = cast("int", frames[defined[:, i]].max().item())
            merge_id = cast("int", merge_ids[i].item())
            parent_id = cast("int", parent_ids[i].item())
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
    def check_tracks(tracks: Collection[Track], *, warn=False) -> None:
        """Performs additional consistency checks that cannot be done at the Track level.

        It will check that each track in the Collection has a different identifier.
        And it will check that merge_ids and parent_ids are correctly defined:

        1. Several tracks should have the same merge id to another following one.
        2. Several tracks should have the same parent id to another preceding one.

        Args:
            tracks (Collection[Track]): Collection of tracks to check
            warn (bool): Will only raise a warning instead of an Exception

        """
        if not _check_tracks(tracks) and not warn:
            raise ValueError("Invalid tracks")

    @staticmethod
    def reverse(tracks: Collection[Track], video_length=-1) -> list[Track]:
        """Reverse tracks in time.

        It will keep the same order for tracks and the same identifiers. Points are in reversed orders.
        A merge id becomes a parent id (and vice versa).

        Args:
            tracks (Collection[Track]): Collection of tracks to reverse
            video_length (int): Optional length of the video. If not provided, it is inferred
                from the last tracked object.
                Default: -1 (inferred from tracks)

        Returns:
            list[Track]: Reversed tracks

        """
        if video_length == -1:
            video_length = max(track.start + len(track) for track in tracks)

        return [
            Track(
                video_length - (track.start + len(track)),
                torch.flip(track.points, (0,)),
                track.identifier,
                torch.flip(track.detection_ids, (0,)),
                merge_id=track.parent_id,
                parent_id=track.merge_id,
            )
            for track in tracks
        ]


def update_detection_ids(
    tracks: Collection[Track],
    detections_sequence: Sequence[byotrack.Detections],
    *,
    threshold=1e-5,
    use_segmentation=True,
) -> None:
    """Update the `detections_ids` attribute of each track inplace.

    For each frame and each track, a perfectly matching detection is searched (the track position should be equal
    to the detection position). If a match is found, it is registered in the `detections_ids` attribute.

    This is useful to fill the `detection_ids` attributes after a wrapping linking code (See EMHT or TrackMate).
    For this code to work, the linking algorithm that produces tracks should use the detection position
    as the track position without using any temporal/spatial smoothing.

    Args:
        tracks (Collection[Track]): The tracks to update inplace.
        detections_sequence (Sequence[byotrack.Detections]): Detections for the different frames
            It should directly be the detections used in the linking algorithm.
        threshold (float): Distance threshold for matching with a detection.
            Should be small to enforce a perfect match.
            Default: 1e-5.
        use_segmentation (bool): Extract the mean position from the `segmentation` property instead of
            relying on the `position` property (Icy and Fiji are only given the segmentation).

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
        if use_segmentation:
            position = torch.tensor(_position_from_segmentation(detections.segmentation.numpy()))
        else:
            position = detections.position

        # Compute dist between tracks and detections at time t (Shape: (N_det, N_track))
        dist = (valid_points[None] - position[:, None]).abs().sum(dim=-1)
        mini, argmin = torch.min(dist, dim=0)
        match = mini < threshold  # Keep only perfect matches
        argmin = argmin[match]

        for i, track in enumerate(tracks_array[valid_tracks.numpy()][match.numpy()]):
            track.detection_ids[frame_id - track.start] = argmin[i]


def _resolve_disk_radii(
    radius: float | torch.Tensor,
    n_tracks: int,
    n_frames: int,
    dim: int,
    anisotropy: tuple[float, float, float],
) -> torch.Tensor:
    """Expand a radius spec to (T, N, dim), applying anisotropy per axis.

    Args:
        radius (float | torch.Tensor): Either a scalar, or a tensor directly broadcastable to (T, N, dim)
            using standard (right-aligned) torch broadcasting rules. In particular, a per-track-only
            radius must be shaped (N, 1), not (N,) (which would broadcast against `dim` instead of `N`),
            and a per-track-and-frame radius must be shaped (T, N, 1), not (T, N), for the same reason.
            Should be expressed in non-scaled coordinates.
        n_tracks (int): Number of tracks (N).
        n_frames (int): Number of frames (T).
        dim (int): Spatial dimension (2 or 3).
        anisotropy (tuple[float, float, float]): Anisotropy factors (ani_z, ani_y, ani_x) used to
            convert an isotropic radius into per-axis pixel-space radii (See `statistics.anisotropy`).

    Returns:
        torch.Tensor: Per-frame, per-track and per-axis radius.
            Shape: (T, N, dim), dtype: float32
    """
    if isinstance(radius, (int, float)):
        radius = torch.full((1, 1, 1), radius, dtype=torch.float32)

    return radius.to(torch.float32).expand(n_frames, n_tracks, dim) / torch.tensor(anisotropy)[-dim:]


def update_detections_from_tracks(  # noqa: C901
    detections_sequence: Sequence[byotrack.Detections],
    tracks: Collection[Track],
    *,
    radius: float | torch.Tensor = 2.0,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    drop_false_positives: bool = True,
    draw_false_negatives: bool = True,
    overwrite: bool = False,
) -> list[byotrack.Detections]:
    """Update a sequence of Detections using tracking results.

    For each frame, detections matched to a track (`detection_ids` should already be populated)
    are relabeled with their track `identifier`. Unmatched detections (false positives) can
    optionally be dropped, and tracks with a known position but no matching detection (false
    negatives) can optionally be materialized as a disk-shaped detection (See `Detections.add_disks`).

    The returned sequence always has the same length as `detections_sequence` (this function never extends it,
    even if tracks span further in time -- pad `detections_sequence` first if that behavior is wanted).

    Args:
        detections_sequence (Sequence[byotrack.Detections]): Detections for each frame to update.
        tracks (Collection[Track]): Tracks to use. `detection_ids` should already be set.
        radius (float | torch.Tensor): Radius of the disks drawn for false negatives. Either a scalar, or
            a tensor directly broadcastable to (T, N, dim) using standard (right-aligned) torch broadcasting
            rules (See `_resolve_disk_radii`). In particular, a per-track-only radius must be shaped (N, 1),
            not (N,), and a per-track-and-frame radius must be shaped (T, N, 1), not (T, N).
            Should be expressed in non-scaled coordinates.
            Default: 2.0
        anisotropy (tuple[float, float, float]): Anisotropy factors (ani_z, ani_y, ani_x) used to convert
            `radius` into per-axis pixel-space radii.
            Default: (1.0, 1.0, 1.0) (no scaling)
        drop_false_positives (bool): Drop detections that are not matched to any track.
            Be aware that if you do not remove false positives, then, the track identifiers
            should not overlap with the remaining false positives detections.
            Default: True
        draw_false_negatives (bool): Draw a disk for tracks with a known position but no matching detection.
            Default: True
        overwrite (bool): Allow disks (false negatives) to overwrite pre-existing detection pixels
            (Only relevant for `SegmentationDetections`, see `Detections.add_disks`).
            Default: False

    Returns:
        list[byotrack.Detections]: The updated detections, one per frame of `detections_sequence`.
    """
    if not tracks:
        return list(detections_sequence)

    if not detections_sequence:
        return []

    radii: torch.Tensor | None = None
    if draw_false_negatives:
        radii = _resolve_disk_radii(
            radius, len(tracks), len(detections_sequence), detections_sequence[0].dim, anisotropy
        )

    updated: list[byotrack.Detections] = []

    for frame_id, detections in enumerate(tqdm.tqdm(detections_sequence, desc="Update detections")):
        new_labels = detections.labels.clone()
        is_labeled = torch.zeros(detections.length, dtype=torch.bool)

        missed_positions = []
        missed_labels = []
        missed_track_indices = []

        for track_index, track in enumerate(tracks):
            if not track.start <= frame_id < track.start + len(track):
                continue

            det_id = int(track.detection_ids[frame_id - track.start])
            if det_id != -1:
                new_labels[det_id] = track.identifier
                is_labeled[det_id] = True
            elif draw_false_negatives:
                position = track[frame_id]
                if not torch.isnan(position).any():
                    missed_positions.append(position)
                    missed_labels.append(track.identifier)
                    missed_track_indices.append(track_index)
                else:
                    warnings.warn(
                        f"Track {track.identifier} has an undefined position at frame {frame_id} (inside its "
                        "own segment) and no matching detection: it will be skipped in the output, which may "
                        "cause issues with some evaluation software (traccuracy, CTC, ...). Consider filling "
                        "holes with Interpolaters.",
                        stacklevel=2,
                    )

        kept = is_labeled if drop_false_positives else torch.ones(detections.length, dtype=torch.bool)

        # Relabel the detections with their associated track.identifier
        # Then filter false positive detections
        new_detections = detections.relabel(new_labels).filter(kept)

        # Finally, draw disks for missed detections
        if missed_positions and radii is not None:
            new_detections = new_detections.add_disks(
                torch.stack(missed_positions),
                radii[frame_id, missed_track_indices],
                labels=torch.tensor(missed_labels, dtype=torch.int32),
                overwrite=overwrite,
            )

            # A disk may not appear in the result (out of frame, or occluded with overwrite=False),
            # or, with overwrite=True, may have clobbered an already-matched detection.
            expected = set(new_labels[kept].tolist()) | set(missed_labels)
            absent = expected - set(new_detections.labels.tolist())
            if absent:
                warnings.warn(
                    f"Some tracks could not be represented in detections at frame {frame_id}: {absent}.\n"
                    "This may cause errors with some evaluation software (traccuracy, CTC, ...).",
                    stacklevel=2,
                )

        updated.append(new_detections)

    return updated
