from __future__ import annotations

import os
from typing import cast, Collection, Optional, Tuple, Union

import torch


def _check_points(points: torch.Tensor) -> None:
    """Check points validity (type, shape)"""
    assert len(points.shape) == 2
    assert points.shape[0] > 0
    assert points.shape[1] in (2, 3)  # Only 2d and 3d supported
    assert points.dtype is torch.float32


class Track:
    """Track for a given particle

    A track is defined by an (non-unique) identifier, a starting frame and a succession of positions.

    Attributes:
        identifier (int): Identifier of the track (non-unique)
        start (int): Starting frame of the track
        points (torch.Tensor): Positions (i, j) of the particle (from starting frame to ending frame)
            Shape: (T, D), dtype: float32

    """

    _next_identifier = 0

    def __init__(self, start: int, points: torch.Tensor, identifier: Optional[int] = None) -> None:
        if identifier is None:
            self.identifier = Track._next_identifier
            Track._next_identifier += 1
        else:
            self.identifier = identifier

        _check_points(points)

        self.start = start
        self.points = points

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, frame_id: int) -> torch.Tensor:
        """Return the position of tracked particle for the given frame_id

        Args:
            frame_id (int): id of the frame in the video

        Returns:
            torch.Tensor: Position (i, j) at the given frame (NaN if unknown)
                Shape: (D, ), dtype: float32

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
                Shape: (T, N, D), dtype: float32

        """
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
    def save(tracks: Collection[Track], path: Union[str, os.PathLike]) -> None:
        """Save a collection of tracks to path

        Format: pt (pytorch)

        .. code-block::

            {
                "offset": int
                "ids": Tensor (N, ), int64
                "points": Tensor (T, N, D), float32
            }

        Args:
            tracks (Collection[Track]): Tracks to save
            path (str | os.PathLike): Output path

        """
        ids = torch.tensor([track.identifier for track in tracks])
        offset = min(track.start for track in tracks)
        points = Track.tensorize(tracks)
        torch.save({"offset": offset, "ids": ids, "points": points}, path)

    @staticmethod
    def load(path: Union[str, os.PathLike]) -> Collection[Track]:
        """Load a collection of tracks from path

        Args:
            path (str | os.PathLike): Input path

        """
        data: dict = torch.load(path, map_location="cpu")
        offset: int = data.get("offset", 0)
        points: torch.Tensor = data["points"]
        ids: torch.Tensor = data["ids"]

        frames = torch.arange(points.shape[0])

        tracks = []

        for i, identifier in enumerate(ids.tolist()):
            track_points = points[:, i]
            defined = ~torch.isnan(track_points).all(dim=-1)
            track_points = track_points[defined]
            start = cast(int, frames[defined].min().item())
            tracks.append(Track(start + offset, track_points, identifier))

        return tracks
