from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import tqdm.auto as tqdm

import byotrack
import byotrack.video.video

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence


def detections_to_napari_segmentation(detections_sequence: Sequence[byotrack.Detections]) -> np.ndarray:
    """Convert detections data to a segmentation array compatible with Napari labels layer.

    Args:
        detections_sequence (Sequence[byotrack.Detections]): Sequence of T detections (one per frame).
            All detections are expected to share the same spatial shape.

    Returns:
        np.ndarray: Instance segmentation video built by stacking each frame segmentation.
            Shape: (T, [D, ]H, W), dtype: uint16

    """
    segmentation = np.zeros((len(detections_sequence), *detections_sequence[0].shape), dtype=np.uint16)

    for frame_id, detections in enumerate(detections_sequence):
        segmentation[frame_id] = detections.segmentation.numpy().astype(np.uint16)

    return segmentation


def detections_to_napari_points(detections_sequence: Sequence[byotrack.Detections]) -> np.ndarray:
    """Convert detections data to a points array compatible with Napari points layer.

    Args:
        detections_sequence (Sequence[byotrack.Detections]): Sequence of T detections (one per frame).

    Returns:
        np.ndarray: Points with their frame index prepended to their position ([t, [k, ]i, j]).
            Shape: (N, dim + 1), dtype: float32, where N is the total number of detections.

    """
    points = np.zeros(
        (sum(len(detections) for detections in detections_sequence), detections_sequence[0].dim + 1), dtype=np.float32
    )

    seen = 0
    for frame_id, detections in enumerate(detections_sequence):
        points[seen : seen + len(detections), 0] = frame_id
        points[seen : seen + len(detections), 1:] = detections.position.numpy()

        seen += len(detections)

    return points


# TODO: Detection to bbox? => Shape layer is not very intuitive for 3D bbox...


def tracks_to_napari_tracks(tracks: Collection[byotrack.Track]) -> tuple[np.ndarray, dict[int, list[int]], np.ndarray]:
    """Convert tracks data to Napari tracks layer format (points, graph, lineage_ids).

    NaN positions (undetected frames within a track) are dropped from the resulting points.

    Args:
        tracks (Collection[byotrack.Track]): Tracks to convert.

    Returns:
        tuple: (points, graph, lineage_ids)
            points (np.ndarray): Points with track identifier and frame index prepended to their
                position ([identifier, t, [k, ]i, j]), as expected by Napari's ``add_tracks``.
                Shape: (N, dim + 2), dtype: float32
            graph (dict[int, list[int]]): Mapping of each track identifier to the list of its parent
                track identifiers (split and merge events), as expected by Napari's ``add_tracks``.
            lineage_ids (np.ndarray): Lineage identifier of each point in `points`, shared by all tracks
                that are connected (through splits/merges) in the same lineage tree.
                Shape: (N,), dtype: uint16

    """
    dim = next(iter(tracks)).dim
    track_points = np.zeros((sum(len(track) for track in tracks), dim + 2), dtype=np.float32)

    seen = 0
    for track in tracks:
        track_points[seen : seen + len(track), 0] = track.identifier
        track_points[seen : seen + len(track), 1] = np.arange(track.start, track.start + len(track))
        track_points[seen : seen + len(track), 2:] = track.points.numpy()

        seen += len(track)

    # Remove NaNs points if any
    track_points = track_points[~np.isnan(track_points).any(axis=-1)]

    # Extract the "graph" attribute of Napari Tracks layer
    parents: dict[int, list[int]] = {track.identifier: [] for track in tracks}
    for track in tracks:
        if track.parent_id >= 0:
            parents[track.identifier].append(track.parent_id)

        if track.merge_id >= 0:
            parents[track.merge_id].append(track.identifier)

    # Let's find the connected components to extract a consistent lineage id property
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_nodes_from(parents.keys())
    for track_id, parents_ in parents.items():
        for parent_id in parents_:
            graph.add_edge(parent_id, track_id)

    track_to_lineage = np.zeros(max(parents) + 1, dtype=np.uint16)
    for lineage_id, track_ids in enumerate(nx.connected_components(graph.to_undirected())):
        for track_id in track_ids:
            track_to_lineage[track_id] = lineage_id

    return track_points, parents, track_to_lineage[track_points[:, 0].astype(np.int32)]


def _initialize_grid(spatial_shape: tuple[int, ...], grid_step: int, scale: tuple[float, ...]) -> np.ndarray:
    """Build a uniform grid of control points covering a spatial shape.

    Args:
        spatial_shape (tuple[int, ...]): Spatial shape of the frame ([D, ]H, W) the grid should cover.
        grid_step (int): Spacing (in scaled/world units) between two consecutive grid points.
        scale (tuple[float, ...]): Spatial anisotropy ([Z, ]Y, X) used to convert `grid_step` into
            a pixel spacing for each spatial dimension.

    Returns:
        np.ndarray: Grid points with a leading time coordinate set to 0 ([0, [k, ]i, j]).
            Shape: ([D', ]H', W', dim + 1), dtype: float

    """
    dim = len(spatial_shape)

    # Spatial grid
    grid = np.stack(
        np.meshgrid(*[np.arange(0, spatial_shape[i], grid_step / scale[i]) for i in range(dim)], indexing="ij"),
        axis=-1,
    )

    # Prepend time dimension to the grid
    return np.concatenate((np.zeros_like(grid[..., :1]), grid), axis=-1)


def _find_paths_in_grid(grid_shape: tuple[int, ...]) -> list[np.ndarray]:
    """Build the index paths of the grid edges to display, along each spatial axis.

    For each axis of the grid, this builds one path (a line of consecutive point indices) per
    remaining position, so that drawing all the returned paths reconstructs the full grid wireframe.

    Args:
        grid_shape (tuple[int, ...]): Shape of the grid of control points ([D, ]H, W).

    Returns:
        list[np.ndarray]: Paths of point indices (into the flattened grid) to draw, one per grid line.

    """
    paths: list[np.ndarray] = []

    indices = np.arange(np.prod(grid_shape)).reshape(grid_shape)

    for axis in range(len(grid_shape)):
        swapped = indices.transpose(*range(axis), *range(axis + 1, len(grid_shape)), axis)
        swapped = swapped.reshape(-1, swapped.shape[-1])

        paths.extend(swapped)

    return paths


class _LazyVideoArray:
    """Duck array wrapping of byotrack.Video for lazy Napari's Image layer.

    Internal to ``add_video(..., lazy=True)``. Its indexing near the channel axis is special-cased
    to match Napari's one-shot ``channel_axis`` splitting call (see
    ``napari.layers.utils.stack_utils.slice_from_axis``), which sends a full-length index tuple with
    an int on the channel axis. This is the only situation where `Video` raises on channel-axis
    indexing (``Video`` only supports selecting a channel through a `ChannelProjection` preprocessor,
    which keeps the axis at size 1 rather than dropping it). Verified against napari 0.7.1.

    Attributes:
        video (byotrack.Video): Wrapped video.
        channel (int | None): If set, the channel already selected (and dropped) from `video`.

    """

    def __init__(self, video: byotrack.Video, channel: int | None = None) -> None:
        self.video = video
        self.channel = channel

    @property
    def shape(self) -> tuple[int, ...]:
        return self.video.shape[:-1] if self.channel is not None else self.video.shape

    @property
    def dtype(self) -> np.dtype:
        return self.video.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return len(self.video)

    def __getitem__(self, key: int | slice | tuple[int, slice]) -> _LazyVideoArray | np.ndarray:
        # NOTE: In practice, napari is always slicing and never indexing except for the channel axis
        if isinstance(key, tuple):
            key = byotrack.video.video.expand_ellipsis(key, self.ndim)

            if len(key) > self.ndim:
                raise IndexError("Too many indices for video.")

            if self.channel is None and len(key) == self.ndim and isinstance(key[-1], int):
                # Only valid meaning of this key: select and drop the channel axis.
                return _LazyVideoArray(self.video, channel=key[-1])[key[:-1]]

        result = self.video[key]

        if isinstance(result, byotrack.Video):
            return _LazyVideoArray(result, channel=self.channel)

        return result if self.channel is None else result[..., self.channel]

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        if self.channel is None:
            return np.asarray(self.video, dtype=dtype)

        return np.asarray(self.video[..., self.channel : self.channel + 1], dtype=dtype)[..., 0]


class _LazySegmentationArray:
    """Duck array wrapping of Sequence[byotrack.Detections] for lazy Napari's Labels layer.

    Internal to ``add_detections(..., lazy=True)``. It first handles the temporal slicing
    to only select the required frames, and then apply the spatial slicing.

    Warning: This may reduce performance if the temporal axis is transposed with one spatial axis.

    Warning: This is not compatible with the label editing tools of Napari.

    Attributes:
        detections_sequence (Sequence[byotrack.Detections]): Wrapped detections.

    """

    def __init__(self, detections_sequence: Sequence[byotrack.Detections]) -> None:
        self.detections_sequence = detections_sequence

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.detections_sequence), *self.detections_sequence[0].shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint16)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return len(self.detections_sequence)

    def __getitem__(self, key: int | slice | tuple) -> _LazySegmentationArray | np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)

        key = byotrack.video.video.expand_ellipsis(key, self.ndim)

        time_key = key[0]
        spatial_key = key[1:]

        if isinstance(time_key, int):
            return self.detections_sequence[time_key].segmentation.numpy().astype(np.uint16)[spatial_key]

        return detections_to_napari_segmentation(self.detections_sequence[time_key])[(slice(None), *spatial_key)]

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        return detections_to_napari_segmentation(self.detections_sequence)


def precompute_optical_flow(
    video: Sequence[np.ndarray] | np.ndarray,
    optflow: byotrack.OpticalFlow,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Precompute bidirectional optical flow maps between each pair of consecutive frames.

    Args:
        video (Sequence[np.ndarray] | np.ndarray): Sequence of T frames (array).
            Each array is expected to have a shape ([D, ]H, W, C)
        optflow (byotrack.OpticalFlow): Optical flow algorithm used to preprocess frames and
            compute the flow maps.

    Returns:
        tuple: (forward_flows, backward_flows)
            forward_flows (list[np.ndarray]): Flow map from frame t to frame t+1, for each of the
                T - 1 consecutive frame pairs.
            backward_flows (list[np.ndarray]): Flow map from frame t+1 to frame t, for each of the
                T - 1 consecutive frame pairs.

    """
    if len(video) < 1:
        return [], []

    forward_flows: list[np.ndarray] = []
    backward_flows: list[np.ndarray] = []

    src = optflow.preprocess(video[0])
    for frame in tqdm.tqdm(video[1:], desc="Pre-computing optical flows"):
        dst = optflow.preprocess(frame)
        forward_flows.append(optflow.compute(src, dst))
        backward_flows.append(optflow.compute(dst, src))
        src = dst

    return forward_flows, backward_flows
