from __future__ import annotations

import functools
import pathlib
from typing import TYPE_CHECKING, Any

import geff
import geff_spec  # type: ignore[import-untyped]
import numpy as np
import torch
import tqdm.auto as tqdm
import zarr  # type: ignore[import-untyped]

import byotrack

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Collection, Sequence


def _delay_execution(fn: Callable) -> Callable:
    """Delay the execution of the wrapped function.

    It returns an executor that needs to be called again to execute the function.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Callable:
        def executor():
            return fn(*args, **kwargs)

        return executor

    return wrapper


def save_video_to_zarr(
    video: Sequence[np.ndarray] | np.ndarray,
    path: str | os.PathLike,
    *,
    chunks: tuple[int, ...] | None = None,
    channel: int | None = None,
    **zarr_kwargs: Any,
) -> zarr.Array:
    """Save a video into a single on-disk zarr array.

    Shape and dtype are preserved from the original video.

    However, as a channel axis is not always supported (e.g. napari-geff), it can be
    removed via the `channel` argument, saving only the given channel, without channel axis.

    Args:
        video (Sequence[np.ndarray]): Sequence of T frames (array).
            Each array is expected to have a shape ([D, ]H, W, C)
        path (str | os.PathLike): Output zarr store path.
        chunks (tuple[int, ...] | None): Chunk shape. Defaults to one chunk per
            frame, i.e. ``(1, *frame_shape)``.
        channel (int | None): Optional channel selection. If a channel is selected,
            the channel axis is removed from the zarr and only this channel is saved.
        **zarr_kwargs: Extra kwargs forwarded to ``zarr.open_array`` (e.g.
            ``compressors``, ``overwrite``).

    Returns:
        zarr.Array: The zarr array, opened in write mode.
            Shape: (T, [D, ]H, W[, C])

    """
    shape = byotrack.video.video_shape(video)
    if channel is not None:
        shape = shape[:-1]

    chunks = chunks or (1, *shape[1:])

    store = zarr.open_array(
        store=path,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=byotrack.video.video_dtype(video),
        **zarr_kwargs,
    )

    for t in tqdm.trange(shape[0], desc="Converting video into zarr"):
        store[t] = video[t] if channel is None else video[t][..., channel]

    return store


def save_detections_to_zarr(
    detections_sequence: Sequence[byotrack.DetectionsLike],
    path: str | os.PathLike,
    *,
    chunks: tuple[int, ...] | None = None,
    **zarr_kwargs: Any,
) -> zarr.Array:
    """Save a sequence of per-frame Detections into a single on-disk zarr array.

    It stores the ``Detections.segmentation`` into a uint16 zarr of shape (T, [D, ]H, W).

    Note that it only stores the segmentation, but not the metadata (labels, confidence, etc...).
    It will also remove detections that are out of the frame shape.

    Args:
        detections_sequence (Sequence[byotrack.DetectionsLike]): Per-frame detections
            (all frames are expected to share the same spatial shape).
        path (str | os.PathLike): Output zarr store path.
        chunks (tuple[int, ...] | None): Chunk shape. Defaults to one chunk per
            frame, i.e. ``(1, [D, ]H, W)``.
        **zarr_kwargs: Extra kwargs forwarded to ``zarr.open_array`` (e.g. ``compressors``).

    Returns:
        zarr.Array: The zarr array, opened in write mode.
            Shape: (T, [D, ]H, W), dtype: np.uint16

    Raises:
        ValueError: If ``detections_sequence`` is empty.

    """
    n_frames = len(detections_sequence)
    if n_frames == 0:
        raise ValueError("detections_sequence is empty: cannot infer the array shape.")

    detections_sequence_ = [byotrack.as_detections(detections) for detections in detections_sequence]

    frame_shape = detections_sequence_[0].shape
    shape = (n_frames, *frame_shape)
    chunks = chunks or (1, *frame_shape)

    store = zarr.open_array(
        store=path,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=np.uint16,
        **zarr_kwargs,
    )

    for frame_id in tqdm.trange(n_frames, desc="Converting detections to zarr"):
        store[frame_id] = detections_sequence_[frame_id].segmentation.numpy().astype(np.uint16)

    return store


def save_tracks_to_geff(
    tracks: Collection[byotrack.Track],
    path: str | os.PathLike,
    *,
    video: Sequence[np.ndarray] | np.ndarray = (),
    detections_sequence: Sequence[byotrack.DetectionsLike] = (),
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    drop_nan: bool = True,
    split_channels: bool = True,
) -> None:
    """Save tracks (and optionally the source video/detections) into a geff store.

    Tracks are first converted into a `byotrack.TrackingGraph` and written as a geff graph at
    ``path``. If ``video`` and/or ``detections_sequence`` are given, they are saved as sibling
    on-disk zarr arrays next to the geff graph and referenced through the geff metadata
    ``related_objects``.

    Note that the geff graph is written first (creating/clearing the zarr group at ``path``, as
    ``geff.write`` is called with ``overwrite=True``), and only then are the video/segmentation
    siblings written into that now-existing directory.

    Note:
        By default, this function produce a napari-geff compatible file, using ``drop_nan=True`` and
        ``split_channels=True``. The stored tracks and video may not be fully recovered by ByoTrack.
        If you are using this method as a storage for ByoTrack, consider changing these parameters to False.

    Args:
        tracks (Collection[Track]): Tracks to save.
        path (str | os.PathLike): Output geff store path.
        video (Sequence[np.ndarray] | np.ndarray): Optional source video to save alongside the
            tracks.
            Defaults to an empty tuple (no video saved).
            Shape: (T, [D, ]H, W, C)
        detections_sequence (Sequence[byotrack.DetectionsLike]): Optional per-frame detections to
            save alongside the tracks (only the segmentation is kept, see `save_detection_to_zarr`).
            Defaults to an empty tuple (no detections saved).
        anisotropy (tuple[float, float, float]): Physical scale of the (z, y, x) axes, used as the
            geff ``axis_scales`` for the spatial axes.
        drop_nan (bool): Forwarded to `byotrack.TrackingGraph.from_tracks`: if True, points with
            undefined (NaN) positions are dropped instead of being kept as (NaN-valued) nodes.
            This is useful for readers that do not support NaN node positions. Note that
            `TrackingGraph.to_tracks` fills any gaps between non-NaN nodes. (Outer NaNs will be missing)
            Default: True.
        split_channels (bool): Split the video into one single-channel zarr array per channel
            (``video-0``, ``video-1``, ...), without channel axis (see `save_video_to_zarr`).
            This is useful for napari-geff which currently seems limited to single-channel video.
            Default: True. (`read_video_from_geff` will only read the first channel)

    """
    first_track = next(iter(tracks), None)
    dim = first_track.dim if first_track is not None else 2
    path = pathlib.Path(path)

    graph = byotrack.TrackingGraph.from_tracks(tracks, drop_nan=drop_nan)

    executors: list[Callable] = []
    related_objects: list[geff_spec.RelatedObject] = []
    if byotrack.video.video_length(video):
        # Currently napari-geff seems limited to single channel video
        # So let's break it into multiple video file, one per channel (as done for napari visualization)
        shape = byotrack.video.video_shape(video)

        if split_channels:
            for channel in range(shape[-1]):
                executors.append(
                    _delay_execution(save_video_to_zarr)(video, path / f"video-{channel}", channel=channel)
                )
                related_objects.append(geff_spec.RelatedObject(type="image", path=f"video-{channel}"))
        else:
            executors.append(_delay_execution(save_video_to_zarr)(video, path / "video"))
            related_objects.append(geff_spec.RelatedObject(type="image", path="video"))

    if len(detections_sequence):
        executors.append(_delay_execution(save_detections_to_zarr)(detections_sequence, path / "segmentation"))
        related_objects.append(geff_spec.RelatedObject(type="labels", path="segmentation"))

    metadata = geff.GeffMetadata(
        directed=True,
        node_props_metadata={},
        edge_props_metadata={},
        related_objects=related_objects,
        track_node_props={"tracklet": "track_id"},
    )

    geff.write(
        graph,
        path,
        metadata=metadata,
        overwrite=True,
        axis_names=["t", *["z", "y", "x"][-dim:]],
        axis_units=["frame", *["pixel", "pixel", "pixel"][-dim:]],
        axis_types=["time", *["space", "space", "space"][-dim:]],
        axis_scales=[1, *anisotropy[-dim:]],
    )

    # Now that the folder is created, we can save the related objects
    for executor in executors:
        executor()


def load_video_from_zarr(path: str | os.PathLike) -> byotrack.Video:
    """Load a video previously saved with `save_video_to_zarr`.

    The zarr array is opened lazily (not loaded into memory): frames are read on demand.

    Args:
        path (str | os.PathLike): Path of the zarr array to load.

    Returns:
        byotrack.Video: The video, wrapping the on-disk zarr array.
            Shape and dtype are preserved from the saved zarr array.

    """
    # Simply let's open the zarr and return the video
    # TODO: Officially support zarr in Video(zarr) and ArrayVideoReader ?
    return byotrack.Video(zarr.open_array(path))  # type: ignore[arg-type]


def load_detections_from_zarr(path: str | os.PathLike) -> list[byotrack.SegmentationDetections]:
    """Load a sequence of per-frame Detections previously saved with `save_detection_to_zarr`.

    Args:
        path (str | os.PathLike): Path of the zarr array to load.
            Shape: (T, [D, ]H, W), dtype: np.uint16 (as produced by `save_detection_to_zarr`).

    Returns:
        list[byotrack.SegmentationDetections]: One `Detections` per frame.

    """
    segmentations = zarr.open_array(path)

    return [
        byotrack.SegmentationDetections(torch.from_numpy(segmentation.astype(np.int32)))  # type: ignore[union-attr]
        for segmentation in tqdm.tqdm(
            iter(segmentations), total=segmentations.shape[0], desc="Converting zarr to detections"
        )
    ]


def load_tracks_from_geff(path: str | os.PathLike) -> list[byotrack.Track]:
    """Load tracks from a geff store, e.g. one previously saved with `save_tracks_to_geff`.

    The geff graph is read back with the networkx backend and converted into
    `byotrack.TrackingGraph` (sanitizing/remapping attribute keys) before being turned into
    `byotrack.Track` objects.

    Args:
        path (str | os.PathLike): Path of the geff store to load.

    Returns:
        list[byotrack.Track]: The reconstructed tracks from the geff store.

    """
    graph, _ = geff.read(path, backend="networkx")
    return byotrack.TrackingGraph.from_nx(graph).to_tracks()  # type: ignore[arg-type]


def load_video_from_geff(path: str | os.PathLike) -> byotrack.Video:
    """Load the video attached to a geff store, e.g. one saved with `save_tracks_to_geff`.

    Looks up ``metadata.related_objects`` for the first entry of type ``"image"`` and loads it
    with `load_video_from_zarr`.

    Note:
        `save_tracks_to_geff` may split a multi-channel video into several sibling zarr arrays
        (one ``related_object`` per channel, see its docstring). This function only loads the
        *first* one found, i.e. a single channel, not the reconstructed multi-channel video.

    Args:
        path (str | os.PathLike): Path of the geff store to load the video from.

    Returns:
        byotrack.Video: The (first) video attached to the geff store.

    Raises:
        FileNotFoundError: If the geff store has no related object of type ``"image"``.

    """
    related_objects = geff.GeffReader(path).metadata.related_objects

    if related_objects is None:  # pragma: no cover
        raise FileNotFoundError("The geff store has no image (video) related objects.")

    for related_object in related_objects:
        if related_object.type != "image":
            continue

        return load_video_from_zarr(path / related_object.path)

    raise FileNotFoundError("The geff store has no image (video) related objects.")


def load_detections_from_geff(path: str | os.PathLike) -> list[byotrack.SegmentationDetections]:
    """Load the detections attached to a geff store, e.g. one saved with `save_tracks_to_geff`.

    Looks up ``metadata.related_objects`` for the first entry of type ``"labels"`` and loads it
    with `load_detections_from_zarr`.

    Args:
        path (str | os.PathLike): Path of the geff store to load the detections from.

    Returns:
        list[byotrack.SegmentationDetections]: The (first) per-frame detections attached to the
            geff store.

    Raises:
        FileNotFoundError: If the geff store has no related object of type ``"labels"``.

    """
    related_objects = geff.GeffReader(path).metadata.related_objects

    if related_objects is None:  # pragma: no cover
        raise FileNotFoundError("The geff store has no labels (detections) related objects.")

    for related_object in related_objects:
        if related_object.type != "labels":
            continue

        return load_detections_from_zarr(path / related_object.path)

    raise FileNotFoundError("The geff store has no labels (detections) related objects.")
