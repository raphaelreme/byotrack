from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import napari  # type: ignore[import-untyped]
import napari.layers  # type: ignore[import-untyped]
import napari.utils.colormaps  # type: ignore[import-untyped]
import numpy as np

import byotrack
from byotrack.napari.utils import (
    _detections_to_labels,
    _find_paths_in_grid,
    _initialize_grid,
    _LazySegmentationArray,
    _LazyVideoArray,
    detections_to_napari_points,
    detections_to_napari_segmentation,
    precompute_optical_flow,
    tracks_to_napari_tracks,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence


def _find_dim(
    video: Sequence[np.ndarray] | np.ndarray = (),
    detections_sequence: Sequence[byotrack.Detections] = (),
    tracks: Collection[byotrack.Track] = (),
) -> int:
    """Check and return the dimension of the visualization.

    Ensures that the video, detections and tracks are consistent with each other (when given) and
    infers the spatial dimension (2 or 3) of the data to visualize.

    Args:
        video (Sequence[np.ndarray] | np.ndarray): Optional sequence of T frames (array).
            Each array is expected to have a shape ([D, ]H, W, C)
            Default: () (No video)
        detections_sequence (Sequence[byotrack.Detections]): Optional sequence of detections (one per frame).
            Default: () (No detections)
        tracks (Collection[byotrack.Track]): Optional tracks.
            Default: () (No tracks)

    Returns:
        int: The dimension (2 or 3) shared by the given video, detections and tracks. 0 if none is given.

    Raises:
        ValueError: If the given video, detections and tracks do not share the same dimension.

    """
    dim = 0
    if byotrack.video.video_length(video):
        dim = len(byotrack.video.video_shape(video)) - 2  # Exclude time & channels

    if detections_sequence:
        dim_ = detections_sequence[0].dim
        if dim and dim != dim_:
            raise ValueError(f"Frames are {dim}D, but detections are {dim_}D.")

        dim = dim_

    if tracks:
        dim_ = next(iter(tracks)).dim

        if dim and dim != dim_:
            data = "frames" if len(video) else "detections"
            raise ValueError(f"Tracks is {dim}D, but the {data} are {dim_}D.")

        dim = dim_

    if dim == 0:
        raise ValueError("Please provide at least one of `video`, `detections_sequence` or `tracks`.")

    return dim


def add_video(
    viewer: napari.Viewer,
    video: Sequence[np.ndarray] | np.ndarray,
    *,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rgb=True,
    lazy: bool = False,
) -> None:
    """Add a video to a Napari viewer.

    Args:
        viewer (napari.Viewer): Napari viewer to add the video to.
        video (Sequence[np.ndarray] | np.ndarray): Sequence of T frames (array).
            Each array is expected to have a shape ([D, ]H, W, C)
        anisotropy (tuple[float, float, float]): Spatial anisotropy ([Z, ]Y, X) used to scale the layer.
            Default: (1.0, 1.0, 1.0)
        rgb (bool): If True and the video has 3 or 4 channels, add it as a single RGB(A) layer.
            Otherwise, add one grayscale layer per channel.
            Default: True
        lazy (bool): If True, frames are read from `video` on demand (one frame at a time) instead of
            loading the whole video into memory upfront. Requires `video` to be a `byotrack.Video`.
            Please be aware, that for advanced napari usage (e.g. swapped temporal axis), this may fail.
            Default: False

    """
    video_: _LazyVideoArray | np.ndarray
    if lazy:
        if not isinstance(video, byotrack.Video):
            raise TypeError(
                f"lazy=True requires a byotrack.Video instance (got {type(video).__name__}). "
                "Construct one from a path (`byotrack.Video(path)`) to get a lazy reader."
            )
        video_ = _LazyVideoArray(video)
    else:
        video_ = np.asarray(video)

    dim = video_.ndim - 2  # Exclude time and channels
    axis_labels: tuple[str, ...] = ("Time", "Depth", "Height", "Width") if dim == 3 else ("Time", "Height", "Width")  # noqa: PLR2004
    scale = (1.0, *anisotropy[-dim:])  # Add the temporal scale

    if rgb and video_.shape[-1] in (3, 4):  # RGB only supports 3 or 4 channels
        viewer.add_image(video_, name="Video (RGB)", axis_labels=axis_labels, scale=scale, rgb=True)
    else:
        layers = viewer.add_image(video_, channel_axis=-1, axis_labels=axis_labels, scale=scale)
        for i, layer in enumerate(layers):
            layer.name = f"Video (Ch. {i})"


def add_detections(
    viewer: napari.Viewer,
    detections_sequence: Sequence[byotrack.DetectionsLike] = (),
    *,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    detections_mode: Literal["segmentation", "points"] = "segmentation",
    detection_size=10.0,
    color_from_labels: bool = True,
    lazy: bool = False,
) -> None:
    """Add detections to a Napari viewer.

    Args:
        viewer (napari.Viewer): Napari viewer to add the detections to.
        detections_sequence (Sequence[byotrack.DetectionsLike]): Sequence of T detections (one per
            frame, sorted in time).
            Default: () (No detections)
        anisotropy (tuple[float, float, float]): Spatial anisotropy ([Z, ]Y, X) used to scale the layer.
            Default: (1.0, 1.0, 1.0)
        detections_mode (Literal["segmentation", "points"]): Whether to display detections as an
            instance segmentation (labels layer) or as points (points layer).
            Default: "segmentation"
        detection_size (float): Size of the points, when `detections_mode` is "points".
            Default: 10.0
        color_from_labels (bool): Use `labels` from `Detections` to assign a color to each segmentation.
            If False, it will use the detection identifier (0 to N-1).
            Default: True
        lazy (bool): If True in "segmentation" mode, detections are read from `detections_sequence` on demand
            (one frame at a time) instead of loading the whole segmentation into memory upfront.
            Please be aware, that for advanced napari usage (e.g. label edition, swapped temporal axis), this may fail.
            Default: False

    """
    detections_sequence_ = [byotrack.as_detections(detections) for detections in detections_sequence]
    dim = detections_sequence_[0].dim
    axis_labels: tuple[str, ...] = ("Time", "Depth", "Height", "Width") if dim == 3 else ("Time", "Height", "Width")  # noqa: PLR2004
    scale = (1.0, *anisotropy[-dim:])  # Add the temporal scale

    if detections_mode == "segmentation":
        segmentation = (
            _LazySegmentationArray(detections_sequence_, color_from_labels=color_from_labels)
            if lazy
            else detections_to_napari_segmentation(detections_sequence_, color_from_labels=color_from_labels)
        )
        viewer.add_layer(napari.layers.Labels(segmentation, name="Segmentations", axis_labels=axis_labels, scale=scale))
    else:
        points = detections_to_napari_points(detections_sequence_)
        labels = _detections_to_labels(detections_sequence_, color_from_labels=color_from_labels)
        viewer.add_layer(
            napari.layers.Points(
                points,
                name="Detections",
                size=detection_size,
                face_color=napari.utils.colormaps.label_colormap().map(labels),
                axis_labels=axis_labels,
                scale=scale,
                opacity=0.70,
            )
        )


def add_tracks(
    viewer: napari.Viewer,
    tracks: Collection[byotrack.Track],
    *,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    track_width=5.0,
) -> None:
    """Add tracks to a Napari viewer.

    Adds a points layer for the tracked points at the current frame, and a tracks layer showing
    the trails of each track (including splits and merges).

    Args:
        viewer (napari.Viewer): Napari viewer to add the tracks to.
        tracks (Collection[byotrack.Track]): Tracks to display.
        anisotropy (tuple[float, float, float]): Spatial anisotropy ([Z, ]Y, X) used to scale the layer.
            Default: (1.0, 1.0, 1.0)
        track_width (float): Size of the tracked points and width of the track trails.
            Default: 5.0

    """
    dim = next(iter(tracks)).dim
    axis_labels: tuple[str, ...] = ("Time", "Depth", "Height", "Width") if dim == 3 else ("Time", "Height", "Width")  # noqa: PLR2004
    scale = (1.0, *anisotropy[-dim:])  # Add the temporal scale

    points, parents, lineage_ids = tracks_to_napari_tracks(tracks)
    viewer.add_layer(
        napari.layers.Points(
            points[:, 1:],
            name="Tracked points",
            size=track_width,
            axis_labels=axis_labels,
            scale=scale,
            blending="additive",
        )
    )
    viewer.add_layer(
        napari.layers.Tracks(
            points,
            name="Tracks",
            graph=parents,
            features={"time": points[:, 1], "lineage_ids": lineage_ids},
            tail_width=track_width,
            axis_labels=axis_labels,
            scale=scale,
        )
    )


def add_optical_flow(  # noqa: C901, PLR0913
    viewer: napari.Viewer,
    video: Sequence[np.ndarray] | np.ndarray,
    optflow: byotrack.OpticalFlow,
    *,
    grid_step: int = 20,
    display_grid: bool = True,
    size: int = 5,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    forward_flows: Sequence[np.ndarray] = (),
    backward_flows: Sequence[np.ndarray] = (),
) -> None:
    """Add an optical flow grid visualization to a Napari viewer.

    Displays a deformable grid of points (and optionally the edges linking them) that is warped by
    the optical flow as the temporal slider is scrubbed, forward or backward. If the flow maps are
    not (fully) precomputed, they are first pre-computed with `_precompute_optical_flow`.

    Press 'g' to reset the grid to a uniform distribution at the current frame.

    Args:
        viewer (napari.Viewer): Napari viewer to add the visualization to.
        video (Sequence[np.ndarray] | np.ndarray): Sequence of T frames (array).
            Each array is expected to have a shape ([D, ]H, W, C)
        optflow (byotrack.OpticalFlow): Optical flow algorithm used to (pre)compute and apply the flow.
        grid_step (int): Spacing (in scaled/world units) between two consecutive grid control points.
            Default: 20
        display_grid (bool): If True, also draw the edges linking the grid points (wireframe).
            Note that this may slow down the visualization with large grids.
            Default: True
        size (int): Size of the grid points.
            Default: 5
        anisotropy (tuple[float, float, float]): Spatial anisotropy ([Z, ]Y, X) used to scale the layers.
            Default: (1.0, 1.0, 1.0)
        forward_flows (Sequence[np.ndarray]): Precomputed flow maps from frame t to frame t+1, for each
            of the T - 1 consecutive frame pairs. If shorter than `video`, all flows are recomputed.
            Default: () (Flows are computed from `video`)
        backward_flows (Sequence[np.ndarray]): Precomputed flow maps from frame t+1 to frame t, for each
            of the T - 1 consecutive frame pairs. If shorter than `video`, all flows are recomputed.
            Default: () (Flows are computed from `video`)

    """
    shape = byotrack.video.video_shape(video)

    if len(forward_flows) < shape[0] - 1 or len(backward_flows) < shape[0] - 1:
        forward_flows, backward_flows = precompute_optical_flow(video, optflow)

    dim = len(shape) - 2  # Exclude time & channels
    axis_labels: tuple[str, ...] = ("Time", "Depth", "Height", "Width") if dim == 3 else ("Time", "Height", "Width")  # noqa: PLR2004
    scale = (1.0, *anisotropy[-dim:])

    # Initialize the grid points and edges
    grid = _initialize_grid(shape[1:-1], grid_step, scale[1:])
    grid[..., 0] = viewer.dims.current_step[0]
    paths = _find_paths_in_grid(grid.shape[:-1])
    grid = grid.reshape(-1, grid.shape[-1])

    if display_grid:
        lines = viewer.add_layer(
            napari.layers.Shapes(
                [grid[paths[i]] for i in range(len(paths))],
                name="Optical Flow Deformation Grid (edges)",
                shape_type="path",
                axis_labels=axis_labels,
                scale=scale,
            )
        )

    points = viewer.add_layer(
        napari.layers.Points(
            grid.copy(),
            name="Optical Flow Deformation Grid (vertices)",
            size=size,
            axis_labels=axis_labels,
            scale=scale,
            blending="additive",
            face_color="red",
        )
    )

    def on_step_change(event) -> None:
        """Warp the grid points (and edges) to the new frame reached by the temporal slider."""
        old_frame_id = int(points.data[0, 0])
        new_frame_id = int(event.value[0])

        if new_frame_id == old_frame_id:
            return  # Moved along depth/height/width but not time

        pts: np.ndarray = points.data[:, 1:]

        if new_frame_id > old_frame_id:
            for frame_id in range(old_frame_id, new_frame_id):
                pts = optflow.transform(forward_flows[frame_id], pts)

        elif new_frame_id < old_frame_id:
            for frame_id in range(old_frame_id - 1, new_frame_id - 1, -1):
                pts = optflow.transform(backward_flows[frame_id], pts)

        points.data[:, 0] = new_frame_id
        points.data[:, 1:] = pts

        if display_grid:
            # Slow because of the the shape computations in the setters
            lines.data = [points.data[paths[i]] for i in range(len(paths))]
            lines.refresh()

        points.refresh()

    def reset_grid(viewer: napari.Viewer) -> None:
        """Reset the grid to a uniform distribution at the current frame (key binding: 'g')."""
        points.data = grid.copy()
        points.data[:, 0] = viewer.dims.current_step[0]

        if display_grid:
            lines.data = [points.data[paths[i]] for i in range(len(paths))]
            lines.refresh()

        points.refresh()

    viewer.dims.events.current_step.connect(on_step_change)
    viewer.bind_key("g", reset_grid)


def visualize(  # noqa: PLR0913
    video: Sequence[np.ndarray] | np.ndarray = (),
    detections_sequence: Sequence[byotrack.DetectionsLike] = (),
    tracks: Collection[byotrack.Track] = (),
    *,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rgb=True,
    lazy: bool = False,
    detections_mode: Literal["segmentation", "points"] = "segmentation",
    detection_size=10.0,
    color_from_labels: bool = True,
    track_width=5.0,
    run=True,
) -> napari.Viewer:
    """Open a Napari viewer to visualize a video, detections and/or tracks.

    Any combination of `video`, `detections_sequence` and `tracks` can be given (at least one is
    required). When several are given, they are expected to share the same dimension (2D or 3D).

    Args:
        video (Sequence[np.ndarray] | np.ndarray): Optional sequence of T frames (array).
            Each array is expected to have a shape ([D, ]H, W, C)
            Default: () (No video)
        detections_sequence (Sequence[byotrack.DetectionsLike]): Optional sequence of T detections
            (one per frame, sorted in time).
            Default: () (No detections)
        tracks (Collection[byotrack.Track]): Optional tracks.
            Default: () (No tracks)
        anisotropy (tuple[float, float, float]): Spatial anisotropy ([Z, ]Y, X) used to scale the layers.
            Default: (1.0, 1.0, 1.0)
        rgb (bool): If True and the video has 3 or 4 channels, add it as a single RGB(A) layer.
            Otherwise, add one grayscale layer per channel.
            Default: True
        lazy (bool): If True, video frames and detections are read on demand instead of loading the whole video
            into memory upfront. Requires `video` to be a `byotrack.Video`. See `add_video`, `add_detections`.
            Please be aware, that for advanced napari usage (e.g. label edition, swapped temporal axis), this may fail.
            Default: False
        detections_mode (Literal["segmentation", "points"]): Whether to display detections as an
            instance segmentation (labels layer) or as points (points layer).
            Default: "segmentation"
        detection_size (float): Size of the points, when `detections_mode` is "points".
            Default: 10.0
        color_from_labels (bool): Use `labels` from `Detections` to assign a color to each segmentation.
            If False, it will use the detection identifier (0 to N-1).
            Default: True
        track_width (float): Size of the tracked points and width of the track trails.
            Default: 5.0
        run (bool): If True, blocks and starts the Napari Qt event loop (``napari.run()``).
            Default: True

    Returns:
        napari.Viewer: The created Napari viewer.

    """
    dim = _find_dim(video, [byotrack.as_detections(detections) for detections in detections_sequence], tracks)
    axis_labels: tuple[str, ...] = ("Time", "Depth", "Height", "Width") if dim == 3 else ("Time", "Height", "Width")  # noqa: PLR2004

    viewer = napari.Viewer(title="ByoTrack x Napari", axis_labels=axis_labels, ndisplay=dim)

    if byotrack.video.video_length(video):
        add_video(viewer, video, anisotropy=anisotropy, rgb=rgb, lazy=lazy)

    if detections_sequence:
        add_detections(
            viewer,
            detections_sequence,
            anisotropy=anisotropy,
            detections_mode=detections_mode,
            detection_size=detection_size,
            color_from_labels=color_from_labels,
            lazy=lazy,
        )

    if tracks:
        add_tracks(viewer, tracks, anisotropy=anisotropy, track_width=track_width)

    viewer.dims.point = (0.0, *viewer.dims.point[1:])  # Set temporal axis at the initial frame (0)

    if run:
        napari.run()

    return viewer


# XXX: No lazy option != from older InteractiveFlowVisualizer
def visualize_flow_deformation(  # noqa: PLR0913
    video: Sequence[np.ndarray] | np.ndarray,
    optflow: byotrack.OpticalFlow,
    *,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    grid_step: int = 20,
    display_grid: bool = True,
    size: int = 5,
    forward_flows: Sequence[np.ndarray] = (),
    backward_flows: Sequence[np.ndarray] = (),
    rgb=True,
    run=True,
) -> napari.Viewer:
    """Open a Napari viewer to visualize a video together with its optical flow deformation grid.

    Args:
        video (Sequence[np.ndarray] | np.ndarray): Sequence of T frames (array).
            Each array is expected to have a shape ([D, ]H, W, C)
        optflow (byotrack.OpticalFlow): Optical flow algorithm used to (pre)compute and apply the flow.
        grid_step (int): Spacing (in scaled/world units) between two consecutive grid control points.
            Default: 20
        display_grid (bool): If True, also draw the edges linking the grid points (wireframe).
            Default: True
        size (int): Size of the grid points.
            Default: 5
        anisotropy (tuple[float, float, float]): Spatial anisotropy ([Z, ]Y, X) used to scale the layers.
            Default: (1.0, 1.0, 1.0)
        forward_flows (Sequence[np.ndarray]): Precomputed flow maps from frame t to frame t+1, for each
            of the T - 1 consecutive frame pairs. If shorter than `video`, all flows are recomputed.
            Default: () (Flows are computed from `video`)
        backward_flows (Sequence[np.ndarray]): Precomputed flow maps from frame t+1 to frame t, for each
            of the T - 1 consecutive frame pairs. If shorter than `video`, all flows are recomputed.
            Default: () (Flows are computed from `video`)
        rgb (bool): If True and the video has 3 or 4 channels, add it as a single RGB(A) layer.
            Otherwise, add one grayscale layer per channel.
            Default: True
        run (bool): If True, blocks and starts the Napari Qt event loop (``napari.run()``).
            Default: True

    Returns:
        napari.Viewer: The created Napari viewer.

    """
    dim = len(byotrack.video.video_shape(video)) - 2  # Exclude time & channels
    axis_labels: tuple[str, ...] = ("Time", "Depth", "Height", "Width") if dim == 3 else ("Time", "Height", "Width")  # noqa: PLR2004

    viewer = napari.Viewer(title="ByoTrack x Napari", axis_labels=axis_labels, ndisplay=dim)

    add_video(viewer, video, anisotropy=anisotropy, rgb=rgb)
    viewer.dims.point = (0.0, *viewer.dims.point[1:])  # Set temporal axis at the initial frame (0)

    add_optical_flow(
        viewer,
        video,
        optflow,
        grid_step=grid_step,
        display_grid=display_grid,
        size=size,
        anisotropy=anisotropy,
        forward_flows=forward_flows,
        backward_flows=backward_flows,
    )

    if run:
        napari.run()

    return viewer
