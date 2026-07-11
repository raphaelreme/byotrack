"""IOs for CTC datasets [10].

We provide loading and saving functions for detections and tracks.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import TYPE_CHECKING

import numpy as np
import tifffile  # type: ignore[import-untyped]
import torch
import tqdm.auto as tqdm

import byotrack
import byotrack.utils
import byotrack.video.reader
from byotrack.api.detections.detections import draw_disk_2d, draw_disk_3d, fast_relabel, labels_of, relabel_consecutive
from byotrack.api.detections.segmentation_detections import _position_from_segmentation

if TYPE_CHECKING:
    import os
    from collections.abc import Collection, Sequence


def _parse_meta_data(file: pathlib.Path) -> dict[int, tuple[int, int, int]]:
    """Parse the CTC meta_data (res_track.txt or man_track.txt) into a dict."""
    meta: dict[int, tuple[int, int, int]] = {}

    for line in file.read_text(encoding="utf-8").strip().split("\n"):
        identifier, start, end, parent = (int(word) for word in line.strip().split())
        meta[identifier - 1] = (start, end, parent)

    return meta


def load_detections(path: str | os.PathLike) -> list[byotrack.Detections]:
    """Load detections stored in the CTC format [10].

    The CTC format for detections consists of one tiff file for each frame
    which contains the instance segmentation on the frame.

    See the official documentation of the CTC format at
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf

    This will simply run the byotrack.GroundTruthDetector on the given folder to load the detection
    in memory.

    Args:
        path (str | os.PathLike): Path to the detections data

    Returns:
        list[byotrack.Segmentation]: Loaded detections
    """
    return byotrack.GroundTruthDetector().run(byotrack.Video(path))


def load_tracks(  # noqa: C901, PLR0912, PLR0915
    path: str | os.PathLike,
) -> list[byotrack.Track]:
    """Load tracks stored in the CTC format [10].

    The CTC format for tracks consists of one tiff file for each frame which contains the segmentation
    of active tracks on the frame and a text file containing track ids, start and end frames and parent track.

    First the code parses the segmentation tiff files (either "man*{frame_id}.tif" or "mask{frame_id}.tif")
    and recovers all the known positions (plus the associated detections_ids) of the tracks. Then it parses
    the metadata in the txt file (either "man_track.txt" or "res_track.txt") and validate the tracks creation.

    See the official documentation of the CTC format at
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf

    Example:

    .. code-block:: python

        import byotrack
        import byotrack.visualize
        from byotrack.dataset import ctc

        # Load the video and normalize it
        video = byotrack.Video("dataset/01")  # Load videos
        video = video.normalize()

        # Optionally, load ground-truth segmentations (may take a lot of RAM)
        detections_sequence = ctc.load_detections("dataset/01_GT/TRA")

        # Load ground-truth tracks
        tracks = ctc.load_tracks("dataset/01_GT/TRA")

        # Visualize everything
        byotrack.visualize.InteractiveVisualizer(video, detections_sequence, tracks)


    Args:
        path (str | os.PathLike): Path to the tracks data

    Returns:
        list[byotrack.Track]: Loaded tracks

    """
    path = pathlib.Path(path)
    is_res = False
    tracks_meta: dict[int, tuple[int, int, int]] = {}

    if (path / "res_track.txt").exists():
        is_res = True
        tracks_meta = _parse_meta_data(path / "res_track.txt")
    elif (path / "man_track.txt").exists():
        tracks_meta = _parse_meta_data(path / "man_track.txt")
    else:
        # We can load without it, but not resolve the dividing of tracks
        warnings.warn("res_track.txt or man_track.txt not found. Loading will be done without it.", stacklevel=2)

    # Dict containing for each track_id, the list of frames found for the track, the associated det_id and its position
    tracks_data: dict[int, tuple[list[int], list[int], list[np.ndarray]]] = {}

    # Let's iterate through any found segmentation
    segmentation_paths = path.glob("mask*.tif") if is_res else path.glob("man_*.tif")
    loader = byotrack.video.reader.FrameTiffLoader()

    for path_ in tqdm.tqdm(byotrack.utils.sorted_alphanumeric(segmentation_paths), desc="Loading CTC tracks"):
        if is_res:
            frame_id = int(path_.stem[len("mask") :])
        elif "seg" in path.stem:
            frame_id = int(path_.stem[len("man_seg") :])
        else:
            frame_id = int(path_.stem[len("man_track") :])

        frame = loader(path_)

        if frame.shape[-1] != 1:
            raise ValueError("Multichannel segmentation are not supported")
        frame = frame[..., 0]

        # Compute the mapping done by relabel consecutive
        track_ids: list[int] = labels_of(frame).tolist()
        positions = _position_from_segmentation(relabel_consecutive(frame, inplace=True))

        for det_id, (track_id, position) in enumerate(zip(track_ids, positions, strict=True)):
            if track_id not in tracks_data:
                tracks_data[track_id] = ([], [], [])

            tracks_data[track_id][0].append(frame_id)
            tracks_data[track_id][1].append(det_id)
            tracks_data[track_id][2].append(position)

    # Build tracks
    tracks = []
    for track_id, data in tracks_data.items():
        if track_id not in tracks_meta and tracks_meta:
            warnings.warn(f"Missing identifier {track_id} in the txt metadata.", stacklevel=2)

        if track_id in tracks_meta:
            start = tracks_meta[track_id][0]
            last = tracks_meta[track_id][1]
            parent = tracks_meta[track_id][2] - 1  # Offset of 1 in identifiers

            if min(data[0]) < start:
                raise ValueError(f"Found track {track_id} on frame {min(data[0])} before for it started")

            if max(data[0]) > last:
                raise ValueError(f"Found track {track_id} on frame {max(data[0])} after for it ended")
        else:
            start, last = min(data[0]), max(data[0])
            parent = -1

        known_frames = torch.tensor(data[0]) - start
        known_detection_ids = torch.tensor(data[1], dtype=torch.int32)
        known_positions = torch.tensor(np.array(data[2]), dtype=torch.float32)

        points = torch.full((last - start + 1, known_positions.shape[1]), torch.nan)
        detection_ids = torch.full((last - start + 1,), -1, dtype=torch.int32)

        detection_ids[known_frames] = known_detection_ids
        points[known_frames] = known_positions

        tracks.append(byotrack.Track(start, points, track_id, detection_ids, parent_id=parent))

    byotrack.Track.check_tracks(tracks, warn=True)

    return tracks


def save_detections(
    detections_sequence: Sequence[byotrack.Detections],
    path: str | os.PathLike,
    *,
    as_res=True,
    as_seg=False,
    n_digit=4,
) -> None:
    """Save detections in the CTC format [10].

    It will save one tiff image for each frame containing the segmentation of objects.

    See the official documentation of the CTC format at
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf

    Args:
        path (str | os.PathLike): Folder path where to store the .tif files
        detections_sequence (Sequence[byotrack.Detections]): Detections for each frame
        as_res (bool): Whether to store as a results or as a ground-truth.
            Ground-truth are stored as "man_trackT.tif"
            Results as "maskT.tif"
            Default: True
        as_seg (bool): Only for ground-truth: file names are "man_segT.tif" instead of "man_trackT.tif"
            Default: False
        n_digit (int): Number of digit used to encode time in file names.
            Default: 4

    """
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for frame_id, detections in enumerate(tqdm.tqdm(detections_sequence, desc="Saving Detections to CTC")):
        segmentation = detections.segmentation.numpy().astype(np.uint16)

        if as_res:
            name = f"mask{frame_id:0{n_digit}}.tif"
        elif as_seg:
            name = f"man_seg{frame_id:0{n_digit}}.tif"
        else:
            name = f"man_track{frame_id:0{n_digit}}.tif"

        if detections.dim == 2:  # Probably not required here, but let's be precise just in case  # noqa: PLR2004
            segmentation = segmentation[None, None, None, ..., None]  # ImageJ tiff format: TZCYXS
        else:
            segmentation = segmentation[None, :, None, ..., None]  # ImageJ tiff format: TZCYXS

        tifffile.imwrite(path / name, segmentation, imagej=True, compression="zlib")


def _save_metadata(tracks: Collection[byotrack.Track], path: pathlib.Path) -> None:
    """Saves tracks metadata in the given file."""
    lines = []
    for track in tracks:
        if track.merge_id != -1:
            warnings.warn("CTC format do not support merge events. Merge events are removed.", stacklevel=2)

        lines.append(
            " ".join(
                [
                    str(track.identifier + 1),  # Offset of 1 for track id as 0 is not valid for CTC
                    str(track.start),
                    str(track.start + len(track) - 1),
                    str(track.parent_id + 1),  # Offset of 1 for identifiers
                ]
            )
        )

    path.write_text("\n".join(lines))


def _build_radii(n: int, dim: int, radius: float, anisotropy: float) -> np.ndarray:
    """Build a constant radii array accounting for anisotropy."""
    radii = np.full((n, dim), radius, dtype=np.float32)
    if dim == 3:  # noqa: PLR2004
        radii[:, -1] /= anisotropy

    return radii


def save_tracks(  # noqa: C901, PLR0912, PLR0913, PLR0915
    tracks: Collection[byotrack.Track],
    path: str | os.PathLike,
    *,
    detections_sequence: Sequence[byotrack.Detections] = (),
    as_res=True,
    as_seg=False,
    default_radius=3.0,
    last=0,
    shape: tuple[int, ...] | None = None,
    n_digit=4,
    anisotropy=1.0,
    overwrite_detections=False,
) -> None:
    """Save tracks in the CTC format [10].

    It will save one tiff image for each frame containing the segmentation of objects and a metadata txt file
    describing the tracks identifiers, start/end frames and parents.

    Parent information is not supported yet.

    When no detections_sequence is given, tracks segmentations are simply drawn as disk with `default_radius`
    at the track localization.
    When detections_sequence is given, then for tracks without detections associated, a disk is drawn
    with `default_radius` (set at 0 to drop this behavior), otherwise the detection segmentation is used.

    For smarter behaviors, one can directly modify the segmentation before saving.

    See the official documentation of the CTC format at
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf

    Args:
        path (str | os.PathLike): Folder path where to store the .tif files
        tracks (Collection[byotrack.Track]): Tracks to save
        detections_sequence (Sequence[byotrack.Detections]): Optional detections for each frame
            Default: ()
        as_res (bool): Whether to store as a results or as a ground-truth.
            Ground-truth are stored as "man_trackT.tif" and "man_track.txt"
            Results as "maskT.tif" and "res_track.txt"
            Default: True
        as_seg (bool): Only for ground-truth: file names are "man_segT.tif" instead of "man_trackT.tif"
            Note that it will also store the meta data to allow reloading the tracks.
            Default: False
        default_radius (float): Radius of drawn disk when no segmentation is available.
            Default: 5.0 (pixels)
        last (int): Overwrite last frame to consider
            Default: 0 (Will compute it from the last tracked particles)
        shape (tuple[int, ...] | None): Optional shape. Required when no detections_sequence is provided
            Default: None
        n_digit (int): Number of digit used to encode time in file names.
            Default: 4
        anisotropy (float): Relative size of a pixel along the depth dimension
            versus height/width dimensions.
            Default: 1.0
        overwrite_detections (bool): Overwrite the segmentation of objects with disk.
            Default: False (Disk are only drawn on background)

    """
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if as_res:
        _save_metadata(tracks, path / "res_track.txt")
    else:
        _save_metadata(tracks, path / "man_track.txt")

    if not last:
        last = max(track.start + len(track) - 1 for track in tracks)

    if detections_sequence:
        if shape and detections_sequence[0].shape != shape:
            raise ValueError("Given shape is not compatible with provided Detections")
        shape = detections_sequence[0].shape
    elif shape is None:
        raise ValueError("Without `detections_sequence`, `shape` should be provided")

    for frame_id in tqdm.trange(last + 1, desc="Saving tracks to CTC"):
        has_detections = len(detections_sequence) > frame_id

        disk_positions = []
        disk_ids = []

        if has_detections:
            det_to_track_ids = np.zeros(detections_sequence[frame_id].length, dtype=np.uint16) - 1

        for track in tracks:
            position = track[frame_id]

            if torch.isnan(position).any():  # Track not defined
                if track.start <= frame_id < track.start + len(track):
                    warnings.warn(
                        "Found a missing position inside a track segment. This is not supported by CTC software."
                        "Consider filling holes with Interpolators.",
                        stacklevel=2,
                    )
                continue

            if has_detections:
                detection_id = track.detection_ids[frame_id - track.start]
                if detection_id != -1:
                    det_to_track_ids[detection_id] = track.identifier
                    continue

            disk_positions.append(position)
            disk_ids.append(track.identifier)

        if has_detections:
            segmentation = detections_sequence[frame_id].segmentation.clone().numpy().astype(np.uint16)
            fast_relabel(segmentation, det_to_track_ids)
        else:
            segmentation = np.zeros(shape, dtype=np.uint16)

        # Add circle to the segmentation
        if disk_ids:
            labels = set(labels_of(segmentation).tolist())

            draw_disk = draw_disk_3d if segmentation.ndim == 3 else draw_disk_2d  # noqa: PLR2004

            draw_disk(
                segmentation,
                torch.stack(disk_positions).numpy(),
                _build_radii(len(disk_ids), segmentation.ndim, default_radius, anisotropy=anisotropy),
                np.array(disk_ids, dtype=np.uint16),
                overwrite=overwrite_detections,
            )

            # Safety checks because CTC is quite restrictive
            new_labels = set(labels_of(segmentation).tolist())
            diff = labels.difference(new_labels)
            if diff:
                warnings.warn(
                    f"Some tracks were fully occluded: {diff}. This will induce"
                    " a missing position in the track segment which is not supported by CTC software."
                    " Consider decreasing the radius of track disks, or adding manually their segmentations.",
                    stacklevel=2,
                )

            diff = {identifier for identifier in disk_ids if identifier not in new_labels}
            if diff:
                warnings.warn(
                    f"The disk of some added tracks are not found (outside of image or occluded): {diff}."
                    " This will induce a missing position in the track segment which is not supported by CTC software."
                    " Fixing this can be done by changing the radius, or manually dropping/correcting the track.",
                    stacklevel=2,
                )

        # Save a tiff
        if len(shape) == 2:  # noqa: PLR2004
            segmentation = segmentation[None, None, None, ..., None]  # ImageJ tiff format: TZCYXS
        else:
            segmentation = segmentation[None, :, None, ..., None]  # ImageJ tiff format: TZCYXS

        if as_res:
            name = f"mask{frame_id:0{n_digit}}.tif"
        elif as_seg:
            name = f"man_seg{frame_id:0{n_digit}}.tif"
        else:
            name = f"man_track{frame_id:0{n_digit}}.tif"

        tifffile.imwrite(path / name, segmentation, imagej=True, compression="zlib")
