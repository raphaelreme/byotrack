"""IOs for CTC datasets [10]

We provide loading and saving functions for detections and tracks.
"""

import os
import pathlib
from typing import Collection, Dict, List, Optional, Union, Sequence, Tuple
import warnings

import numba  # type: ignore
import numpy as np
import tifffile  # type: ignore
import torch
import tqdm.auto as tqdm

import byotrack
import byotrack.utils
import byotrack.video.reader
from byotrack.api.detector.detections import _fast_unique, _position_from_segmentation, relabel_consecutive


class GroundTruthDetector(byotrack.BatchDetector):
    """Converts a 'ground-truth' video of segmentation into byotrack Detections

    The video should not be normalized and each pixel is expected to be an integer.

    Example:

    .. code-block:: python

        # Load the segmentation video, it can be directly loaded from any folder containing tiff images
        video = byotrack.Video("dataset/01_ERR_SEG")  # Load segmentation for CLB
        # video = byotrack.Video("dataset/01_GT/SEG")  # Load ground-truth segmentation
        # video = byotrack.Video("dataset/01_RES/TRA")  # Load predicted tracks segmentation

        detector = GroundTruthDetector()

        detections_sequence = detector.run(video)

    """

    progress_bar_description = "Detections (Load from CTC format)"

    def detect(self, batch: np.ndarray) -> List[byotrack.Detections]:
        assert batch.shape[-1] == 1, "Multichannel segmentation are not supported"
        assert np.issubdtype(batch.dtype, np.integer)

        detections_sequence = []
        for frame in batch:
            detections_sequence.append(
                byotrack.Detections({"segmentation": torch.tensor(frame[..., 0], dtype=torch.int32)})
            )

        return detections_sequence


def _parse_meta_data(file: pathlib.Path) -> Dict[int, Tuple[int, int, int]]:
    """Parse the CTC meta_data (res_track.txt or man_track.txt) into a Dict"""
    meta: Dict[int, Tuple[int, int, int]] = {}

    for line in file.read_text(encoding="utf-8").strip().split("\n"):
        identifier, start, end, parent = (int(word) for word in line.strip().split())
        meta[identifier - 1] = (start, end, parent)

    return meta


def load_tracks(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    path: Union[str, os.PathLike],
) -> List[byotrack.Track]:
    """Load saved tracks at the CTC format [10]

    The CTC format for tracks consists of one tiff file for each frame which contains the segmentation
    of active tracks on the frame and a text file containing track ids, start and end frames and parent track.

    First the code parses the segmentation tiff files (either "man*{frame_id}.tif" or "mask{frame_id}.tif")
    and recovers all the known positions (plus the associated detections_ids) of the tracks. Then it parses
    the metadata in the txt file (either "man_track.txt" or "res_track.txt") and validate the tracks creation.

    See the official documentation of CTC at
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf

    Tracks with parent are not supported yet.

    Example:

    .. code-block:: python

        import byotrack
        import byotrack.visualize
        from byotrack.dataset import ctc

        # Load the video and normalize it
        video = byotrack.Video("dataset/01")  # Load videos
        video.set_transform(byotrack.VideoTransformConfig(aggregate=True, normalize=True))

        # Optionnally, load ground-truth segmentations (may take a lot of RAM)
        detections_sequence = ctc.GroundTruthDetector().run(byotrack.Video("dataset/01_GT/TRA"))

        # Load ground-truth tracks
        tracks = ctc.load_tracks("dataset/01_GT/TRA")

        # Visualize everything
        byotrack.visualize.InteractiveVisualizer(video, detections_sequence, tracks)


    Args:
        path (Union[str, os.PathLike]): Path to the tracks data

    Returns:
        List[byotrack.Track]: Saved tracks

    """

    path = pathlib.Path(path)
    is_res = False
    tracks_meta: Dict[int, Tuple[int, int, int]] = {}

    if (path / "res_track.txt").exists():
        is_res = True
        tracks_meta = _parse_meta_data(path / "res_track.txt")
    elif (path / "man_track.txt").exists():
        tracks_meta = _parse_meta_data(path / "man_track.txt")
    else:
        # We can load without it, but not resolve the dividing of tracks
        warnings.warn("res_track.txt or man_track.txt not found. Loading will be done without it.")

    # Dict containing for each track_id, the list of frames found for the track, the associated det_id and its position
    tracks_data: Dict[int, Tuple[List[int], List[int], List[np.ndarray]]] = {}

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

        assert frame.shape[-1] == 1, "Multichannel segmentation are not supported"
        frame = frame[..., 0]

        # Compute the mapping done by relabel consecutive
        unique = _fast_unique(frame)[1:] - 1  # Remove the background (0)
        positions = _position_from_segmentation(relabel_consecutive(frame, inplace=True))

        for det_id, (track_id, position) in enumerate(zip(unique, positions)):
            if track_id not in tracks_data:
                tracks_data[track_id] = ([], [], [])

            tracks_data[track_id][0].append(frame_id)
            tracks_data[track_id][1].append(det_id)
            tracks_data[track_id][2].append(position)

    # Build tracks
    tracks = []
    for track_id, data in tracks_data.items():
        if track_id not in tracks_meta and tracks_meta:
            warnings.warn(f"Missing identifier {track_id} in the txt metadata.")

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
    path: Union[str, os.PathLike],
    detections_sequence: Sequence[byotrack.Detections],
    as_res=True,
    as_seg=False,
    n_digit=4,
) -> None:
    """Save detections in the CTC format [10]

    It will save one tiff image for each frame containing the segmentation of objects.

    See the official documentation of CTC at
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf

    Args:
        path (Union[str, os.PathLike]): Folder path where to store the .tif files
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

    os.makedirs(path, exist_ok=True)

    for frame_id, detections in enumerate(tqdm.tqdm(detections_sequence, desc="Saving Detections to CTC")):
        segmentation = detections.segmentation.numpy().astype(np.uint16)

        if as_res:
            name = f"mask{frame_id:0{n_digit}}.tif"
        elif as_seg:
            name = f"man_seg{frame_id:0{n_digit}}.tif"
        else:
            name = f"man_track{frame_id:0{n_digit}}.tif"

        if detections.dim == 2:  # Probably not required here, but let's be precise just in case
            segmentation = segmentation[None, None, None, ..., None]  # ImageJ tiff format: TZCYXS
        else:
            segmentation = segmentation[None, :, None, ..., None]  # ImageJ tiff format: TZCYXS

        tifffile.imwrite(path / name, segmentation, imagej=True, compression="zlib")


def _save_metadata(path: pathlib.Path, tracks: Collection[byotrack.Track]):
    """Saves tracks metadata in the given file"""
    lines = []
    for track in tracks:
        if track.merge_id != -1:
            warnings.warn("CTC format do not support merge events. Merge events are removed.")

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


@numba.njit(parallel=True, cache=byotrack.NUMBA_CACHE)
def _fast_relabel(segmentation: np.ndarray, mapping: np.ndarray):
    """Inplace fast relabel with given mapping

    It assumes that seg.max() is small before the number of pixels of the image (which is always the case in practice).
    """
    segmentation = segmentation.reshape(-1)

    for i in numba.prange(segmentation.size):  # pylint: disable=not-an-iterable
        if segmentation[i]:
            segmentation[i] = mapping[segmentation[i] - 1]


@numba.njit(parallel=True, cache=byotrack.NUMBA_CACHE)
def _fast_disk_2d(
    segmentation: np.ndarray,
    bbox: np.ndarray,
    positions: np.ndarray,
    identifiers: np.ndarray,
    radius: np.ndarray,
    *,
    overwrite=False,
):
    """Fast inplace drawing of disk in 2D

    Args:
        segmentation (np.ndarray): Segmentation image to draw on
            Shape: (H, W), dtype: uint16
        bbox (np.ndarray): Indices to consider around the particles (see `draw_disk`)
            Shape: (m, d), dtype: int
        positions (np.ndarray): Positions of the disks' centers
            Shape: (n, d), dtype: float32
        identifiers (np.ndarray): Identifier of each disk
            Shape: (n,), dtype: uint16
        radius (np.ndarray): Radius of each disk
            Shape: (n, ), dtype: float32
        overwrite (bool): Overwrite pixels that are already written (!=0)
            Default: False

    """
    positions_round = np.round(positions).astype(np.int64)
    radius = radius**2
    best_dist = np.full(segmentation.shape, np.inf, dtype=np.float32)

    for k in range(positions_round.shape[0]):
        for l in numba.prange(bbox.shape[0]):  # pylint: disable=not-an-iterable
            pos = bbox[l] + positions_round[k]

            i, j = pos

            if not 0 <= i < segmentation.shape[0] or not 0 <= j < segmentation.shape[1]:
                continue

            if not overwrite and segmentation[i, j] != 0 and best_dist[i, j] == np.inf:
                continue

            delta = pos - positions[k]
            dist = delta @ delta

            if dist <= radius[k]:
                if dist < best_dist[i, j]:
                    segmentation[i, j] = identifiers[k]
                    best_dist[i, j] = dist


@numba.njit(parallel=True, cache=byotrack.NUMBA_CACHE)
def _fast_disk_3d(  # pylint: disable=too-many-locals
    segmentation: np.ndarray,
    bbox: np.ndarray,
    positions: np.ndarray,
    identifiers: np.ndarray,
    radius: np.ndarray,
    *,
    anisoptropy=1.0,
    overwrite=False,
):
    """Fast inplace drawing of disk in 3D

    Args:
        segmentation (np.ndarray): Segmentation image to draw on
            Shape: (D, H, W), dtype: uint16
        bbox (np.ndarray): Indices to consider around the particles (see `draw_disk`)
            Shape: (m, d), dtype: int
        positions (np.ndarray): Positions of the disks' centers
            Shape: (n, d), dtype: float32
        identifiers (np.ndarray): Identifier of each disk
            Shape: (n,), dtype: uint16
        radius (np.ndarray): Radius of each disk
            Shape: (n, ), dtype: float32
        anisotropy (float): Relative size of a pixel along the depth dimension
            versus height/width dimensions.
            Default: 1.0
        overwrite (bool): Overwrite pixels that are already written (!=0)
            Default: False

    """
    positions_round = np.round(positions).astype(np.int64)
    radius = radius**2
    best_dist = np.full(segmentation.shape, np.inf, dtype=np.float32)

    for k in range(positions_round.shape[0]):
        for l in numba.prange(bbox.shape[0]):  # pylint: disable=not-an-iterable
            pos = bbox[l] + positions_round[k]

            z, i, j = pos

            if (
                not 0 <= z < segmentation.shape[0]
                or not 0 <= i < segmentation.shape[1]
                or not 0 <= j < segmentation.shape[2]
            ):
                continue

            if not overwrite and segmentation[z, i, j] != 0 and best_dist[z, i, j] == np.inf:
                continue

            delta = pos - positions[k]
            delta[0] *= anisoptropy  # Increase distance in Z by the anisotropy
            dist = delta @ delta

            if dist <= radius[k]:
                if dist < best_dist[z, i, j]:
                    segmentation[z, i, j] = identifiers[k]
                    best_dist[z, i, j] = dist


def draw_disk(
    segmentation: np.ndarray,
    positions: np.ndarray,
    identifiers: np.ndarray,
    radius: np.ndarray,
    *,
    anisotropy=1.0,
    overwrite=False,
):
    """Draw disks on the segmentation

    Args
        segmentation (np.ndarray): Segmentation image to draw on
            Shape: (D, H, W), dtype: uint16
        positions (np.ndarray): Positions of the disks' centers
            Shape: (n, d), dtype: float32
        identifiers (np.ndarray): Identifier of each disk
            Shape: (n,), dtype: uint16
        radius (np.ndarray): Radius of each disk
            Shape: (n, ), dtype: float32)
        anisotropy (float): Relative size of a pixel along the depth dimension
            versus height/width dimensions.
            Default: 1.0
        overwrite (bool): Overwrite pixels that are already written (!=0)
            Default: False

    """
    # Wrapped to redirect in 2D/3D and numba does not support np.indices in 3.8
    if segmentation.ndim == 3:
        thresh = round(radius.max())
        bbox: np.ndarray = np.indices((thresh * 2 + 1, thresh * 2 + 1, thresh * 2 + 1)).transpose(1, 2, 3, 0) - thresh
        bbox = bbox.reshape(-1, 3)
        _fast_disk_3d(segmentation, bbox, positions, identifiers, radius, anisoptropy=anisotropy, overwrite=overwrite)
    else:
        thresh = int(round(radius.max()))
        bbox = np.indices((thresh * 2 + 1, thresh * 2 + 1)).transpose(1, 2, 0) - thresh
        bbox = bbox.reshape(-1, 2)
        _fast_disk_2d(segmentation, bbox, positions, identifiers, radius, overwrite=overwrite)


def save_tracks(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements,too-many-arguments
    path: Union[str, os.PathLike],
    tracks: Collection[byotrack.Track],
    detections_sequence: Sequence[byotrack.Detections] = (),
    *,
    as_res=True,
    as_seg=False,
    default_radius=3.0,
    last=0,
    shape: Optional[Tuple[int, ...]] = None,
    n_digit=4,
    anisotropy=1.0,
    overwrite_detections=False,
):
    """Save tracks in the CTC format [10]

    It will save one tiff image for each frame containing the segmentation of objects and a metadata txt file
    describing the tracks identifiers, start/end frames and parents.

    Parent information is not supported yet.

    When no detections_sequence is given, tracks segmentations are simply drawn as disk with `default_radius`
    at the track localization.
    When detections_sequence is given, then for tracks without detections associated, a disk is drawn
    with `default_radius` (set at 0 to drop this behavior), otherwise the detection segmentation is used.

    For smarter behaviors, one can directly modify the segmentation before saving.

    See the official documentation of CTC at
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf

    Args:
        path (Union[str, os.PathLike]): Folder path where to store the .tif files
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
        shape (Optional[Tuple[int, ...]]): Optional shape. Required when no detections_sequence is provided
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

    os.makedirs(path, exist_ok=True)

    if as_res:
        _save_metadata(path / "res_track.txt", tracks)
    else:
        _save_metadata(path / "man_track.txt", tracks)

    if not last:
        last = max(track.start + len(track) - 1 for track in tracks)

    if detections_sequence:
        if shape:
            assert detections_sequence[0].shape == shape, "Given shape is not compatible with detections one"
        shape = detections_sequence[0].shape
    else:
        assert shape is not None, "Without detections_sequence, you need to provide the shape argument"

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
                        "Found a missing position inside a track segment. This is not supported by CTC softwares."
                        "Consider filling holes with Interpolators."
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
            _fast_relabel(segmentation, det_to_track_ids + 1)  # Offset of 1 for track id as 0 is not valid for CTC
        else:
            segmentation = np.zeros(shape, dtype=np.uint16)

        # Add circle to the segmentation
        if disk_ids:
            unique = set(_fast_unique(segmentation).tolist())
            draw_disk(
                segmentation,
                torch.stack(disk_positions).numpy(),
                np.array(disk_ids, dtype=np.uint16) + 1,
                np.full(len(disk_ids), default_radius, dtype=np.float32),
                anisotropy=anisotropy,
                overwrite=overwrite_detections,
            )

            # Safety checks because CTC is quite restrictive
            new_unique = set(_fast_unique(segmentation).tolist())
            diff = {identifier - 1 for identifier in unique.difference(new_unique)}
            if diff:
                warnings.warn(
                    f"Some tracks were fully occluded: {diff}. This will induce"
                    " a missing position in the track segment which is not suppoted by CTC softwares."
                    " Consider decreasing the radius of track disks, or adding manually their segmentations."
                )

            diff = {identifier for identifier in disk_ids if identifier + 1 not in new_unique}
            if diff:
                warnings.warn(
                    f"The disk of some added tracks are not found (outside of image or occluded): {diff}."
                    " This will induce a missing position in the track segment which is not suppoted by CTC softwares."
                    " Fixing this can be done by changing the radius, or manually dropping/correcting the track."
                )

        # Save a tiff
        if len(shape) == 2:
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
