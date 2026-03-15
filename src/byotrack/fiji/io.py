"""Utilities for inputs/outputs with fiji."""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import numpy as np
import tifffile  # type: ignore[import-untyped]
import torch

import byotrack

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence


def save_detections(detections_sequence: Sequence[byotrack.Detections], path: str | os.PathLike) -> None:
    """Save a sequence of detections as a stack of segmentation that can be reload by the trackmate label detector.

    Args:
        detections_sequence (Sequence[Detections]): Detections for each frame. There should be one Detections object
            for every frame of the video (even if empty it should be provided)
        path (str | os.PathLike): Output path

    """
    dim = 2
    if detections_sequence:
        dim = detections_sequence[0].dim

    segmentations = np.concatenate(
        [detections.segmentation[None].numpy().astype(np.uint16) for detections in detections_sequence]
    )
    if dim == 2:  # noqa: PLR2004, SIM108
        segmentations = segmentations[:, None, None, ..., None]  # ImageJ tiff format: TZCYXS
    else:
        segmentations = segmentations[:, :, None, ..., None]  # ImageJ tiff format: TZCYXS

    tifffile.imwrite(path, segmentations, imagej=True, compression="zlib")


def load_tracks(path: str | os.PathLike) -> list[byotrack.Track]:
    """Load tracks saved by trackmate.

    Format example:

    .. code-block:: xml

        <TrackMate>
            <Model>
                <AllSpots>
                    <SpotsInFrame frame="0">
                        <Spot ID="2811" POSITION_X="552.0" POSITION_Y="143.0" POSITION_Z="0.0">...</Spot>
                        ...
                    </SpotsInFrame>
                    ...
                </AllSpots>
                <AllTracks>
                    <Track TRACK_ID="0">
                        <Edge SPOT_SOURCE_ID="24068" SPOT_TARGET_ID="20796"/>
                        ...
                    </Track>
                </AllTracks>
            </Model>
        </TrackMate>

    Args:
        path (str | os.PathLike): Input path

    Returns:
        List[Track]: Parsed tracks

    """
    tracks = []
    model = ET.parse(path).getroot().find("Model")  # noqa: S314 # XXX: Check how to solve this
    if model is None:
        raise ValueError("No tracks found in the given file.")

    # Load all spots to rebuild tracks
    z_s = set()
    spots = {}
    for frame_spots in model.findall("AllSpots")[0]:
        for spot in frame_spots:
            spots[spot.get("ID")] = (
                int(frame_spots.attrib["frame"]),
                float(spot.attrib["POSITION_X"]),
                float(spot.attrib["POSITION_Y"]),
                float(spot.attrib["POSITION_Z"]),
            )
            z_s.add(float(spot.attrib["POSITION_Z"]))

    dim = 2 if len(z_s) == 1 else 3

    # Go through tracks and rebuild from edges
    for track in model.findall("AllTracks")[0].findall("Track"):
        track_spots = {spots[edge.get("SPOT_SOURCE_ID")] for edge in track.findall("Edge")}
        track_spots.update(spots[edge.get("SPOT_TARGET_ID")] for edge in track.findall("Edge"))
        spot_data = sorted(track_spots)
        start = spot_data[0][0]

        old_frame = start - 1
        points = []
        for frame, x, y, z in spot_data:
            # Extend with nan if there is a temporal gap
            if frame <= old_frame:
                raise NotImplementedError("Reading track splitting is not supported yet")
            points.extend([[torch.nan] * dim] * (frame - old_frame - 1))
            points.append([z, y, x] if dim == 3 else [y, x])  # noqa: PLR2004
            old_frame = frame

        tracks.append(byotrack.Track(start, torch.tensor(points), int(track.attrib["TRACK_ID"])))

    return tracks


# Tracks could be saved in a simpler format by TrackMate with ExportTracksToXML.
# But this format is the one used in GUI, so let's focus on this one.
