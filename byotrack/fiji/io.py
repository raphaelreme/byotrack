"""Utilities for inputs/outputs with icy"""


import os
from typing import Collection, Union
from xml.etree import ElementTree as ET

import numpy as np
import tifffile  # type: ignore
import torch

import byotrack


def save_detections(detections_sequence: Collection[byotrack.Detections], path: Union[str, os.PathLike]) -> None:
    """Save a sequence of detections as a stack of segmentation that can be reload by the trackmate label detector

    .. warning:: Only supports consecutive detections

    Args:
        detections_sequence (Collection[Detections]): Detections for each frame (Should be consecutives)
        path (str | os.PathLike): Output path

    """
    segmentations = np.concatenate([detections.segmentation[None].numpy() for detections in detections_sequence])
    tifffile.imwrite(path, segmentations, photometric="minisblack")


def load_tracks(path: Union[str, os.PathLike]) -> Collection[byotrack.Track]:
    """Load tracks saved by trackmate

    Currently only 2d tracks are supported

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
        Collection[Track]: Parsed tracks

    """
    tracks = []
    model = ET.parse(path).getroot().find("Model")
    assert model is not None, "Data not found in file"

    # Load all spots to rebuild tracks
    spots = {}
    for frame_spots in model.findall("AllSpots")[0]:
        for spot in frame_spots:
            spots[spot.get("ID")] = (
                int(frame_spots.attrib["frame"]),
                float(spot.attrib["POSITION_X"]),
                float(spot.attrib["POSITION_Y"]),
            )

    # Go through tracks and rebuild from edges
    for track in model.findall("AllTracks")[0].findall("Track"):
        track_spots = set(map(lambda edge: spots[edge.get("SPOT_SOURCE_ID")], track.findall("Edge")))
        track_spots.update(map(lambda edge: spots[edge.get("SPOT_SOURCE_ID")], track.findall("Edge")))
        spot_data = sorted(track_spots)
        start = spot_data[0][0]

        old_frame = start - 1
        points = []
        for frame, x, y in spot_data:
            # Extend with nan if there is a temporal gap
            points.extend([[torch.nan, torch.nan]] * (frame - old_frame - 1))
            points.append([y, x])
            old_frame = frame

        tracks.append(byotrack.Track(start, torch.tensor(points), int(track.attrib["TRACK_ID"])))

    return tracks


# Tracks could be saved in a simpler format by TrackMate with ExportTracksToXML.
# But this format is the one used in GUI, so let's keept it
