from __future__ import annotations

from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import pytest
import torch

from byotrack.api.detections import SegmentationDetections

if TYPE_CHECKING:
    import os


@pytest.fixture
def seg_2d_det() -> SegmentationDetections:
    seg = torch.zeros(10, 12, dtype=torch.int32)
    seg[2:5, 3:7] = 1
    seg[6:9, 8:11] = 2
    return SegmentationDetections(seg)


@pytest.fixture
def seg_3d_det() -> SegmentationDetections:
    seg = torch.zeros(4, 10, 12, dtype=torch.int32)
    seg[1:3, 2:5, 3:7] = 1
    return SegmentationDetections(seg)


def _write_minimal_trackmate_xml(
    path: str | os.PathLike,
    spots: dict[int, tuple[int, float, float, float]],
    tracks: list[tuple[int, list[tuple[int, int]]]],
) -> None:
    """Write a minimal TrackMate XML file.

    Args:
        path: Output path.
        spots: dict mapping spot_id -> (frame, x, y, z).
        tracks: list of (track_id, list of (source_id, target_id) edge pairs).
    """
    root = ET.Element("TrackMate")
    model = ET.SubElement(root, "Model")

    # AllSpots: group spots by frame
    all_spots = ET.SubElement(model, "AllSpots")
    by_frame: dict[int, list] = {}
    for spot_id, (frame, x, y, z) in spots.items():
        by_frame.setdefault(frame, []).append((spot_id, x, y, z))
    for frame in sorted(by_frame):
        frame_el = ET.SubElement(all_spots, "SpotsInFrame", {"frame": str(frame)})
        for spot_id, x, y, z in by_frame[frame]:
            ET.SubElement(
                frame_el,
                "Spot",
                {"ID": str(spot_id), "POSITION_X": str(x), "POSITION_Y": str(y), "POSITION_Z": str(z)},
            )

    # AllTracks
    all_tracks = ET.SubElement(model, "AllTracks")
    for track_id, edges in tracks:
        track_el = ET.SubElement(all_tracks, "Track", {"TRACK_ID": str(track_id)})
        for src, tgt in edges:
            ET.SubElement(track_el, "Edge", {"SPOT_SOURCE_ID": str(src), "SPOT_TARGET_ID": str(tgt)})

    ET.ElementTree(root).write(path)


@pytest.fixture
def trackmate_2d_path(tmp_path):
    """TrackMate XML with two 2D tracks spanning frames 0-1."""
    spots = {
        1: (0, 10.0, 20.0, 0.0),
        2: (1, 11.0, 21.0, 0.0),
        3: (0, 30.0, 40.0, 0.0),
        4: (1, 31.0, 41.0, 0.0),
    }
    tracks = [(0, [(1, 2)]), (1, [(3, 4)])]
    path = tmp_path / "trackmate_2d.xml"
    _write_minimal_trackmate_xml(path, spots, tracks)
    return path


@pytest.fixture
def trackmate_3d_path(tmp_path):
    """TrackMate XML with one 3D track spanning frames 0-1."""
    spots = {
        1: (0, 10.0, 20.0, 2.0),
        2: (1, 11.0, 21.0, 3.0),
    }
    tracks = [(0, [(1, 2)])]
    path = tmp_path / "trackmate_3d.xml"
    _write_minimal_trackmate_xml(path, spots, tracks)
    return path
