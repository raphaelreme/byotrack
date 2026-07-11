from __future__ import annotations

import zlib
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import pytest
import torch

import byotrack
from byotrack.icy.io import load_tracks, save_detections, save_tracks

if TYPE_CHECKING:
    import pathlib

    from byotrack.api.detections import SegmentationDetections


## save_tracks


def test_save_tracks_2d_xml_structure(tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track]):
    path = tmp_path / "tracks.xml"
    save_tracks(tracks_2d, path)

    root = ET.parse(path).getroot()  # noqa: S314
    track_group = root.find("trackgroup")
    assert track_group is not None
    track_elements = track_group.findall("track")
    assert len(track_elements) == 2
    for track_el in track_elements:
        detections = track_el.findall("detection")
        assert len(detections) == 2


def test_save_tracks_2d_positions(tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track]):
    path = tmp_path / "tracks.xml"
    save_tracks(tracks_2d, path)

    root = ET.parse(path).getroot()  # noqa: S314
    track_group = root.find("trackgroup")
    assert track_group is not None
    # Find track with id "0"
    track_el = next(t for t in track_group.findall("track") if t.attrib["id"] == "0")
    det0 = track_el.findall("detection")[0]

    assert det0.attrib["t"] == "0"
    assert float(det0.attrib["y"]) == pytest.approx(20.0)
    assert float(det0.attrib["x"]) == pytest.approx(10.0)
    assert det0.attrib["z"] == "-1"  # 2D => z=-1


def test_save_tracks_3d_positions(tmp_path: pathlib.Path, tracks_3d: list[byotrack.Track]):
    path = tmp_path / "tracks.xml"
    save_tracks(tracks_3d, path)

    root = ET.parse(path).getroot()  # noqa: S314
    track_group = root.find("trackgroup")
    assert track_group is not None
    det0 = track_group.findall("track")[0].findall("detection")[0]

    assert float(det0.attrib["z"]) == pytest.approx(2.0)
    assert float(det0.attrib["y"]) == pytest.approx(20.0)
    assert float(det0.attrib["x"]) == pytest.approx(10.0)


def test_save_tracks_nan_warns(tmp_path: pathlib.Path):
    track = byotrack.Track(0, torch.tensor([[float("nan"), float("nan")], [5.0, 3.0]]), 0)
    path = tmp_path / "tracks.xml"

    with pytest.warns(UserWarning, match="NaN"):
        save_tracks([track], path)


def test_save_tracks_name_in_trackgroup(tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track]):
    path = tmp_path / "tracks.xml"
    save_tracks(tracks_2d, path, name="MyExperiment")

    root = ET.parse(path).getroot()  # noqa: S314
    track_group = root.find("trackgroup")
    assert track_group is not None
    assert track_group.attrib["description"] == "MyExperiment"


## load_tracks


def _write_icy_xml(path, track_specs):
    """Write a minimal Icy track XML.

    Args:
        path: Output path.
        track_specs: list of (id, list of (t, x, y, z)).
    """
    root = ET.Element("root")
    track_group = ET.SubElement(root, "trackgroup", {"description": "test"})
    for tid, detections in track_specs:
        track_el = ET.SubElement(track_group, "track", {"id": str(tid)} if tid is not None else {})
        for t, x, y, z in detections:
            ET.SubElement(
                track_el,
                "detection",
                {"t": str(t), "x": str(x), "y": str(y), "z": str(z)},
            )
    ET.ElementTree(root).write(path)


def test_load_tracks_2d(tmp_path: pathlib.Path):
    path = tmp_path / "tracks.xml"
    _write_icy_xml(path, [(1, [(0, 10.0, 20.0, -1.0), (1, 11.0, 21.0, -1.0)])])

    tracks = load_tracks(path)
    assert len(tracks) == 1
    assert tracks[0].start == 0
    assert len(tracks[0]) == 2
    assert tracks[0].dim == 2
    assert torch.allclose(tracks[0].points[0], torch.tensor([20.0, 10.0]))


def test_load_tracks_3d(tmp_path: pathlib.Path):
    path = tmp_path / "tracks.xml"
    _write_icy_xml(path, [(1, [(0, 10.0, 20.0, 2.0)])])

    tracks = load_tracks(path)
    assert len(tracks) == 1
    assert tracks[0].dim == 3
    assert torch.allclose(tracks[0].points[0], torch.tensor([2.0, 20.0, 10.0]))


def test_load_tracks_with_gaps(tmp_path: pathlib.Path):
    path = tmp_path / "tracks.xml"
    _write_icy_xml(path, [(1, [(0, 10.0, 20.0, -1.0), (2, 11.0, 21.0, -1.0)])])  # frame 1 missing

    tracks = load_tracks(path)
    assert len(tracks[0]) == 3
    assert torch.isnan(tracks[0].points[1]).all()


def test_load_tracks_negative_id_abs(tmp_path: pathlib.Path):
    path = tmp_path / "tracks.xml"
    _write_icy_xml(path, [(-42, [(0, 10.0, 20.0, -1.0)])])

    tracks = load_tracks(path)
    assert tracks[0].identifier == 42


def test_load_tracks_without_id_assigns(tmp_path: pathlib.Path):
    path = tmp_path / "tracks.xml"
    _write_icy_xml(path, [(None, [(0, 10.0, 20.0, -1.0)])])

    tracks = load_tracks(path)
    assert tracks[0].identifier >= 0


def test_load_tracks_no_trackgroup_raises(tmp_path: pathlib.Path):
    path = tmp_path / "bad.xml"
    path.write_text("<root></root>")

    with pytest.raises(ValueError, match="No track"):
        load_tracks(path)


## save/load round-trips


def test_save_load_round_trip_2d(tmp_path: pathlib.Path, tracks_2d: list[byotrack.Track]):
    path = tmp_path / "tracks.xml"
    save_tracks(tracks_2d, path)
    loaded = load_tracks(path)

    for original, result in zip(tracks_2d, loaded, strict=True):  # Order is preserved
        assert result.identifier == original.identifier
        assert result.start == original.start
        assert torch.allclose(result.points, original.points)


def test_save_load_round_trip_3d(tmp_path: pathlib.Path, tracks_3d: list[byotrack.Track]):
    path = tmp_path / "tracks.xml"
    save_tracks(tracks_3d, path)
    loaded = load_tracks(path)

    assert len(loaded) == 1
    assert torch.allclose(loaded[0].points, tracks_3d[0].points)


## save_detections


def _xml_text(element: ET.Element, tag: str) -> str:
    child = element.find(tag)
    assert child is not None, f"<{tag}> not found"
    assert child.text is not None, f"<{tag}> has no text"
    return child.text


def test_save_detections_2d_xml_structure(tmp_path: pathlib.Path, seg_2d_det: SegmentationDetections):
    path = tmp_path / "rois.xml"
    save_detections([seg_2d_det], path)

    root = ET.parse(path).getroot()  # noqa: S314
    rois = root.findall("roi")
    assert len(rois) == 2

    for roi in rois:
        assert _xml_text(roi, "classname") == "plugins.kernel.roi.roi2d.ROI2DArea"
        assert _xml_text(roi, "t") == "0"
        assert _xml_text(roi, "z") == "0"
        assert roi.find("boolMaskData") is not None


def test_save_detections_2d_bounds(tmp_path: pathlib.Path, seg_2d_det: SegmentationDetections):
    path = tmp_path / "rois.xml"
    save_detections([seg_2d_det], path)

    root = ET.parse(path).getroot()  # noqa: S314
    rois = root.findall("roi")

    bounds = {
        (int(_xml_text(roi, "boundsX")), int(_xml_text(roi, "boundsY"))): (
            int(_xml_text(roi, "boundsW")),
            int(_xml_text(roi, "boundsH")),
        )
        for roi in rois
    }
    # seg[2:5, 3:7]=1 => boundsX=3(j), boundsY=2(i), boundsW=4, boundsH=3
    assert bounds[(3, 2)] == (4, 3)
    # seg[6:9, 8:11]=2 => boundsX=8, boundsY=6, boundsW=3, boundsH=3
    assert bounds[(8, 6)] == (3, 3)


def test_save_detections_2d_mask_data(tmp_path: pathlib.Path, seg_2d_det: SegmentationDetections):
    path = tmp_path / "rois.xml"
    save_detections([seg_2d_det], path)

    root = ET.parse(path).getroot()  # noqa: S314
    for roi in root.findall("roi"):
        bounds_w = int(_xml_text(roi, "boundsW"))
        bounds_h = int(_xml_text(roi, "boundsH"))
        mask_data_text = _xml_text(roi, "boolMaskData")

        compressed = bytes(int(s, 16) for s in mask_data_text.split(":"))
        decoded = zlib.decompress(compressed)
        # Decoded bytes are the raw boolean mask values
        assert len(decoded) == bounds_w * bounds_h
        assert all(b == 1 for b in decoded)  # entire bounding box is filled


def test_save_detections_3d_xml_structure(tmp_path: pathlib.Path, seg_3d_det: SegmentationDetections):
    path = tmp_path / "rois.xml"
    save_detections([seg_3d_det], path)

    root = ET.parse(path).getroot()  # noqa: S314
    rois = root.findall("roi")
    assert len(rois) == 1

    roi = rois[0]
    assert _xml_text(roi, "classname") == "plugins.kernel.roi.roi3d.ROI3DArea"
    slices = roi.findall("slice")
    assert len(slices) > 0
    for slice_ in slices:
        assert _xml_text(slice_, "classname") == "plugins.kernel.roi.roi2d.ROI2DArea"


def test_save_detections_frame_id_in_t(tmp_path: pathlib.Path, seg_2d_det: SegmentationDetections):
    path = tmp_path / "rois.xml"
    save_detections([seg_2d_det, seg_2d_det], path)

    root = ET.parse(path).getroot()  # noqa: S314
    rois = root.findall("roi")
    ts = [int(_xml_text(roi, "t")) for roi in rois]
    assert 0 in ts
    assert 1 in ts


def test_save_detections_empty(tmp_path: pathlib.Path):
    path = tmp_path / "rois.xml"

    with pytest.raises(ValueError, match="No detections to save"):
        save_detections([], path)


def test_save_detections_with_outside_labels_warns(tmp_path: pathlib.Path):
    path = tmp_path / "rois.xml"

    # 3 detections as circle, where the second detections is fully outside.
    point_det = byotrack.PointDetections(
        torch.tensor([[10.0, 10.0], [-10.0, -10.0], [-2.0, 5.0]]), radius=4, shape=(20, 20)
    )

    with pytest.warns(UserWarning, match="missing from the segmentation"):
        save_detections([point_det], path)

    root = ET.parse(path).getroot()  # noqa: S314
    rois = root.findall("roi")
    assert len(rois) == 2  # (10, 10) & (-2.0, 5.0)
