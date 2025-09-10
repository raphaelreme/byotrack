"""Utilities for inputs/outputs with icy"""

import os
from typing import Collection, List, Sequence, Union
import warnings
from xml.etree import ElementTree as ET
import zlib

import torch

import byotrack


def save_detections(  # pylint: disable=too-many-locals
    detections_sequence: Sequence[byotrack.Detections], path: Union[str, os.PathLike]
) -> None:
    """Save a sequence of detections as valid rois for icy

    Format example (2D):

    .. code-block:: xml

        <root>
            <roi>
                <classname>plugins.kernel.roi.roi2d.ROI2DArea</classname>
                <id>30</id>
                <name>spot #0</name>
                <selected>false</selected>
                <readOnly>false</readOnly>
                <properties>None</properties>
                <color>-16711936</color>
                <stroke>2</stroke>
                <opacity>0.3</opacity>
                <showName>false</showName>
                <z>0</z>
                <t>0</t>
                <c>-1</c>
                <boundsX>238</boundsX>
                <boundsY>486</boundsY>
                <boundsW>1</boundsW>
                <boundsH>2</boundsH>
                <boolMaskData>78:5e:63:64:4:0:0:5:0:3</boolMaskData>
            </roi>
            ...
        </root>

    Format example (3D):

    .. code-block:: xml

        <root>
            <roi>
                <classname>plugins.kernel.roi.roi3d.ROI3DArea</classname>
                <id>30</id>
                <name>spot #0</name>
                <selected>false</selected>
                <readOnly>false</readOnly>
                <properties>None</properties>
                <color>-16711936</color>
                <stroke>2</stroke>
                <opacity>0.3</opacity>
                <showName>false</showName>
                <t>0</t>
                <c>-1</c>
                <slice>
                    <classname>plugins.kernel.roi.roi2d.ROI2DArea</classname>
                    ...  # See 2D format
                </slice>
                ...
            </roi>
            ...
        </root>


    Only needed tags are filled in the current implementation

    Args:
        detections_sequence (Sequence[Detections]): Detections for each frame.
            The `frame_id` attribute is not used and we rely on the position of
            the Detections in the sequence.
        path (str | os.PathLike): Output path

    """
    root = ET.Element("root")
    for frame_id, detections in enumerate(detections_sequence):
        shape = torch.tensor(detections.shape)
        start = detections.bbox[:, : detections.dim]
        end = start + detections.bbox[:, detections.dim :]

        # Filter outside of frame detections
        invalid = torch.logical_or(end.min(dim=-1).values <= 0, (start >= shape).any(dim=-1))

        # Clip on frame shape
        start = torch.maximum(torch.tensor(0), start)
        end = torch.minimum(shape, end)

        for label in range(detections.length):
            roi = ET.SubElement(root, "roi")

            if invalid[label]:
                continue

            if detections.dim == 2:
                i, j = start[label].tolist()
                end_i, end_j = end[label].tolist()
                mask = detections.segmentation[i:end_i, j:end_j] == label + 1

                ET.SubElement(roi, "classname").text = "plugins.kernel.roi.roi2d.ROI2DArea"
                ET.SubElement(roi, "t").text = str(frame_id)
                ET.SubElement(roi, "z").text = "0"
                ET.SubElement(roi, "boundsX").text = str(j)
                ET.SubElement(roi, "boundsY").text = str(i)
                ET.SubElement(roi, "boundsW").text = str(end_j - j)
                ET.SubElement(roi, "boundsH").text = str(end_i - i)

                # The mask is converted into bytes and zipped
                compressed_bytes = zlib.compress(bytes(mask.reshape(-1).numpy()), 2)

                # and then converted to the good string format: byte:byte:...:byte
                ET.SubElement(roi, "boolMaskData").text = ":".join(map(lambda byte: hex(byte)[2:], compressed_bytes))
            else:
                k, i, j = start[label].tolist()
                end_k, end_i, end_j = end[label].tolist()
                ET.SubElement(roi, "classname").text = "plugins.kernel.roi.roi3d.ROI3DArea"
                ET.SubElement(roi, "t").text = str(frame_id)

                for z in range(k, end_k):
                    slice_ = ET.SubElement(roi, "slice")
                    mask = detections.segmentation[z, i:end_i, j:end_j] == label + 1

                    ET.SubElement(slice_, "classname").text = "plugins.kernel.roi.roi2d.ROI2DArea"
                    ET.SubElement(slice_, "t").text = str(frame_id)
                    ET.SubElement(slice_, "z").text = str(z)
                    ET.SubElement(roi, "boundsX").text = str(j)
                    ET.SubElement(roi, "boundsY").text = str(i)
                    ET.SubElement(roi, "boundsW").text = str(end_j - j)
                    ET.SubElement(roi, "boundsH").text = str(end_i - i)

                    # The mask is converted into bytes and zipped
                    compressed_bytes = zlib.compress(bytes(mask.reshape(-1).numpy()), 2)

                    # and then converted to the good string format: byte:byte:...:byte
                    ET.SubElement(slice_, "boolMaskData").text = ":".join(
                        map(lambda byte: hex(byte)[2:], compressed_bytes)
                    )

    ET.ElementTree(root).write(path)


def load_tracks(path: Union[str, os.PathLike]) -> List[byotrack.Track]:
    """Load tracks in Icy format

    Format example:

    .. code-block:: xml

        <root>
            <trackfile version="1">
                <trackgroup description="mhtTracks-Run1">
                    <track id="-1743400864">
                        <detection classname="plugins.nchenouard.particletracking.DetectionSpotTrack" color="-6553856"
                            t="0" type="1" x="338.14285714285717" y="207.71428571428572" z="0"/>
                        <detection classname="plugins.nchenouard.particletracking.DetectionSpotTrack" color="-6553856"
                            t="1" type="1" x="338.5" y="207.5" z="0"/>
                        ...
                    </track>
                    ...
                </trackgroup>
            </trackfile>
        </root>

    For each point in each track the frame (time) and position (x, y, z) are given. An additional type precise if
    the detection is real or extrapolated. (Unused in ByoTrack)

    Args:
        path (str | os.PathLike): Input path

    Returns:
        List[Track]: Parsed tracks

    """
    tree = ET.parse(path)

    track_group = tree.find("trackgroup")
    assert track_group is not None, "Track group not found in file"

    tracks = []

    for track in track_group:
        identifier = None
        if track.attrib.get("id"):  # XXX: Find why ids have this format
            identifier = abs(int(track.attrib["id"]))

        points = []
        frames = []
        unused_z = True
        for point in track:
            frames.append(int(point.attrib["t"]))
            points.append((float(point.attrib["z"]), float(point.attrib["y"]), float(point.attrib["x"])))
            if float(point.attrib["z"]) > 0:
                unused_z = False

        # Usually frames are sorted and consecutives. But we rather expect less to be more robust
        start = min(frames)
        end = max(frames) + 1

        points_tensor = torch.full((end - start, 3 - unused_z), torch.nan, dtype=torch.float32)
        points_tensor[torch.tensor(frames) - start] = torch.tensor(points)[:, unused_z:]

        tracks.append(byotrack.Track(start, points_tensor, identifier))

    return tracks


def save_tracks(tracks: Collection[byotrack.Track], path: Union[str, os.PathLike], name: str = "ByoTrack") -> None:
    """Save tracks in Icy format

    .. warning:: Icy do not support partial tracks (track with undefined positions). Before calling this function
        you should first interpolate any missing position (See `ForwardBackwardInterpolater`)

    Format example:

    .. code-block:: xml

        <root>
            <trackfile version="1"/>
            <trackgroup description="mhtTracks-Run1">
                <track id="-1743400864">
                    <detection classname="plugins.nchenouard.particletracking.DetectionSpotTrack" color="-6553856"
                        t="0" type="1" x="338.14285714285717" y="207.71428571428572" z="0"/>
                    <detection classname="plugins.nchenouard.particletracking.DetectionSpotTrack" color="-6553856"
                        t="1" type="1" x="338.5" y="207.5" z="0"/>
                    ...
                </track>
                ...
            </trackgroup>
        </root>

    For each point in each track the frame (t) and position (x, y, z) are given. An additional type precise if
    the detection is real or extrapolated. (Unused in ByoTrack).

    Only needed tags are filled in the current implementation. (t, x, y, z)

    Args:
        tracks (Collection[Track]): Tracks to save. (Should not contain any NaN positions)
        path (str | os.PathLike): Output path
        name (str): Name of the tracks that will be displayed by TrackManager
            Default: ByoTrack

    """
    root = ET.Element("root")
    ET.SubElement(root, "trackfile", {"version": "1"})
    track_group = ET.SubElement(root, "trackgroup", {"description": name})
    for track in tracks:
        track_element = ET.SubElement(track_group, "track", {"id": str(track.identifier)})

        frame = track.start
        for point in track.points:
            if torch.isnan(point).any():
                warnings.warn(
                    "Found NaN in track points and Icy do not support partial tracks. "
                    "You should consider first using an interpolater on the tracks."
                )
                frame += 1
                continue  # No detection for this frame

            if point.shape[0] == 3:
                x = str(point[2].item())
                y = str(point[1].item())
                z = str(point[0].item())
            else:
                x = str(point[1].item())
                y = str(point[0].item())
                z = "-1"

            ET.SubElement(
                track_element,
                "detection",
                {
                    "classname": "plugins.nchenouard.particletracking.DetectionSpotTrack",
                    "t": str(frame),
                    "x": x,
                    "y": y,
                    "z": z,
                },
            )

            frame += 1

    ET.ElementTree(root).write(path)
