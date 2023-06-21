import enum
import os
import pathlib
import shutil
import subprocess
from typing import Collection, Iterable, Optional, Union
from xml.etree import ElementTree as ET
import zlib

import numpy as np
import torch

import byotrack
from byotrack.api.parameters import ParameterEnum


PROTOCOL = pathlib.Path(__file__).parent / "emht_protocol.xml"


class IcyEMHTLinker(byotrack.Linker):
    """Run EMHT [4] from Icy [1]

    This code is only a wrapper arounds Icy implementation as EMHT is painful to implement.

    About EMHT:
    It is a probabilistic tracking that uses statistical motion model on particles. It uses multiple
    kalman filters for each particle allowing a particle to have several mode of motions. It also keeps
    multiple hypothesis of tracking at each frame so that a final detections linking decision is made
    after seeing several frames in the past and future of these detections.

    Here we rely on the handmade protocol "emht_protocol" that expects detections as a valid rois file
    for icy and some hyperparameters.

    The workflow is:

    1. Detections to rois in Icy format
    2. Run the Icy protocol
    3. [In ICY] Read rois, estimate emht parameters, run emht, export tracks to xml
    4. Read Icy tracks and return

    Note:
        This implementation requires Icy to be installed (https://icy.bioimageanalysis.org/download/)

    Attributes:
        icy_path (str | os.PathLike): Path to the icy jar (Icy is called with java -jar <icy_jar>)
        motion (Motion): Prior on the underlying motion model (Brownian vs Directed vs Switching)
            Given to the Icy block that estimates EMHT parameters.
            Default: Motion.BROWNIAN

    ## XXX: Should we add update_cov (bool): Whether to update covariances in Kalman filters

    """

    class Motion(enum.Enum):
        """Different kind of motion model:

        Brownian: Random gaussian displacement at each time
        Directed: Random gaussian noise around a directed trajectory
        Multi: Switch randomly between Brownian and Directed

        """

        BROWNIAN = (0, 0)
        DIRECTED = (1, 0)
        MULTI = (0, 1)

    parameters = {"motion": ParameterEnum({Motion.BROWNIAN, Motion.DIRECTED, Motion.MULTI})}
    rois_file = "_tmp_rois.xml"
    tracks_file = "_tmp_tracks.xml"
    cmd = (
        "java -jar icy.jar -hl -x plugins.adufour.protocols.Protocols "
        "protocol={protocol} rois={rois} tracks={tracks} directed={directed} multi={multi}"
    )

    def __init__(self, icy_path: Optional[Union[str, os.PathLike]] = None) -> None:
        super().__init__()
        if icy_path is None:
            icy_path = shutil.which("icy")
            if icy_path is None:
                raise RuntimeError("Icy not found, please use `icy_path` to precise where it should be found")

        assert os.path.isfile(os.path.join(os.path.dirname(icy_path), "icy.jar")), f"Icy jar not found at {icy_path}"

        self.icy_path = icy_path
        self.motion = IcyEMHTLinker.Motion.BROWNIAN

    def run(
        self, video: Iterable[np.ndarray], detections_sequence: Collection[byotrack.Detections]
    ) -> Collection[byotrack.Track]:
        try:
            self.save_detections_as_icy_rois(detections_sequence, self.rois_file)
            self._run_icy()
            # Sort tracks by starting time and then position (They seem to be randomly sorted otherwise and in addition
            # with EMC2 torch-tps approximate propagation it yields undeterministic behaviors)
            return sorted(
                self.parse_tracks(self.tracks_file), key=lambda track: (track.start, track.points[0].sum().item())
            )
        finally:
            if os.path.exists(self.rois_file):
                os.remove(self.rois_file)
            if os.path.exists(self.tracks_file):
                os.remove(self.tracks_file)

    def _run_icy(self):
        """Run the icy process"""
        directed, multi = self.motion.value
        cmd = self.cmd.format(
            protocol=PROTOCOL,
            rois=os.path.abspath(self.rois_file),
            tracks=os.path.abspath(self.tracks_file),
            directed=directed,
            multi=multi,
        )
        print("Launching ICY with:", cmd)

        subprocess.run(cmd.split(), check=True, cwd=os.path.dirname(self.icy_path))

        # Has to check because icy do not return any non-zero return code
        if not os.path.exists(self.tracks_file):
            raise RuntimeError(
                """No track found from Icy.

            This probably results from a failure in Icy software. Please look at the Java Exception displayed by Icy.

            Note that Icy may fail the first time you launch this code because Icy updates the required protocols
            and is unable to continue to run after the update.
            """
            )

    @staticmethod
    def save_detections_as_icy_rois(
        detections_sequence: Collection[byotrack.Detections], path: Union[str, os.PathLike]
    ) -> None:
        """Save a sequence of detections as valid rois for icy

        Format:

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

        Only needed tags are filled in the current implementation

        Args:
            detections_sequence (Collection[Detections]): Detections for each frame
            path (str | os.PathLike): Output path

        """
        ## TODO: Add support for 3D images
        root = ET.Element("root")
        for detections in detections_sequence:
            for label, bbox in enumerate(detections.bbox):
                roi = ET.SubElement(root, "roi")
                i, j, height, width = bbox.tolist()
                mask = detections.segmentation[i : i + height, j : j + width] == label + 1

                ET.SubElement(roi, "classname").text = "plugins.kernel.roi.roi2d.ROI2DArea"
                ET.SubElement(roi, "t").text = str(detections.frame_id)
                ET.SubElement(roi, "boundsX").text = str(j)
                ET.SubElement(roi, "boundsY").text = str(i)
                ET.SubElement(roi, "boundsW").text = str(width)
                ET.SubElement(roi, "boundsH").text = str(height)

                # The mask is converted into bytes and zipped
                compressed_bytes = zlib.compress(bytes(mask.reshape(height * width)), 2)

                # and then converted to the good string format: byte:byte:...:byte
                ET.SubElement(roi, "boolMaskData").text = ":".join(map(lambda byte: hex(byte)[2:], compressed_bytes))

        ET.ElementTree(root).write(path)

    @staticmethod
    def parse_tracks(path: Union[str, os.PathLike]) -> Collection[byotrack.Track]:
        """Parse tracks output by ICY

        Args:
            path (str | os.PathLike): Input path

        Returns:
            Collection[Track]: Parsed tracks

        """
        tree = ET.parse(path)

        track_group = tree.find("trackgroup")
        assert track_group, "Track group not found in file"

        tracks = []

        for track in track_group:
            identifier = None
            if track.attrib.get("id"):
                identifier = int(track.attrib["id"])

            points = []
            frames = []
            for point in track:
                frames.append(int(point.attrib["t"]))
                points.append((float(point.attrib["x"]), float(point.attrib["y"]), float(point.attrib["z"])))

            start = frames[0]
            # Temporal check
            assert frames == list(range(start, start + len(frames))), "Found a track with non consecutive points"

            points_tensor = torch.tensor(points)
            if (points_tensor[:, 2] <= 0).all():
                points_tensor = points_tensor[:, :2].clone()

            tracks.append(byotrack.Track(start, points_tensor, identifier))

        return tracks
