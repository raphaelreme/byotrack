import dataclasses
import enum
import os
import pathlib
from typing import List, Optional, Sequence, Union
import warnings
from xml.etree import ElementTree as ET

import byotrack
from byotrack import icy
from byotrack.api.tracks import update_detection_ids


PROTOCOL = pathlib.Path(__file__).parent / "emht_protocol.xml"
PARAMETRIZED_PROTOCOL = pathlib.Path(__file__).parent / "emht_protocol_with_full_specs.xml"


class Motion(enum.Enum):
    """Different motion models:

    * BROWNIAN
        Random gaussian displacement at each time
    * DIRECTED
        Random gaussian noise around a directed trajectory
    * MULTI
        Uses both models and switches between them

    """

    BROWNIAN = (0, 0)
    DIRECTED = (1, 0)
    MULTI = (0, 1)


@dataclasses.dataclass
class EMHTParameters:  # pylint: disable=too-many-instance-attributes
    """Parameters of EMHT algorithm [4]

    Some parameters are still not very clear to us, but we describe them to the best of our knowledge.

    Attributes:
        detections_fpr (float): Estimation of the false positive rate of the detection process
            Default: 0.1
        detections_fnr (float): Estimation of the false negative rate of the detection process
            Default: 0.1
        expected_track_length (int): Expected length of tracks. If negative, it defaults to the sequence size
            Default: -1
        expected_initial_particles (int): Estimation of the number of particles in the first frames. If negative, it
            defaults to the average number of detections by frame in the sequence
            Default: -1
        expected_new_particles (int): Estimation of the number of new particle by frame
            Default: 0
        existence_prob (float): Minimum probability to confirm a track existence
            Default: 0.5
        termination_prob (float): Minimum probability before terminating a track
            Default: 1e-4
        motion (Motion): Motion of the particles (Brownian vs Directed vs Multi). If MULTI, motion 1 is
            brownian and motion 2 is directed.
            Default: Motion.BROWNIAN
        use_most_likely_model (bool): Use the most likely model to predict rather than a weighted predictions
            Default: True
        update_motion_{1, 2} (bool): Update the covariance (Q ?) online of motion 1 or 2
        xy_std_{1,2} (float): Used to define the covariance (Q) of the process. (Looking at Icy code,
            it is not clear what they truly are, depending on the motion the covariance Q is set a bit weirdly)
            Default: 3.0
        z_std_{1,2} (float): Same as `xy_std` but for the 3d axes (if any)
            Default: 3.0
        inertia (float): Probability to not switch of motion model (For MULTI motion)
            Default: 0.8
        gate_factor (float): Max mahalanobis distance for potential association
            Default: 4.0
        tree_depth (int): Number of frames to consider in the search tree
            Default: 4
    """

    # detections
    detecions_fpr: float = 0.1
    detecions_fnr: float = 0.1

    # Particles life
    expected_track_length: int = -1  # -1 => Default to size of the sequence
    expected_initial_particles: int = -1  # -1 => Default to average num of detections
    expected_new_particles: int = 10
    existence_prob: float = 0.5
    termination_prob: float = 1e-4

    # Motion
    motion: Motion = Motion.BROWNIAN
    use_most_likely_model: bool = True
    update_motion_1: bool = False  # Update std online for motion 1?
    update_motion_2: bool = False  # Update std online for motion 2?
    xy_std_1: float = 3.0  # Correspond to the first motion (brownian if multiple)
    z_std_1: float = 3.0  # Correspond to the first motion (brownian if multiple)
    xy_std_2: float = 3.0  # Correspond to the second motion (Always directed)
    z_std_2: float = 3.0  # Correspond to the second motion (Always directed)
    inertia: float = 0.8  # Prob to keep the same motion model

    # MHT
    gate_factor: float = 4.0
    tree_depth: int = 4

    def to_xml(self, detections_sequence: Sequence[byotrack.Detections]) -> ET.ElementTree:
        """Convert to xml format used by Icy

        Format example:

        .. code-block:: xml

            <root>
                <MHTconfiguration>
                    <detectionInput detectionRate="0.9" detectionSource="noSource" numberFalseDetection="20"/>
                    <targetExistence confirmationThreshold="0.5" terminationThreshold="1.0E-4" trackLength="200"/>
                    <motionModel IMMinertia="0.8" isDirectedMotion_1="true" isDirectedMotion_2="true"
                        isMostLikelyModel="true" isSingleMotion="false" updateCovariance_2="false" updateMotion_1="true"
                        yxDisplacement_1="2" yxDisplacement_2="3" zDisplacement_1="2" zDisplacement_2="3"/>
                    <mht gateFactor="4" mhtDepth="4" numberNewObjects="10" numberObjectsFirstFrame="50"/>
                    <ouput trackGroupName="mht-tracks-1"/>
                </MHTconfiguration>
            </root>

        Args:
            detections_sequence (Sequence[byotrack.Detections]): Detections (Required to set default values
                for some parameters)

        Returns:
            xml.etree.ElementTree.ElementTree: Xml tree of the configuration

        """
        num_frames = len(detections_sequence)
        mean_particles = sum(detections.length for detections in detections_sequence) / len(detections_sequence)

        root = ET.Element("root")
        config = ET.SubElement(root, "MHTconfiguration")
        ET.SubElement(
            config,
            "detectionInput",
            {
                "detection_rate": str(1 - self.detecions_fnr),
                "detectionSource": "noSource",
                "numberFalseDetection": str(int(mean_particles * self.detecions_fpr)),
            },
        )

        ET.SubElement(
            config,
            "targetExistence",
            {
                "confirmationThreshold": str(self.existence_prob),
                "terminationThreshold": str(self.termination_prob),
                "trackLength": str(num_frames if self.expected_track_length < 0 else self.expected_track_length),
            },
        )

        ET.SubElement(
            config,
            "motionModel",
            {
                "isSingleMotion": "false" if self.motion is Motion.MULTI else "true",
                "isMostLikelyModel": "true" if self.use_most_likely_model else "false",
                "updateMotion_1": "true" if self.update_motion_1 else "false",
                "updateCovariance_2": "true" if self.update_motion_2 else "false",
                "isDirectedMotion_1": "true" if self.motion is Motion.DIRECTED else "false",
                "isDirectedMotion_2": "true" if self.motion is Motion.MULTI else "false",
                "IMMinertia": str(self.inertia),
                "yxDisplacement_1": str(self.xy_std_1),
                "yxDisplacement_2": str(self.xy_std_2),
                "zDisplacement_1": str(self.z_std_1),
                "zDisplacement_2": str(self.z_std_2),
            },
        )

        ET.SubElement(
            config,
            "mht",
            {
                "gateFactor": str(self.gate_factor),
                "mhtDepth": str(self.tree_depth),
                "numberNewObjects": str(self.expected_new_particles),
                "numberObjectsFirstFrame": str(
                    mean_particles if self.expected_initial_particles < 0 else self.expected_initial_particles
                ),
            },
        )

        return ET.ElementTree(root)


class IcyEMHTLinker(byotrack.Linker):  # pylint: disable=too-few-public-methods
    """Run EMHT [4] from Icy [1]

    It is a wrapper around Icy's tracking code. It supports 2D and 3D data.

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
    3. [In ICY] Read rois, estimate or load emht parameters, run emht, export tracks to xml
    4. Read Icy tracks and return

    Note:
        This implementation requires Icy to be installed (https://icy.bioimageanalysis.org/download/)
        You should also install the Spot Tracking Blocks plugin
        (https://icy.bioimageanalysis.org/tutorial/how-to-install-an-icy-plugin/)

    Attributes:
        runner (byotrack.icy.IcyRunner): Icy runner
        motion (Motion): Prior on the underlying motion model (Brownian vs Directed vs Both)
            Given to the Icy block that estimates EMHT parameters. (See `full_specs` and `EMHTParameters` to have
            a finegrained control over the algorithm parameters)
            Default: Motion.BROWNIAN
        full_specs (Optional[EMHTParameters]): Full specification of the algorithm. If not provided,
            we use the estimation of EMHT parameters provided by Icy (with `motion` the only parameter to set).

    """

    rois_file = "_tmp_rois.xml"
    parameters_file = "_tmp_parameters.xml"
    tracks_file = "_tmp_tracks.xml"

    def __init__(
        self,
        icy_path: Optional[Union[str, os.PathLike]] = None,
        full_specs: Optional[EMHTParameters] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Constructor

        Args:
            icy_path (str | os.PathLike): Path to the icy jar (Icy is called with java -jar <icy_jar>)
                If not given, icy is searched in the PATH
            timeout (Optional[float]): Optional timeout in seconds as EMHT may enter an infinite loop.
        """
        super().__init__()
        self.runner = icy.IcyRunner(icy_path, timeout)
        self.motion = full_specs.motion if full_specs else Motion.BROWNIAN
        self.full_specs = full_specs

    def run(self, video, detections_sequence: Sequence[byotrack.Detections]) -> List[byotrack.Track]:
        try:
            if self.full_specs:
                self.full_specs.to_xml(detections_sequence).write(self.parameters_file)
            icy.save_detections(detections_sequence, self.rois_file)
            self._run_icy()

            tracks = icy.load_tracks(self.tracks_file)
            update_detection_ids(tracks, detections_sequence)

            # Sort tracks by starting time and then position (They seem to be randomly sorted otherwise and in addition
            # with EMC2 torch-tps approximate propagation it yields undeterministic behaviors)
            return sorted(tracks, key=lambda track: (track.start, track.points[0].sum().item()))
        finally:
            if os.path.exists(self.rois_file):
                os.remove(self.rois_file)
            if os.path.exists(self.parameters_file):
                os.remove(self.parameters_file)
            if os.path.exists(self.tracks_file):
                os.remove(self.tracks_file)

    def _run_icy(self):
        """Run the icy process"""

        if self.full_specs:
            if self.full_specs.motion != self.motion:
                warnings.warn(
                    f"""motion attribute ({self.motion}) != full_specs.motion ({self.full_specs.motion})

                    Will use motion attribute (and modify the specifications)
                    """
                )

                self.full_specs.motion = self.motion

            self.runner.run(
                PARAMETRIZED_PROTOCOL,
                rois=f'"{os.path.abspath(self.rois_file)}"',
                parameters=f'"{os.path.abspath(self.parameters_file)}"',
                tracks=f'"{os.path.abspath(self.tracks_file)}"',
            )
        else:
            directed, multi = self.motion.value

            self.runner.run(
                PROTOCOL,
                rois=f'"{os.path.abspath(self.rois_file)}"',
                tracks=f'"{os.path.abspath(self.tracks_file)}"',
                directed=directed,
                multi=multi,
            )

        # Has to check because icy do not return any non-zero return code
        if not os.path.exists(self.tracks_file):
            raise RuntimeError(
                """No track found from Icy.

            This probably results from a failure in Icy software. Please look at the Java Exception displayed by Icy.

            Note that Icy may fail the first time you launch this code because Icy updates the required protocols
            and is unable to continue to run after the update.
            """
            )
