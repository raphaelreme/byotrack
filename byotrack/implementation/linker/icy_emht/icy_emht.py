import enum
import os
import pathlib
from typing import Collection, Iterable, Optional, Union

import numpy as np

import byotrack
from byotrack import icy
from byotrack.api.parameters import ParameterEnum


PROTOCOL = pathlib.Path(__file__).parent / "emht_protocol.xml"


class IcyEMHTLinker(byotrack.Linker):  # pylint: disable=too-few-public-methods
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
        runner (byotrack.icy.IcyRunner): Icy runner
        motion (Motion): Prior on the underlying motion model (Brownian vs Directed vs Switching)
            Given to the Icy block that estimates EMHT parameters.
            Default: Motion.BROWNIAN

    ## XXX: Should we add update_cov (bool): Whether to update covariances in Kalman filters

    """

    class Motion(enum.Enum):
        """Different motion models:

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

    def __init__(self, icy_path: Optional[Union[str, os.PathLike]] = None) -> None:
        """Constructor

        Args:
            icy_path (str | os.PathLike): Path to the icy jar (Icy is called with java -jar <icy_jar>)
                If not given, icy is searched in the PATH
        """
        super().__init__()
        self.runner = icy.IcyRunner(icy_path)
        self.motion = IcyEMHTLinker.Motion.BROWNIAN

    def run(
        self, video: Iterable[np.ndarray], detections_sequence: Collection[byotrack.Detections]
    ) -> Collection[byotrack.Track]:
        try:
            icy.save_detections(detections_sequence, self.rois_file)
            self._run_icy()
            # Sort tracks by starting time and then position (They seem to be randomly sorted otherwise and in addition
            # with EMC2 torch-tps approximate propagation it yields undeterministic behaviors)
            return sorted(
                icy.load_tracks(self.tracks_file), key=lambda track: (track.start, track.points[0].sum().item())
            )
        finally:
            if os.path.exists(self.rois_file):
                os.remove(self.rois_file)
            if os.path.exists(self.tracks_file):
                os.remove(self.tracks_file)

    def _run_icy(self):
        """Run the icy process"""
        directed, multi = self.motion.value

        self.runner.run(
            PROTOCOL,
            rois=os.path.abspath(self.rois_file),
            tracks=os.path.abspath(self.tracks_file),
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
