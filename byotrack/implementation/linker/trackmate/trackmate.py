import os
import pathlib
from typing import Collection, Iterable, Union

import numpy as np

import byotrack
from byotrack import fiji
from byotrack.api.parameters import ParameterBound


SCRIPT = pathlib.Path(__file__).parent / "_trackmate.py"


class TrackMateLinker(byotrack.Linker):  # pylint: disable=too-few-public-methods
    """Run TrackMate [7, 8] from Fiji [6]

    Wrapper around TrackMate, using Fiji headless call/

    About TrackMate:
    It is a global distance minimization tracking. It supports multiple algorithms, but we have only wrapped
    the one from [7]. It solves frame-to-frame GDM between detections. And then sovle a GDM between tracklets
    to correct them (merge, split, etc).

    Here we rely on the handmade ImageJ script "_trackmate.py" that expects detections as instance segmentation
    and the hyperparameters of the linking process.

    The workflow is:

    1. Save detections to segmentation format
    2. Run the Fiji script
    3. [In Fiji] Read segmentation, extract detections, run linking [5], export tracks to xml
    4. Read Fiji tracks and return

    Note:
        This implementation requires Fiji to be installed (https://imagej.net/downloads)

    Note:
        In case of missed detections, positions are filled with nan. To fill nan with true values, use an Interpolator

    Attributes:
        runner (byotrack.fiji.FijiRunner): Fiji runner
        max_link_cost (float): The max distance between two consecutive spots, in pixels, allowed for creating links.
        max_gap (int): The max difference in time-points between two spots to allow for linking.
            For instance a value of 2 means that the tracker will be able to make a link between a spot in frame t
            and a successor spots in frame t+2, effectively bridging over one missed detection in one frame.
        max_gap_cost (float): The max distance between two spots, in pixels, allowed for creating links over
            missing detections.

    """

    detections_file = "_tmp_detections.tif"
    tracks_file = "_tmp_tracks.xml"

    parameters = {
        "max_link_cost": ParameterBound(0.0, 1000.0),
        "max_gap": ParameterBound(0.0, 1000.0),
        "max_gap_cost": ParameterBound(0.0, 1000.0),
    }

    def __init__(
        self, fiji_path: Union[str, os.PathLike], max_link_cost: float, max_gap: int, max_gap_cost: float
    ) -> None:
        """Constructor

        Args:
            fiji_path (str | os.PathLike): Path to the fiji executable
                The executable can be found inside the installation folder of Fiji.
                Linux: Fiji.app/ImageJ-<os>.exe
                Windows: Fiji.app/ImageJ-<os>.exe
                MacOs: Fiji.app/Contents/MacOs/ImageJ-<os>

        """
        super().__init__()
        self.runner = fiji.FijiRunner(fiji_path)
        self.max_link_cost = max_link_cost
        self.max_gap = max_gap
        self.max_gap_cost = max_gap_cost

    def run(
        self, video: Iterable[np.ndarray], detections_sequence: Collection[byotrack.Detections]
    ) -> Collection[byotrack.Track]:
        try:
            fiji.save_detections(detections_sequence, self.detections_file)
            self._run_fiji()
            # Sort tracks by starting time and then position (Prevents undeterministic behaviors with some post
            # processing steps)
            return sorted(
                fiji.load_tracks(self.tracks_file), key=lambda track: (track.start, track.points[0].sum().item())
            )
        finally:
            if os.path.exists(self.detections_file):
                os.remove(self.detections_file)
            if os.path.exists(self.tracks_file):
                os.remove(self.tracks_file)

    def _run_fiji(self):
        """Run the fiji process"""
        self.runner.run(
            SCRIPT,
            detections=os.path.abspath(self.detections_file),
            tracks=os.path.abspath(self.tracks_file),
            max_link_cost=self.max_link_cost,
            max_gap=self.max_gap,
            max_gap_cost=self.max_gap_cost,
        )

        # Has to check because fiji do not return any non-zero return code
        if not os.path.exists(self.tracks_file):
            raise RuntimeError(
                """No track found from Fiji.

            This probably results from a failure inside Fiji script.
            Please look at the Java Exception displayed by Fiji.
            """
            )
