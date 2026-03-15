from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
from typing import TYPE_CHECKING

import byotrack
from byotrack import fiji
from byotrack.api.tracks import update_detection_ids

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

SCRIPT = pathlib.Path(__file__).parent / "_trackmate.py"


@dataclasses.dataclass
class TrackMateParameters:
    """Parameters of TrackMate [7, 8].

    We do *not* support Features penalties yet. Moreover track splitting and merging is currently *not* supported in
    ByoTrack.

    To use kalman filtering in addition to sparse lap tracking, set the `kalman_search_radius` value. By default,
    no kalman filtering is used.

    The main parameters to set are:

    * `linking_max_distance`: The distance threshold for frame-to-frame linking.
    * `max_frame_gap` and `gap_closing_max_distance`: The temporal and spatial distances for track stitching.
    * `kalman_search_radius`: The distance threshold for kalman linking (Switch to AdvancedKalmanTracker).

    Attributes:
        allow_gap_closing (bool): Use a second lap to solve tracklet stitching (Closing temporal gap
            between previously established tracks)
            Default: True
        allow_track_splitting (bool): NOT SUPPORTED (Useful for cell tracking)
            Default: False
        allow_track_merging (bool): NOT SUPPORTED (To allow split and remerge and handle split detections)
            Default: False
        alternative_linking_cost_factor (float): Alternative linking cost
            Default: 1.05
        blocking_value (float): Cost for mon-physical, forbidden links.
            Default: inf
        cutoff_percentile (float): Cutoff percentile
            Default: 0.9
        max_frame_gap (int): The max difference in frames between two spots to allow for linking.
            For instance a value of 2 means that the tracker will be able to make a link between a spot in frame t
            and a successor spots in frame t+2, effectively bridging over one missed detection in one frame.
            Default: 2 (1 missed detections)
        linking_max_distance (float): The max distance between two consecutive spots, in physical units, allowed
            for creating links. If using kalman filters: this is the initial search radius, in physical units,
            specifying how far two spots can be apart when initiating new tracks. (See `kalman_search_radius`)
            Default: 15.0
        gap_closing_max_distance (float): Gap-closing max spatial distance. The max distance between two spots,
            in physical units, allowed for creating links over missing detections.
            Default: 15.0
        merging_max_distance (float): Unused. Track merging max spatial distance.
            Default: 15.0
        splitting_max_distance (float): Unused. Track splitting max spatial distance.
            Default: 15.0
        kalman_search_radius (float | None): Set this parameter to use the AdvancedKalmanTracker. The max search
            radius specifying how far from a predicted position the tracker should look for candidate spots.
            Default: None (Without kalman filters)

    """

    allow_gap_closing: bool = True
    # Track merging/splitting is not supported
    allow_track_splitting: bool = False
    allow_track_merging: bool = False

    alternative_linking_cost_factor: float = 1.05
    blocking_value: float = float("inf")
    cutoff_percentile: float = 0.9

    max_frame_gap: int = 2

    linking_max_distance: float = 15.0
    gap_closing_max_distance: float = 15.0
    merging_max_distance: float = 15.0
    splitting_max_distance: float = 15.0

    # linking_feature_penalties
    # gap_closing_feature_penalties
    # merging_feature_penalties
    # splitting_feature_penalties

    kalman_search_radius: float | None = None

    def save(self, path: str | os.PathLike) -> None:
        """Save the parameters to the given path.

        Uses json format.

        Args:
            path (str | os.PathLike): Path to the output file
        """
        path = pathlib.Path(path)
        with path.open("w", encoding="utf-8") as file:
            json.dump(dataclasses.asdict(self), file)


class TrackMateLinker(byotrack.Linker):
    """Run TrackMate [7, 8] from Fiji [6].

    Wrapper around TrackMate, using Fiji headless call. Supports 2D and 3D frames.

    About TrackMate:
    It is a global distance minimization tracking. It supports multiple algorithms. We have wrapped the more advanced
    ones (SparseLapTracker and AdvancedKalmanTracker that both follows [7]). It solves frame-to-frame GDM between
    detections. And then solve a GDM between tracklets to correct them (stitch, merge, split).
    The AdvancedKalmanTracker additionally uses kalman filters to estimate velocities of particles.

    Here we rely on the handmade ImageJ script "_trackmate.py" that expects detections as instance segmentation
    and the hyperparameters of the linking process.

    We do not support track splitting and merging yet, neither the use of feature-based cost.

    The workflow is:

    1. Save detections to segmentation format
    2. Run the Fiji script
    3. [In Fiji] Read segmentation, load parameters, extract detections, run linking [5], export tracks to xml
    4. Read Fiji tracks and return

    Note:
        This implementation requires Fiji to be installed (https://imagej.net/downloads)

    Note:
        In case of missed detections, positions are filled with nan. To fill nan with true values, use an Interpolator

    Attributes:
        runner (byotrack.fiji.FijiRunner): Fiji runner
        specs (TrackmateParameters): Parameters specifications of the algorithm

    """

    detections_file = "_tmp_detections.tif"
    parameters_file = "_tmp_parameters.json"
    tracks_file = "_tmp_tracks.xml"

    def __init__(self, fiji_path: str | os.PathLike, specs: TrackMateParameters) -> None:  # noqa: D417
        """Constructor.

        Args:
            fiji_path (str | os.PathLike): Path to the fiji executable
                The executable can be found inside the installation folder of Fiji.
                Linux: Fiji.app/ImageJ-<os>
                Windows: Fiji.app/ImageJ-<os>.exe
                MacOs: Fiji.app/Contents/MacOs/ImageJ-<os>

        """
        super().__init__()
        self.runner = fiji.FijiRunner(fiji_path)
        self.specs = specs

    @override
    def run(self, video, detections_sequence: Sequence[byotrack.Detections]) -> list[byotrack.Track]:
        try:
            self.specs.save(self.parameters_file)
            fiji.save_detections(detections_sequence, self.detections_file)
            self._run_fiji()

            tracks = fiji.load_tracks(self.tracks_file)
            update_detection_ids(tracks, detections_sequence)

            # Sort tracks by starting time and then position (Prevents indeterministic behaviors with some post
            # processing steps)
            return sorted(tracks, key=lambda track: (track.start, track.points[0].sum().item()))
        finally:
            if pathlib.Path(self.detections_file).exists():
                pathlib.Path(self.detections_file).unlink()
            if pathlib.Path(self.parameters_file).exists():
                pathlib.Path(self.parameters_file).unlink()
            if pathlib.Path(self.tracks_file).exists():
                pathlib.Path(self.tracks_file).unlink()

    def _run_fiji(self):
        """Run the fiji process."""
        self.runner.run(
            SCRIPT,
            detections=pathlib.Path(self.detections_file).resolve(),
            parameters=pathlib.Path(self.parameters_file).resolve(),
            tracks=pathlib.Path(self.tracks_file).resolve(),
        )

        # Has to check because fiji do not return any non-zero return code
        if not pathlib.Path(self.tracks_file).exists():
            raise RuntimeError(
                """No track found from Fiji.

            This probably results from a failure inside Fiji script.
            Please look at the Java Exception displayed by Fiji.
            """
            )
