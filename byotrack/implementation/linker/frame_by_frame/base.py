from abc import abstractmethod
import dataclasses
from typing import List, Optional, Tuple, Union
import warnings

import enum

import numpy as np
import pylapy
import torch

import byotrack

from .greedy_lap import greedy_assignment_solver


class AssociationMethod(enum.Enum):
    """Association methods (Greedy or Jonker-Volgenant)

    * GREEDY
        Select the best match between tracks and detections iteratively until
        no match can be selected below the cost limit eta. It is usually not optimal for tracking
        but it is much faster.
    * OPT_HARD | SPARSE_OPT_HARD
        Solve the linear association problem (see `pylapy`).
        Hard threshold the association matrix with the cost limit eta.
        Use the sparse version to increase speed with numerous particles.
    * OPT_SMOOTH | SPARSE_OPT_SMOOTH
        Solve a cost_limit extended association problem (see `pylapy`)
        It relaxes the linear association problem, allowing to not link a node
        for the cost limit eta.
        Use the sparse version to increase speed with numerous particles.
    """

    GREEDY = "greedy"
    OPT_HARD = "opt_hard"
    OPT_SMOOTH = "opt_smooth"
    SPARSE_OPT_HARD = "sparse_opt_hard"
    SPARSE_OPT_SMOOTH = "sparse_opt_smooth"

    def solve(self, cost: torch.Tensor, eta: float = np.inf) -> torch.Tensor:
        """Solve tracks-to-detections association

        Args:
            cost (torch.Tensor): Cost matrix
                    Shape: (N, M), dtype: float
            eta (float): Cost limit
                Default: inf (No thresholding)

        Returns:
            torch.Tensor: Links (i, j)
                Shape: (L, 2), dtype: int32
        """
        if self == AssociationMethod.SPARSE_OPT_HARD:
            return torch.tensor(pylapy.LapSolver().sparse_solve(cost.numpy(), eta, hard=True).astype(np.int32))

        if self == AssociationMethod.SPARSE_OPT_SMOOTH:
            return torch.tensor(pylapy.LapSolver().sparse_solve(cost.numpy(), eta, hard=False).astype(np.int32))

        if self == AssociationMethod.OPT_HARD:
            cost[cost > eta] = np.inf
            return torch.tensor(pylapy.LapSolver().solve(cost.numpy()).astype(np.int32))

        if self == AssociationMethod.OPT_SMOOTH:
            return torch.tensor(pylapy.LapSolver().solve(cost.numpy(), eta).astype(np.int32))

        return torch.tensor(greedy_assignment_solver(cost.numpy(), eta).astype(np.int32))


class TrackHandler:
    """Handle a track during the tracking procedure

    It accumulates the track data at each new association and store the optional motion model data.

    A TrackHandler is created for each unlinked detections in the linking process and then updated with
    the following associated detections. At the beginining, the track is considered HYPOTHETICAL.
    For a track to be considered valid, it requires n_valid consecutive associated detections after the track creation
    (state: HYPOTHETICAL => VALID). It a miss detection occurs during this time interval, then the track is deleted
    and considered invalid (state: HYPOTHETICAL => INVALID).

    Once confirmed, a VALID track is resilient to miss detections, waiting n_gap frames before ending the track
    (VALID => FINISHED).

    Attributes:
        n_valid (int): Number of frames with a correct association required to validate the track at its creation.
        n_gap (int): Number of frames with no association before the track termination.
        start (int): Starting frame of the track
        identifier (int): Identifier of the track handler (and of the track)
        track_state (TrackState): Current state of the handler
        last_association (int): Number of frames since the last association
        detection_ids (List[int]): Identifiers of the associated detection (-1 if None)
        track_ids (List[int]): Index of the track at each frame in the `linker.active_tracks` list.
            It allows the linker to store data as tensor and be able to rebuild tracks at the end.

    """

    class TrackState(enum.IntEnum):
        """TrackState of a TrackHandler

        * HYPOTHETICAL
            Initial state before validation of the track.
        * VALID
            The track has been validated and is still active
        * FINISHED
            The track is valid and finished
        * INVALID
            The track is not valid and deleted

        """

        HYPOTHETICAL = 0
        VALID = 1
        FINISHED = 2
        INVALID = 3

    def __init__(self, n_valid: int, n_gap: int, start: int, identifier: int) -> None:
        self.n_valid = n_valid
        self.n_gap = n_gap
        self.start = start
        self.identifier = identifier
        self.track_state = TrackHandler.TrackState.HYPOTHETICAL
        self.last_association = 0
        self.detection_ids: List[int] = []
        self.track_ids: List[int] = []

    def __len__(self) -> int:
        return len(self.detection_ids) - self.last_association

    def is_active(self) -> bool:
        return self.track_state < 2

    def update(self, frame_id: int, detection_id: int) -> None:
        """Update track handler. It stores the detection_id and update the track state.

        It should be called for each time frame and each active track.

        Args:
            frame_id (int): The current frame. This is given for safety checks
                to ensure that the Linker and TrackHandler agree.
            detection_id (int): Detection id in the Detections object.
                -1 if not associated to a particular detection.

        """
        assert self.is_active()
        assert len(self.track_ids) == len(
            self.detection_ids
        ), "The linker should call `update` then `register_track_id` at each linking step"
        assert (
            self.start + len(self.detection_ids) == frame_id
        ), "The linker should update each active track on each time frame."

        self.detection_ids.append(detection_id)

        if detection_id == -1:  # Not associated
            self.last_association += 1

            if self.track_state == TrackHandler.TrackState.HYPOTHETICAL:
                self.track_state = TrackHandler.TrackState.INVALID

            if self.last_association > self.n_gap:
                self.track_state = TrackHandler.TrackState.FINISHED
        else:
            self.last_association = 0

            if self.track_state == TrackHandler.TrackState.HYPOTHETICAL:
                if len(self.detection_ids) >= self.n_valid:
                    self.track_state = TrackHandler.TrackState.VALID

    def register_track_id(self, track_id: int) -> None:
        """For still active tracks, it registers the track id after the update step.

        Args:
            track_id (int): The index of the track in `linker.active_tracks` at this time frame.
        """
        self.track_ids.append(track_id)


class OnlineFlowExtractor:
    """Extract optical flow maps online from a video"""

    def __init__(self, optflow: byotrack.OpticalFlow) -> None:
        self.optflow = optflow
        self.flow_map: Optional[np.ndarray] = None
        self.src: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset the flow extractor"""
        self.flow_map = None
        self.src = None

    def update(self, frame: np.ndarray) -> None:
        """Extract the flow for a new given frame

        It will compute and register the flow map between the last given frame
        and the current frame.

        Args:
            frame (np.ndarray): Current frame of the video

        """
        dst = self.optflow.preprocess(frame)
        if self.src is None:
            self.src = dst

        self.flow_map = self.optflow.compute(self.src, dst)
        self.src = dst


@dataclasses.dataclass
class FrameByFrameLinkerParameters:  # pylint: disable=too-many-instance-attributes
    """Parameters of the abstract FrameByFrameLinker

    Attributes:
        association_threshold (float): This is the main hyperparameter, it defines the threshold on the distance used
            not to link tracks with detections. It prevents to link with false positive detections.
            Default: 5 pixels
        n_valid (int): Number of frames with a correct association required to validate the track at its creation.
            Default: 3
        n_gap (int): Number of frames with no association before the track termination.
            Default: 3
        association_method (AssociationMethod): The frame-by-frame association to use. See `AssociationMethod`.
            It can be provided as a string. (Choice: GREEDY, OPT_HARD, OPT_SMOOTH)
            Default: OPT_SMOOTH
    """

    def __init__(
        self,
        association_threshold: float = 5.0,
        *,
        n_valid=3,
        n_gap=3,
        association_method: Union[str, AssociationMethod] = AssociationMethod.OPT_SMOOTH,
    ):
        self.association_threshold = association_threshold
        self.n_valid = n_valid
        self.n_gap = n_gap
        self.association_method = (
            association_method
            if isinstance(association_method, AssociationMethod)
            else AssociationMethod[association_method.upper()]
        )

    association_threshold: float = 5.0
    n_valid: int = 3
    n_gap: int = 3
    association_method: AssociationMethod = AssociationMethod.OPT_SMOOTH


class FrameByFrameLinker(byotrack.OnlineLinker):
    """Links detections online using frame-by-frame association

    Abstract class for frame-by-frame linker. It decomposes the update step in 5 parts:

    1. Optional optical flow computations (Handled by this class)
    2. Motion modeling to predict track positions (`motion_model`)
    3. Track-to-detection cost computation (`cost`)
    4. Solving the linear association problem (Handled by this class)
    5. Post matching update to handle tracks (`post_association`)

    The association relies on the AssociationMethod enum and tracks handling is done with
    TrackHandler.

    It follows the tracks handling strategy describe in KOFT [9].

    Attributes:
        specs (FrameByFrameLinkerParameters): Parameters specifications of the algorithm.
            See `FrameByFrameLinkerParameters`.
        optflow (Optional[OnlineFlowExtractor]): Optional wrapper around the given optional OpticalFlow that
            will extract flow maps of the video online. (The underlying OpticalFlow object is accessible in
            self.optflow.optflow)
            Default: None
        features_extractor (Optional[FeaturesExtractor]): Optional features extractor that will extract
            features for the detections, which could be useful for tracking.
            Default: None
        save_all (bool): Save metadata useless for the final building of tracks
            but that could be useful for analysis. For instance, it will keep invalid tracks.
            Or the computed features inside the Detections objects.
        frame_id (int): Current frame id of the linking process
        inactive_tracks (List[TrackHandler]): Terminated tracks
        active_tracks (List[TrackHandler]): Current track handlers
        all_positions (List[torch.Tensor]): Positions of the active tracks at each seen frames.
            Using the valid track handlers `track_ids`, it allows the reconstruction of tracks.

    """

    def __init__(
        self,
        specs: FrameByFrameLinkerParameters,
        optflow: Optional[byotrack.OpticalFlow] = None,
        features_extractor: Optional[byotrack.FeaturesExtractor] = None,
        save_all=False,
    ) -> None:
        super().__init__()
        self.specs = specs
        self.optflow = OnlineFlowExtractor(optflow) if optflow else None
        self.features_extractor = features_extractor
        self.save_all = save_all
        self.frame_id = -1
        self.inactive_tracks: List[TrackHandler] = []
        self.active_tracks: List[TrackHandler] = []
        self.all_positions: List[torch.Tensor] = []

    def reset(self) -> None:
        super().reset()
        if self.optflow:
            self.optflow.reset()
        self.frame_id = -1
        self.inactive_tracks = []
        self.active_tracks = []
        self.all_positions = []

    def collect(self) -> List[byotrack.Track]:
        tracks = []
        for handler in self.inactive_tracks + self.active_tracks:
            if handler.track_state in (handler.TrackState.INVALID, handler.TrackState.HYPOTHETICAL):
                continue  # Ignore non-valid tracks

            points = torch.cat(
                [
                    positions[track_id : track_id + 1]
                    for track_id, positions in zip(
                        handler.track_ids[: len(handler)], self.all_positions[handler.start :]
                    )
                ]
            )

            tracks.append(
                byotrack.Track(
                    handler.start,
                    points,
                    handler.identifier,
                    torch.tensor(handler.detection_ids[: len(handler)], dtype=torch.int32),
                )
            )
        return tracks

    @abstractmethod
    def motion_model(self) -> None:
        """Optional modelisation of motion for tracks

        It can be used to update some internal state of the tracker after the optical flow computation
        and before the distance computation.
        """

    @abstractmethod
    def cost(self, frame: np.ndarray, detections: byotrack.Detections) -> Tuple[torch.Tensor, float]:
        """Compute the association cost between active tracks and detections

        It also returns the threshold to use (Depending on the dist you use, association_threshold
        could be related to a more meaning full quantity than the cost itself).
        For instance, when using a squared Euclidean distance, the association threshold could be
        express as the distance in pixel, and this function could square it.
        For likelihood association, you could provide the association threshold as a probability
        and use -log(threshold) as the true threshold. (See `KalmanLinker` and `NearestNeighborLinker`)

        Args:
            frame (np.ndarray): The current frame of the video
                Shape: (H, W, C), dtype: float
            detections (byotrack.Detections): Detections for the given frame

        Returns:
            torch.Tensor: The cost matrix between active tracks and detections
                Shape: (n_tracks, n_dets), dtype: float32
            float: The association threshold to use.
                It can be different than `self.association_threshold` depeding on the dist build here

        """

    @abstractmethod
    def post_association(self, frame: np.ndarray, detections: byotrack.Detections, links: torch.Tensor):
        """Update the tracks and the internal variables of the tracker

        It should call the `update` method of each active tracks and update any internal model/data.
        It should also create new track handlers for each extra detection.
        Finally, it is also responsible to register the position of each active track in `all_positions`
        for the current time frame.

        Args:
            frame (np.ndarray): The current frame of the video
                Shape: (H, W, C), dtype: float
            detections (byotrack.Detections): Detections for the given frame
            links (torch.Tensor): The links made between active tracks and the detections
                Shape: (L, 2), dtype: int32

        """

    def update_active_tracks(self, links: torch.Tensor) -> torch.Tensor:
        """Calls `update` for active tracks and return a boolean mask that indicates which track is still active

        Tracks that are terminated are stored inside `inactive_tracks` and dropped from `active_tracks`.

        It can be called inside `post_association` to simplify the code.

        Args:
            links (torch.Tensor): The links made between active tracks and the detections
                Shape: (L, 2), dtype: int32

        Returns:
            torch.Tensor: Boolean tensor indicating True for still active tracks

        """
        i_to_j = torch.full((len(self.active_tracks),), -1, dtype=torch.int32)
        i_to_j[links[:, 0]] = links[:, 1]
        active_mask = torch.full((len(self.active_tracks),), False)
        still_active = []
        for i, track in enumerate(self.active_tracks):
            track.update(self.frame_id, int(i_to_j[i].item()))

            # Check if track is still active
            if track.is_active():
                still_active.append(track)
                active_mask[i] = True
            elif track.track_state == TrackHandler.TrackState.FINISHED or self.save_all:
                self.inactive_tracks.append(track)

        self.active_tracks = still_active

        return active_mask

    def handle_extra_detections(self, detections: byotrack.Detections, links: torch.Tensor) -> torch.Tensor:
        """Handle extra detections by creating new track handlers

        It can be called inside `post_association` to create track handlers from extra detections. It will
        return the extra detections positions and ids to be further used by `post_association`.

        Args:
            detections (byotrack.Detections): Detections for the given frame
            links (torch.Tensor): The links made between active tracks and the detections
                Shape: (L, 2), dtype: int32

        """
        # Find unmatched measures
        unmatched = torch.full((len(detections),), True)
        unmatched[links[:, 1]] = False
        unmatched_ids = torch.arange(len(detections))[unmatched]
        # unmatched_measures = detections.position[unmatched]

        # Create a new active track for each unmatched measure
        for i in range(unmatched_ids.shape[0]):
            handler = TrackHandler(
                self.specs.n_valid, self.specs.n_gap, self.frame_id, len(self.inactive_tracks) + len(self.active_tracks)
            )
            handler.update(self.frame_id, int(unmatched_ids[i]))
            self.active_tracks.append(handler)

        return unmatched

    def update(self, frame: np.ndarray, detections: byotrack.Detections) -> None:
        self.frame_id += 1

        # Compute features if the extractor is given and register inside the detections
        # Do not recompute the features if some are already registered
        remove_feats = False
        if self.features_extractor is not None:
            if "features" in detections.data:
                warnings.warn("Some features are already computed. They will be used.")
            else:
                remove_feats = True
                self.features_extractor.register(frame, detections)

        # Compute the flow map if optflow given
        if self.optflow is not None:
            self.optflow.update(frame)

        self.motion_model()
        cost, threshold = self.cost(frame, detections)
        links = self.specs.association_method.solve(cost, threshold)

        self.post_association(frame, detections, links)

        assert len(self.all_positions[-1]) == len(
            self.active_tracks
        ), "Wrong implementation of post_association. It should register the positions of each active tracks"
        for i, track in enumerate(self.active_tracks):
            track.register_track_id(i)

        # Remove the computed features if save_all is False
        if not self.save_all and remove_feats:
            detections.data.pop("features")
