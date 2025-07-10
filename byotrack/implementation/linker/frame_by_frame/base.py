from abc import abstractmethod
import dataclasses
from typing import Dict, List, Optional, Tuple, Union
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


class TrackHandler:  # pylint: disable=too-many-instance-attributes
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
        merge_id (int): Identifier to an optional merged track handler (See `Tracks.merge_id`)
        parent_id (int): Identifier to an optional parent track handler (See `Tracks.parent_id`)
        is_split (bool): Just to know if the track splits

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
        self.merge_id = -1
        self.parent_id = -1
        self.is_split = False

    def __len__(self) -> int:
        if self.merge_id != -1 or self.is_split:
            return len(self.detection_ids)  # Last points counts

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

    Note:
        The merging and splitting features is still experimental.

    Attributes:
        association_threshold (float): This is the main hyperparameter, it defines the threshold on the distance used
            not to link tracks with detections. It prevents to link with false positive detections.
            Default: 5 pixels
        n_valid (int): Number associated detections required to validate the track after its creation.
            Default: 3
        n_gap (int): Number of consecutive frames without association before the track termination.
            Default: 3
        association_method (AssociationMethod): The frame-by-frame association to use. See `AssociationMethod`.
            It can be provided as a string. (Choice: GREEDY, [SPARSE_]OPT_HARD, [SPARSE_]OPT_SMOOTH)
            Default: OPT_SMOOTH
        anisotropy (Tuple[float, float, float]): Anisotropy of images (Ratio of the pixel sizes
            for each axis, depth first). This will be used to scale distances.
            Default: (1., 1., 1.)
        split_factor (float): Allow splitting of tracks, using a second association step.
            The association threshold in this case is `split_factor * association_threshold`.
            Default: 0.0 (No splits)
        merge_factor (float): Allow merging of tracks, using a second association step.
            The association threshold in this case is `merge_factor * association_threshold`.
            Default: 0.0 (No merges)

    """

    def __init__(
        self,
        association_threshold: float = 5.0,
        *,
        n_valid=3,
        n_gap=3,
        association_method: Union[str, AssociationMethod] = AssociationMethod.OPT_SMOOTH,
        anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        split_factor: float = 0.0,
        merge_factor: float = 0.0,
    ):
        self.association_threshold = association_threshold
        self.n_valid = n_valid
        self.n_gap = n_gap
        self.association_method = (
            association_method
            if isinstance(association_method, AssociationMethod)
            else AssociationMethod[association_method.upper()]
        )
        self.anisotropy = anisotropy
        self.split_factor = split_factor
        self.merge_factor = merge_factor

        if merge_factor > 1.0 or split_factor > 1.0:
            warnings.warn("Merge or split factors should be lower than 1")

    association_threshold: float = 5.0
    n_valid: int = 3
    n_gap: int = 3
    association_method: AssociationMethod = AssociationMethod.OPT_SMOOTH
    anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    split_factor: float = 0.0
    merge_factor: float = 0.0


class FrameByFrameLinker(byotrack.OnlineLinker):  # pylint: disable=too-many-instance-attributes
    """Links detections online using frame-by-frame association

    Abstract class for frame-by-frame linker. It decomposes the update step in 6 parts:

    1. Optional optical flow computations (Handled by this class with the `optflow` given)
    2. Motion modeling to predict track positions (`motion_model`)
    3. Features extraction (handled by this class with the `features_extractor` given)
    4. Track-to-detection cost computation (`cost`)
    5. Solving the linear association problem (handled in `associate`)
    6. Post matching update to handle tracks (`post_association`)

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
        split_links (torch.Tensor): Current split_links
            shape: (L', 2), dtype: int32
        merge_links (torch.Tensor): Current merge_links
            shape: (L'', 2), dtype: int32

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
        self.active_mass = torch.zeros((0,), dtype=torch.float32)

        # Instantaneous useful quantities
        self._links = torch.zeros((0, 2), dtype=torch.int32)
        self._split_links = torch.zeros((0, 2), dtype=torch.int32)
        self._merge_links = torch.zeros((0, 2), dtype=torch.int32)
        self._unmatched_detections = torch.full((0,), True)
        self._next_identifier = 0

    def reset(self, dim=2) -> None:
        super().reset(dim)
        if self.optflow:
            self.optflow.reset()
        self.frame_id = -1
        self.inactive_tracks = []
        self.active_tracks = []
        self.all_positions = []
        self.active_mass = torch.zeros((0,), dtype=torch.float32)
        self._links = torch.zeros((0, 2), dtype=torch.int32)
        self._split_links = torch.zeros((0, 2), dtype=torch.int32)
        self._merge_links = torch.zeros((0, 2), dtype=torch.int32)
        self._unmatched_detections = torch.full((0,), True)
        self._next_identifier = 0

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
                    parent_id=handler.parent_id,
                    merge_id=handler.merge_id,
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
                Shape: (n_tracks, n_dets), dtype: float
            float: The association threshold to use.
                It can be different than `self.association_threshold` depeding on the dist build here

        """

    @abstractmethod
    def post_association(self, frame: np.ndarray, detections: byotrack.Detections, active_mask: torch.Tensor):
        """Update the internal state of the tracker after `update_active_tracks`

        It should update any internal model/data. It is also responsible to register the position of each active
        track in `all_positions` for the current time frame.

        Args:
            frame (np.ndarray): The current frame of the video
                Shape: (H, W, C), dtype: float
            detections (byotrack.Detections): Detections for the given frame
            active_mask (torch.Tensor): Boolean tensor indicating True for still active tracks
                Shape: (N_tracks), dtype: bool

        """

    def update_active_tracks(  # pylint: disable=too-many-branches,too-many-statements,too-many-locals
        self, detections: byotrack.Detections
    ) -> torch.Tensor:
        """Updates tracks handler and creates new ones for extra detections

        Tracks that are terminated are stored inside `inactive_tracks` and dropped from `active_tracks`.
        It is called by update before `post_association`.

        It also handles merges and splits. In the case of some specific merges, it may change a few links.
        The updated links are returned, with a still_active mask for tracks and a new_track mask
        for detections.

        Args:
            detections (byotrack.Detections): Detections for the given frame

        Returns:
            torch.Tensor: Boolean tensor indicating True for still active tracks
                Shape: (N_tracks), dtype: bool

        """
        if self._split_links.shape[0] + self._merge_links.shape[0] == 0:  # Fall back to old simpler version
            active_mask = self._update_active_tracks()
            self._handle_extra_detections(detections)
            return active_mask

        # Ugly to handle merge and splits smoothly... If you do not care about that, read only the old simpler version
        # Or even use offline split and merge strategies.

        # Create new tracks from unmatched measures
        new_tracks: List[TrackHandler] = []
        for i in torch.arange(len(detections))[self._unmatched_detections].tolist():
            track = TrackHandler(
                self.specs.n_valid,
                self.specs.n_gap,
                self.frame_id,
                self._next_identifier,
            )
            self._next_identifier += 1
            track.update(self.frame_id, i)
            new_tracks.append(track)

        # Lots of useful identifiers mapping
        det_to_new_track = torch.full((len(detections),), -1, dtype=torch.int32)
        det_to_new_track[torch.arange(len(detections))[self._unmatched_detections]] = torch.arange(
            len(new_tracks), dtype=torch.int32
        )

        track_to_det = torch.full((len(self.active_tracks),), -1, dtype=torch.int32)
        track_to_det[self._links[:, 0]] = self._links[:, 1]
        det_to_track = torch.full((len(detections),), -1, dtype=torch.int32)
        det_to_track[self._links[:, 1]] = self._links[:, 0]

        track_to_det_split = torch.full((len(self.active_tracks),), -1, dtype=torch.int32)
        track_to_det_split[self._split_links[:, 0]] = self._split_links[:, 1]

        track_to_det_merge = torch.full((len(self.active_tracks),), -1, dtype=torch.int32)
        track_to_det_merge[self._merge_links[:, 0]] = self._merge_links[:, 1]
        det_to_track_merge = torch.full((len(detections),), -1, dtype=torch.int32)
        det_to_track_merge[self._merge_links[:, 1]] = self._merge_links[:, 0]

        # Update active tracks (for merges and splits, they are still active but replaced by a new handler)
        active_mask = torch.full((len(self.active_tracks),), False)
        still_active: List[TrackHandler] = []
        merges_to_ref: List[TrackHandler] = []
        det_to_merge_id: Dict[int, int] = {}

        for i, track in enumerate(self.active_tracks):
            if track_to_det_split[i] != -1:  # The track splits
                # For the corner case of a hypothetical track, it is now valid (afterall we found 2 detections for it)
                track.track_state = TrackHandler.TrackState.FINISHED
                track.is_split = True
                self.inactive_tracks.append(track)

                # Replace by a new VALID handler
                other = TrackHandler(self.specs.n_valid, self.specs.n_gap, self.frame_id, self._next_identifier)
                self._next_identifier += 1
                other.track_state = TrackHandler.TrackState.VALID

                # Set parent for both new tracks
                other.parent_id = track.identifier
                new_tracks[det_to_new_track[track_to_det_split[i]]].parent_id = track.identifier

                # Swap other with track (the splitted one is finished)
                track = other

            if track_to_det[i] == -1 and track_to_det_merge[i] != -1:  # The track is the second branch of a merge
                assert det_to_track[track_to_det_merge[i]] != -1
                # Get the first branch
                other = self.active_tracks[det_to_track[track_to_det_merge[i]]]

                # If hypothetical, nothing to do, we do not merge with hypothetical tracks
                if track.track_state != TrackHandler.TrackState.HYPOTHETICAL:
                    if other.track_state == TrackHandler.TrackState.HYPOTHETICAL:
                        # If other is hypothetical, we do not merge but we swap the link
                        # This invalidate some mappings for both 'track' and 'other', but it is fine.
                        # If 2nd branch is first to be executed this way, the 1st branch will not be executed
                        track_to_det[det_to_track[track_to_det_merge[i]]] = -1
                        track_to_det[i] = track_to_det_merge[i]
                    else:  # Merge (as this is the second track, the track is just dropped)
                        track.merge_id = int(track_to_det_merge[i])  # This is not merge id yet
                        merges_to_ref.append(track)  # It will be updated once all tracks has been processed

                        track.track_state = TrackHandler.TrackState.FINISHED
                        self.inactive_tracks.append(track)
                        continue  # It does itw own specific update, the track cannot be updated any more.

            if track_to_det[i] != -1 and det_to_track_merge[track_to_det[i]] != -1:  # First branch of a merge
                other = self.active_tracks[det_to_track_merge[track_to_det[i]]]

                # If other is hypothetical, nothing to do, we do not merge with hypothetical tracks
                if other.track_state != TrackHandler.TrackState.HYPOTHETICAL:
                    if track.track_state == TrackHandler.TrackState.HYPOTHETICAL:
                        # If track is hypothetical, we do not merge but we swap the link
                        # If 1st branch is first to be executed this way, the 2nd branch will not be executed
                        track_to_det[det_to_track_merge[track_to_det[i]]] = track_to_det[i]
                        track_to_det[i] = -1
                    else:  # Merge: Create a new track and stop the former one
                        # Replace by a new VALID handler
                        other = TrackHandler(self.specs.n_valid, self.specs.n_gap, self.frame_id, self._next_identifier)
                        self._next_identifier += 1
                        other.track_state = TrackHandler.TrackState.VALID

                        track.merge_id = other.identifier
                        det_to_merge_id[int(track_to_det[i])] = other.identifier

                        # Terminate old track
                        track.track_state = TrackHandler.TrackState.FINISHED
                        self.inactive_tracks.append(track)

                        # Swap other with track (the merged one is finished)
                        track = other  # Will be updated and kept in active

            # Update track (classical link/no link, or a newly created one at a split/merge event)
            track.update(self.frame_id, int(track_to_det[i].item()))

            # Check if track is still active
            if track.is_active():
                still_active.append(track)
                active_mask[i] = True
            elif track.track_state == TrackHandler.TrackState.FINISHED or self.save_all:
                self.inactive_tracks.append(track)

            if track.track_state == TrackHandler.TrackState.INVALID and track.parent_id != -1:
                self._undo_split(track, still_active)

        # Relabel merges
        for track in merges_to_ref:
            track.merge_id = det_to_merge_id[track.merge_id]

        # Relabel links
        self._links[:, 0] = torch.arange(len(track_to_det))[track_to_det != -1]
        self._links[:, 1] = track_to_det[track_to_det != -1]

        self.active_tracks = still_active + new_tracks

        return active_mask

    def _update_active_tracks(self) -> torch.Tensor:
        """Calls `update` for active tracks and return a boolean mask that indicates which track is still active

        Tracks that are terminated are stored inside `inactive_tracks` and dropped from `active_tracks`.

        Returns:
            torch.Tensor: Boolean tensor indicating True for still active tracks

        """
        i_to_j = torch.full((len(self.active_tracks),), -1, dtype=torch.int32)
        i_to_j[self._links[:, 0]] = self._links[:, 1]
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

            if track.track_state == TrackHandler.TrackState.INVALID and track.parent_id != -1:
                self._undo_split(track, still_active)

        self.active_tracks = still_active

        return active_mask

    def _undo_split(self, track: TrackHandler, still_active: List[TrackHandler]):
        """Undo a split if one of the child track is invalidated.

        It finds the finished parent track handler and the valid other child handler and
        concatenates them back together in a single valid track handler.

        This is quite slow, but this should not occur often.
        """
        # This is slow, this should not occurs often. To be really faster it requires other data structures
        # To be changed if it is necessary

        # Find parent and sibling
        parent = [other for other in self.inactive_tracks if other.identifier == track.parent_id][0]
        child = [other for other in self.active_tracks + self.inactive_tracks if other.parent_id == track.parent_id][0]

        # Remove parent from inactive tracks
        self.inactive_tracks.remove(parent)

        if child.is_active():
            # By construction, the valid child is in still_active (because added before the hypothetical child)
            # So no need to check self.active_tracks
            # if child not in still_active:
            #     still_active = self.active_tracks  # In the loop, we haven't found
            child_index = still_active.index(child)
        else:
            child_index = self.inactive_tracks.index(child)

        concatenated_handler = TrackHandler(self.specs.n_valid, self.specs.n_gap, parent.start, child.identifier)
        concatenated_handler.track_state = child.track_state
        concatenated_handler.last_association = child.last_association
        concatenated_handler.parent_id = parent.parent_id
        concatenated_handler.merge_id = child.merge_id
        concatenated_handler.track_ids = parent.track_ids + child.track_ids
        concatenated_handler.detection_ids = parent.detection_ids + child.detection_ids
        concatenated_handler.is_split = child.is_split

        # Replace child by concatenated
        if concatenated_handler.is_active():
            still_active[child_index] = concatenated_handler
        else:
            self.inactive_tracks[child_index] = concatenated_handler

    def _handle_extra_detections(self, detections: byotrack.Detections):
        """Handle extra detections by creating new track handlers

        Args:
            detections (byotrack.Detections): Detections for the given frame

        """
        # Create a new active track for each unmatched measure
        for i in torch.arange(len(detections))[self._unmatched_detections].tolist():
            handler = TrackHandler(self.specs.n_valid, self.specs.n_gap, self.frame_id, self._next_identifier)
            self._next_identifier += 1
            handler.update(self.frame_id, i)
            self.active_tracks.append(handler)

    def update_detections(self, detections: byotrack.Detections) -> byotrack.Detections:
        """Optional modification of the currrent detections based on the current state

        This is called by `update` after the motion modeling but before the cost/association.

        By default, it does not change anything.

        Args:
            detections (byotrack.Detections): Detections at the current frame

        Returns:
            byotrack.Detections: The (optionally modified) detections to use at this current frame

        """
        return detections

    def associate(self, frame: np.ndarray, detections: byotrack.Detections) -> torch.Tensor:
        """Produces links between the current tracks and detections

        Optionnally it handles merges and splits by assocating a second time.

        Args:
            frame (np.ndarray): Current frame
            detections (byotrack.Detections): Current detections

        Returns:
            torch.Tensor: Links (i, j)
                Shape: (L, 2), dtype: int32
        """

        cost, threshold = self.cost(frame, detections)
        self._links = self.specs.association_method.solve(cost, threshold)

        self._unmatched_detections = torch.full((len(detections),), True)
        self._unmatched_detections[self._links[:, 1]] = False

        if self.specs.merge_factor == 0 and self.specs.split_factor == 0:
            return self._links  # No merge or splits

        unmatched_tracks = torch.full((len(self.active_tracks),), True)
        unmatched_tracks[self._links[:, 0]] = False
        valid_tracks = torch.tensor(
            [(track.track_state == TrackHandler.TrackState.VALID) for track in self.active_tracks], dtype=torch.bool
        )

        if self.specs.merge_factor > 0:
            # We simply do a 2nd association between unassociated VALID tracks with associated detections
            tracks_mask = unmatched_tracks & valid_tracks

            # TODO: Mass factor
            # self.active_mass[tracks_mask]

            self._merge_links = self.specs.association_method.solve(
                cost[tracks_mask][:, ~self._unmatched_detections], threshold * self.specs.merge_factor
            )

            # Relabel
            self._merge_links[:, 0] = torch.arange(len(self.active_tracks))[tracks_mask][self._merge_links[:, 0]]
            self._merge_links[:, 1] = torch.arange(len(detections))[~self._unmatched_detections][
                self._merge_links[:, 1]
            ]

        if self.specs.split_factor > 0:
            # We simply do a 2nd association between associated tracks with unassociated detections

            # Split mass factor
            # We increase the distance if the split is not evenly weighted and if it does not sum at the previous mass
            track_mass = self.active_mass[self._links[:, 0]]
            associated_mass = detections.mass[self._links[:, 1]]
            non_associated_mass = detections.mass[self._unmatched_detections]

            even_factor = torch.maximum(associated_mass[:, None], non_associated_mass[None, :]) / torch.minimum(
                associated_mass[:, None], non_associated_mass[None, :]
            )
            sum_ = associated_mass[:, None] + non_associated_mass[None, :]
            mass_factor = torch.maximum(track_mass[:, None], sum_) / torch.minimum(track_mass[:, None], sum_)

            self._split_links = self.specs.association_method.solve(
                cost[~unmatched_tracks][:, self._unmatched_detections] * even_factor * mass_factor,
                threshold * self.specs.split_factor,
            )

            # Relabel
            self._split_links[:, 0] = torch.arange(len(self.active_tracks))[~unmatched_tracks][self._split_links[:, 0]]
            self._split_links[:, 1] = torch.arange(len(detections))[self._unmatched_detections][self._split_links[:, 1]]

        return self._links

    def update(self, frame: np.ndarray, detections: byotrack.Detections) -> None:
        if self.frame_id == -1:
            # Let's reset again just in case with the right dim
            self.reset(detections.dim)

        self.frame_id += 1

        # Compute the flow map if optflow given
        if self.optflow is not None:
            self.optflow.update(frame)

        self.motion_model()
        detections = self.update_detections(detections)

        # Compute features if the extractor is given and register inside the detections
        # Do not recompute the features if some are already registered
        remove_feats = False
        if self.features_extractor is not None:
            if "features" in detections.data:
                warnings.warn("Some features are already computed. They will be used.")
            else:
                remove_feats = True
                self.features_extractor.register(frame, detections)

        self.associate(frame, detections)
        active_mask = self.update_active_tracks(detections)
        self.post_association(frame, detections, active_mask)

        # Handle mass with a fixed ema and concatenate with mass of newly created track
        self.active_mass[self._links[:, 0]] -= (1.0 - 0.8) * (
            self.active_mass[self._links[:, 0]] - detections.mass[self._links[:, 1]]
        )
        self.active_mass = torch.cat((self.active_mass[active_mask], detections.mass[self._unmatched_detections]))

        assert len(self.all_positions[-1]) == len(
            self.active_tracks
        ), "Wrong implementation of post_association. It should register the positions of each active tracks"
        for i, track in enumerate(self.active_tracks):
            track.register_track_id(i)

        # Remove the computed features if save_all is False
        if not self.save_all and remove_feats:
            detections.data.pop("features")
