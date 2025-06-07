import dataclasses
import enum
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch_kf
import torch_kf.ckf

import byotrack

from .nearest_neighbor import AssociationMethod, FrameByFrameLinker, FrameByFrameLinkerParameters


class Cost(enum.Enum):
    """The cost to use for association

    * LIKELIHOOD
        Build the cost matrix to maximize the likelihood of the association.
        The cost is defined as: the negative log likelihood that the detections j is from track i.
        C[i, j] = - log P(det_j | track_i)
        The cost limit is expected to be the smallest probability to accept.
    * MAHALANOBIS
        Mahalanobis distance, using the track uncertainty to correct the euclidean distance.
        The cost limit is expected to be the largest Mahalanobis distance to accept.
    * MAHALANOBIS_SQ
        Squared Mahalanobis distance.
        The cost limit is expected to be the largest Mahalanobis distance to accept. (not squared)
    * EUCLIDEAN
        Euclidean distance.
        The cost limit is expected to be the largest Euclidean distance to accept.
    * EUCLIDEAN_SQ
        Squared euclidean distance.
        The cost limit is expected to be the largest Euclidean distance to accept. (not squared)

    """

    LIKELIHOOD = "likelihood"
    MAHALANOBIS = "mahalanobis"
    MAHALANOBIS_SQ = "mahalanobis_sq"
    EUCLIDEAN = "euclidean"
    EUCLIDEAN_SQ = "euclidean_sq"


class TrackBuilding(enum.Enum):
    """How to build the final tracks

    * DETECTION
        Build tracks from detections without filtering nor filling gaps
    * FILTERED
        Build tracks from the Kalman filter outputs
    * SMOOTHED
        Add a backward run of the Kalman filter (RTS) to smooth tracks position

    """

    DETECTION = "detection"
    FILTERED = "filtered"
    SMOOTHED = "smoothed"


@dataclasses.dataclass
class KalmanLinkerParameters(FrameByFrameLinkerParameters):
    """Parameters of KalmanLinker

    Note:
        The merging and splitting features is still experimental.

    Attributes:
        association_threshold (float): This is the main hyperparameter, it defines the threshold on the distance used
            not to link tracks with detections. It prevents to link with false positive detections.
            Default: 5 pixels
        detection_std (Union[float, torch.Tensor]): Expected measurement noise on the detection process.
            The detection process is modeled with a Gaussian noise with this given std. (You can provide a different
            noise for each dimension). See `torch_kf.ckf.constant_kalman_filter`.
            Default: 3.0 pixels
        process_std (Union[float, torch.Tensor]): Expected process noise. See `torch_kf.ckf.constant_kalman_filter`, the
            process is modeled as constant order-th derivative motion. This quantify how much the supposely "constant"
            order-th derivative can change between two consecutive frames. A common rule of thumb is to use
            3 * process_std ~= max_t(| dx^(order)(t+1) - dx^(order)(t)|). It can be provided for each dimension).
            Default: 1.5 pixels
        kalman_order (int): Order of the Kalman filter to use.
            0 for brownian motions, 1 for directed brownian motion, 2 for accelerated brownian motions, etc...
            Default: 1
        n_valid (int): Number associated detections required to validate the track after its creation.
            Default: 3
        n_gap (int): Number of consecutive frames without association before the track termination.
            Default: 3
        association_method (AssociationMethod): The frame-by-frame association to use. See `AssociationMethod`.
            It can be provided as a string. (Choice: GREEDY, [SPARSE_]OPT_HARD, [SPARSE_]OPT_SMOOTH)
            Default: OPT_SMOOTH
        anisotropy (Tuple[float, float, float]): Anisotropy of images (Ratio of the pixel sizes
            for each axis, depth first). This will be used to scale distances. It will only impact
            EUCLIDEAN[_SQ] costs. For probabilistic cost, anisotropy should be already integrated
            in the stds of the kalman filter (providing one std for each dimension).
            Default: (1., 1., 1.)
        cost_method (CostMethod): The cost method to use. It can be provided as a string.
            See `CostMethod`. It also indicates what is the correct unit of `association_threshold`.
            Default: EUCLIDEAN
        track_building (TrackBuilding): Tells the linker how to build the final tracks.
            Either from detections, or from filtered/smoothed positions computed by the
            Kalman filter. See `TrackBuilding`. It can be provided as a string.
            Default: FILTERED
        split_factor (float): Allow splitting of tracks, using a second association step.
            The association threshold in this case is `split_factor * association_threshold`.
            Default: 0.0 (No splits)
        merge_factor (float): Allow merging of tracks, using a second association step.
            The association threshold in this case is `merge_factor * association_threshold`.
            Default: 0.0 (No merges)
        online_process_std (float): Recomputes the process std online following "A. Genovesio, et al, 2004, October.
            Adaptive gating in Gaussian Bayesian multi-target tracking. ICIP'04. (Vol. 1, pp. 147-150). IEEE."
            Each track has its own process std depending on the errors made in the past. It automatically adjusts to
            process errors, allowing to increase the validation gate. Should be used in conjonction with MAHALANOBIS
            or LIKELIHOOD `cost_method`. As this may be detrimental, it is disabed by default.
            Default: 0.0 (Process_std is constant)
        initial_std_factor (float): The uncertainties on initial velocities/accelerations are set
            to initial_std_factor * process_std. Having a small factor will prevent handling correctly
            starting tracks that already moves on their first frames. But large values will lead to large uncertainty
            on the first prediction, making it hard to associate to a detection with MAHALANOBIS
            or LIKELIHOOD methods. Typical values lies in 3.0 to 10.0.
            Default: 10.0

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        association_threshold: float = 5.0,
        *,
        detection_std: Union[float, torch.Tensor] = 3.0,
        process_std: Union[float, torch.Tensor] = 1.5,
        kalman_order: int = 1,
        n_valid=3,
        n_gap=3,
        association_method: Union[str, AssociationMethod] = AssociationMethod.OPT_SMOOTH,
        anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        cost: Union[str, Cost] = Cost.EUCLIDEAN,
        track_building: Union[str, TrackBuilding] = TrackBuilding.FILTERED,
        split_factor: float = 0.0,
        merge_factor: float = 0.0,
        online_process_std: float = 0.0,
        initial_std_factor: float = 10.0,
    ):
        super().__init__(
            association_threshold=association_threshold,
            n_valid=n_valid,
            n_gap=n_gap,
            anisotropy=anisotropy,
            association_method=association_method,
            split_factor=split_factor,
            merge_factor=merge_factor,
        )

        if isinstance(detection_std, float) and min(anisotropy) != max(anisotropy):
            warnings.warn(
                "A single `detection_std` is provided, but images are anisotrope. Consider giving one std by dimension."
            )
        if isinstance(process_std, float) and min(anisotropy) != max(anisotropy):
            warnings.warn(
                "A single `process_std` is provided, but images are anisotrope. Consider giving one std by dimension."
            )

        self.detection_std = detection_std
        self.process_std = process_std
        self.kalman_order = kalman_order
        self.initial_std_factor = initial_std_factor

        self.cost = cost if isinstance(cost, Cost) else Cost[cost.upper()]
        self.track_building = (
            track_building if isinstance(track_building, TrackBuilding) else TrackBuilding[track_building.upper()]
        )
        self.online_process_std = online_process_std

    detection_std: Union[float, torch.Tensor] = 3.0
    process_std: Union[float, torch.Tensor] = 1.5
    kalman_order: int = 1
    cost: Cost = Cost.EUCLIDEAN
    track_building: TrackBuilding = TrackBuilding.FILTERED
    online_process_std: float = 0.0
    initial_std_factor: float = 10.0


class KalmanLinker(FrameByFrameLinker):
    """Frame by frame linker using Kalman filters

    Motion is modeled with a Kalman filter of a specified order (See `torch_kf.ckf`)
    Matching is done to optimize the given cost. If optical flow is provided, it is used
    online to warp the predicted state positions of the kalman filter. This will work, but it
    is sub-optimal: consider using `KOFTLinker` that exploits in a finer way optical flow
    inside Kalman filters.

    This is an implementation of Simple Kalman Tracking (SKT) from KOFT [9].

    Note:
        This implementation requires torch-kf. (pip install torch-kf)

    See `FrameByFrameLinker` for the other attributes.

    Attributes:
        specs (KalmanLinkerParameters): Parameters specifications of the algorithm.
            See `KalmanLinkerParameters`.
        kalman_filter (Optional[torch_kf.KalmanFilter]): The Kalman filter. (Build once the tracking starts
            allowing to adapt the dimension on the fly)
        active_states (Optional[torch_kf.GaussianState]): The Kalman filter estimation for each track.
            Shape: mean=(N, D * (order + 1), 1), covariance=(N, D * (order + 1), dim * (order + 1))
            dtype: float
        projections (Optional[torch_kf.GaussianState]): The Kalman filter projection for each track.
            Shape: mean=(N, D, 1), covariance=(N, D, D), precision=(N, D, D)
            dtype: float
        process_noises (Optional[torch.Tensor]): The Kalman filter process noise for each track.
            Only used when online_process_std > 0.0. It allows to compute an adaptative process_std
            and therefore gating for each track.
            Shape: (N, D, 1), dtype: float
        all_states (List[torch_kf.GaussianState]): The Kalman filter estimation for each track at each seen
            frame. States are only registered when save_all=True or if you build tracks from RTS smoothing.
            Shape: mean=(N, D * (order + 1), 1), covariance=(N, D * (order + 1), dim * (order + 1))
            dtype: float

    """

    progress_bar_description = "Kalman filter linking"

    def __init__(
        self,
        specs: KalmanLinkerParameters,
        optflow: Optional[byotrack.OpticalFlow] = None,
        features_extractor: Optional[byotrack.FeaturesExtractor] = None,
        save_all=False,
    ) -> None:
        super().__init__(specs, optflow, features_extractor, save_all)

        self.specs: KalmanLinkerParameters
        self.kalman_filter = torch_kf.ckf.constant_kalman_filter(
            self.specs.detection_std,
            self.specs.process_std,
            dim=2,
            order=self.specs.kalman_order,
        )
        self.dtype = torch.float32

        self.active_states = torch_kf.GaussianState(
            torch.empty((0, self.kalman_filter.state_dim, 1)),
            torch.empty((0, self.kalman_filter.state_dim, self.kalman_filter.state_dim)),
        )
        self.projections = self.kalman_filter.project(self.active_states)
        self.process_noises = torch.empty((0, self.kalman_filter.state_dim, self.kalman_filter.state_dim))

        self.all_states: List[torch_kf.GaussianState] = []

    def reset(self, dim=2) -> None:
        super().reset(dim)

        self.kalman_filter = torch_kf.ckf.constant_kalman_filter(
            self.specs.detection_std,
            self.specs.process_std,
            dim=dim,
            order=self.specs.kalman_order,
            approximate=True,
        )
        self.kalman_filter = self.kalman_filter.to(self.dtype)
        self.active_states = torch_kf.GaussianState(
            torch.empty((0, self.kalman_filter.state_dim, 1), dtype=self.dtype),
            torch.empty((0, self.kalman_filter.state_dim, self.kalman_filter.state_dim), dtype=self.dtype),
        )
        self.projections = self.kalman_filter.project(self.active_states)
        self.process_noises = torch.empty(
            (0, self.kalman_filter.state_dim, self.kalman_filter.state_dim), dtype=self.dtype
        )
        self.all_states = []

    def collect(self) -> List[byotrack.Track]:
        if self.specs.track_building == TrackBuilding.SMOOTHED:
            dim = self.kalman_filter.state_dim
            tracks_handlers = [
                handler
                for handler in self.inactive_tracks + self.active_tracks
                if handler.track_state in (handler.TrackState.VALID, handler.TrackState.FINISHED)
            ]

            states = torch_kf.GaussianState(
                torch.full((len(self.all_states), len(tracks_handlers), dim, 1), torch.nan, dtype=self.dtype),
                torch.zeros((len(self.all_states), len(tracks_handlers), dim, dim), dtype=self.dtype),
            )
            is_defined = torch.full((len(self.all_states), len(tracks_handlers)), False)

            for t, states_ in enumerate(self.all_states):
                for i, handler in enumerate(tracks_handlers):
                    time_offset = t - handler.start
                    if 0 <= time_offset < len(handler):
                        states[t, i] = states_[handler.track_ids[time_offset]]
                        is_defined[t, i] = True

            # Iterate backward to update all states (Update done for active t where t+1 is defined)
            for t in range(len(self.all_states) - 2, -1, -1):
                mask = is_defined[t + 1] & is_defined[t]
                cov_at_process = states.covariance[t, mask] @ self.kalman_filter.process_matrix.mT
                predicted_covariance = (
                    self.kalman_filter.process_matrix @ cov_at_process + self.kalman_filter.process_noise
                )

                kalman_gain = cov_at_process @ predicted_covariance.inverse().mT
                states.mean[t, mask] += kalman_gain @ (
                    states.mean[t + 1, mask] - self.kalman_filter.process_matrix @ states.mean[t, mask]
                )
                states.covariance[t, mask] += (
                    kalman_gain @ (states.covariance[t + 1, mask] - predicted_covariance) @ kalman_gain.mT
                )

            dim = self.all_positions[0].shape[-1]  # For KOFT, using kf.measure_dim would not work
            tracks = []
            for i, handler in enumerate(tracks_handlers):
                tracks.append(
                    byotrack.Track(
                        handler.start,
                        states.mean[handler.start : handler.start + len(handler), i, :dim, 0].to(torch.float32),
                        handler.identifier,
                        torch.tensor(handler.detection_ids[: len(handler)], dtype=torch.int32),
                        merge_id=handler.merge_id,
                        parent_id=handler.parent_id,
                    )
                )

            return tracks

        return super().collect()

    def motion_model(self) -> None:
        # Use the Kalman filter to predict the current states of each active tracks
        self.active_states = self.kalman_filter.predict(
            self.active_states, process_noise=self.process_noises if self.specs.online_process_std else None
        )

        # Add optical flow motion to the position
        if self.optflow and self.optflow.flow_map is not None:
            positions = self.active_states.mean[:, : self.kalman_filter.measure_dim, 0]
            positions[:] = torch.tensor(self.optflow.optflow.transform(self.optflow.flow_map, positions.numpy()))

        # Project states for association
        self.projections = self.kalman_filter.project(self.active_states)

    def cost(self, _: np.ndarray, detections: byotrack.Detections) -> Tuple[torch.Tensor, float]:
        anisotropy = torch.tensor(self.specs.anisotropy)[-detections.dim :]

        if self.specs.cost == Cost.EUCLIDEAN:
            return (
                torch.cdist(self.projections.mean[..., 0] * anisotropy, detections.position * anisotropy),
                self.specs.association_threshold,
            )

        if self.specs.cost == Cost.EUCLIDEAN_SQ:
            return (
                torch.cdist(self.projections.mean[..., 0] * anisotropy, detections.position * anisotropy).pow_(2),
                self.specs.association_threshold**2,
            )
        if self.specs.cost == Cost.MAHALANOBIS:
            return (
                self.projections[:, None].mahalanobis(detections.position[None, ..., None]),
                self.specs.association_threshold,
            )
        if self.specs.cost == Cost.MAHALANOBIS_SQ:
            return (
                self.projections[:, None].mahalanobis_squared(detections.position[None, ..., None]),
                self.specs.association_threshold**2,
            )

        # LIKELIHOOD: cost = -log likelihood
        cost = -self.projections[:, None].log_likelihood(detections.position[None, ..., None])
        return cost, -torch.log(torch.tensor(self.specs.association_threshold)).item()

    def post_association(self, _: np.ndarray, detections: byotrack.Detections, active_mask: torch.Tensor):
        positions = detections.position.to(self.dtype)

        # Update the state of associated tracks (unassociated tracks keep the predicted state)
        if self.specs.online_process_std:
            updated = self.kalman_filter.update(
                self.active_states[self._links[:, 0]],
                positions[self._links[:, 1]][..., None],
                projection=self.projections[self._links[:, 0]],
            )

            # Update the process_noise based on the prediction errors
            # Err = x_t - Fx_{t-1} (low estimation of errors)
            # Then Q_t = a * (b * Q_t + 1 - b err @ err.T) + (1- a) Q_0
            # In this implem, we fix b = 0.75 (around 4 frames to estimate errors) and a is user-defined.
            errors = self.active_states.mean[self._links[:, 0]] - updated.mean
            self.process_noises[self._links[:, 0]] *= self.specs.online_process_std * 0.75  # Fixed EMA ~4 frames lag
            self.process_noises[self._links[:, 0]] += self.specs.online_process_std * 0.25 * errors @ errors.mT
            self.process_noises[self._links[:, 0]] += (
                1 - self.specs.online_process_std
            ) * self.kalman_filter.process_noise

            self.active_states[self._links[:, 0]] = updated
        else:
            self.active_states[self._links[:, 0]] = self.kalman_filter.update(
                self.active_states[self._links[:, 0]],
                positions[self._links[:, 1]][..., None],
                projection=self.projections[self._links[:, 0]],
            )

        # Create new states for unmatched measures
        unmatched_measures = positions[self._unmatched_detections]

        # Build the initial states for tracks:
        # Infinite uncertainty on the prior position => Given positional measurement, we initialize on its position
        #                                               with the positional measurement uncertainty.
        # For derivatives, we assume they are centered on 0 (no bias toward a direction) with an uncertainty of
        # initial_std * process_std:
        # Small factors (~1) are not suited for tracking objects that appears with a large initial velocity.
        # Large factors (>> 1) will create association problem on the second frame, as the predicted
        # position inherit from this uncertainty.
        initial_state = torch_kf.GaussianState(
            torch.zeros(len(unmatched_measures), self.kalman_filter.state_dim, 1, dtype=self.dtype),
            (
                torch.eye(self.kalman_filter.state_dim, dtype=self.dtype)
                * (self.specs.initial_std_factor * self.specs.process_std) ** 2
            )[None]
            .expand(len(unmatched_measures), self.kalman_filter.state_dim, self.kalman_filter.state_dim)
            .clone(),
        )
        initial_state.mean[:, : detections.dim, 0] = unmatched_measures
        initial_state.covariance[:, : detections.dim, : detections.dim] = self.kalman_filter.measurement_noise[
            : detections.dim, : detections.dim
        ]

        # Merge still active states with the initial ones of the created tracks
        self.active_states = torch_kf.GaussianState(
            torch.cat((self.active_states.mean[active_mask], initial_state.mean)),
            torch.cat((self.active_states.covariance[active_mask], initial_state.covariance)),
        )
        if self.specs.online_process_std:
            self.process_noises = torch.cat(
                (
                    self.process_noises[active_mask],
                    self.kalman_filter.process_noise[None].expand(
                        len(unmatched_measures), self.kalman_filter.state_dim, self.kalman_filter.state_dim
                    ),
                )
            )

        if self.specs.track_building == TrackBuilding.DETECTION:
            self.all_positions.append(
                torch.cat(
                    [
                        (
                            detections.position[track.detection_ids[-1]][None]
                            if track.detection_ids[-1] != -1
                            else torch.full((1, detections.position.shape[1]), torch.nan)
                        )
                        for track in self.active_tracks
                    ]
                )
            )
        else:
            self.all_positions.append(self.active_states.mean[:, : detections.dim, 0].to(torch.float32))

        if self.save_all or self.specs.track_building == TrackBuilding.SMOOTHED:
            self.all_states.append(self.active_states)
