# pylint: disable=duplicate-code

import dataclasses
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch_kf
import torch_kf.ckf

import byotrack

from .base import AssociationMethod, OnlineFlowExtractor
from .kalman_linker import Cost, KalmanLinker, KalmanLinkerParameters, TrackBuilding


@dataclasses.dataclass
class KOFTLinkerParameters(KalmanLinkerParameters):
    """Parameters of KOFTLinker

    Attributes:
        association_threshold (float): This is the main hyperparameter, it defines the threshold on the distance used
            not to link tracks with detections. It prevents to link with false positive detections.
        detection_std (Union[float, torch.Tensor]): Expected measurement noise on the detection process.
            The detection process is modeled with a Gaussian noise with this given std. (You can provide a different
            noise for each dimension). See `torch_kf.ckf.constant_kalman_filter`.
            Default: 3.0 pixels
        flow_std (Union[float, torch.Tensor]): Expected measurement noise on the optical flow process.
            The optical flow process is modeled with a Gaussian noise with this given std. (You can provide a different
            noise for each dimension).
            Default: 1.0 pixels
        process_std (Union[float, torch.Tensor]): Expect process noise. See `torch_kf.ckf.constant_kalman_filter`, the
            process is modeled as constant order-th derivative motion. This quantify how much the supposely "constant"
            order-th derivative can change between two consecutive frames. A common rule of thumb is to use
            3 * process_std ~= max_t(| x^(order)(t) - x^(order)(t+1)|). It can be provided for each dimension).
            Default: 1.5 pixels / frame^order
        kalman_order (int): Order of the Kalman filter to use. 0 is not supported.
            1 for directed brownian motion, 2 for accelerated brownian motions, etc...
            Default: 1
        n_valid (int): Number of frames with a correct association required to validate the track at its creation.
            Default: 3
        n_gap (int): Number of frames with no association before the track termination.
            Default: 3
        association_method (AssociationMethod): The frame-by-frame association to use. See `AssociationMethod`.
            It can be provided as a string. (Choice: GREEDY, OPT_HARD, OPT_SMOOTH)
            Default: OPT_SMOOTH
        cost_method (CostMethod): The cost method to use. It can be provided as a string.
            See `CostMethod`. It also indicates what is the correct unit of `association_threshold`.
            Default: EUCLIDEAN
        track_building (TrackBuilding): Tells the linker how to build the final tracks.
            Either from detections, or from filtered/smoothed positions computed by the
            Kalman filter. See `TrackBuilding`. It can be provided as a string.
            Default: FILTERED
        extract_flows_on_detections (bool): If True it extracts the optical flow at the detection location if possible.
            Otherwise it extract the flow from the curent estimate of the track position.
            Default: False
        always_measure_velocity (bool): Update velocity for all tracks even non-linked ones.
            If set to False, it implements KOFT-- from the paper. This is sub-optimal, you should keep it True.
            Default: True

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        association_threshold: float,
        *,
        detection_std: Union[float, torch.Tensor] = 3.0,
        flow_std: Union[float, torch.Tensor] = 1.0,
        process_std: Union[float, torch.Tensor] = 1.5,
        kalman_order: int = 1,
        n_valid=3,
        n_gap=3,
        association_method: Union[str, AssociationMethod] = AssociationMethod.OPT_SMOOTH,
        cost: Union[str, Cost] = Cost.EUCLIDEAN,
        track_building: Union[str, TrackBuilding] = TrackBuilding.FILTERED,
        extract_flows_on_detections=False,
        always_measure_velocity=True,
    ):
        super().__init__(
            association_threshold=association_threshold,
            detection_std=detection_std,
            process_std=process_std,
            kalman_order=kalman_order,
            n_valid=n_valid,
            n_gap=n_gap,
            association_method=association_method,
            cost=cost,
            track_building=track_building,
        )

        self.flow_std = flow_std
        self.extract_flows_on_detections = extract_flows_on_detections
        self.always_measure_velocity = always_measure_velocity

        assert self.kalman_order >= 1, "With KOFT, the velocity is measured and thus should be modeled."

    flow_std: Union[float, torch.Tensor] = 1.0
    extract_flows_on_detections: bool = False
    always_measure_velocity: bool = True


class KOFTLinker(KalmanLinker):
    """Kalman and Optical Flow Tracking [9]

    Motion is modeled with a Kalman filter of a specified order >= 1 (See `torch_kf.ckf`)
    Positions are measured through the detection process. A second update step is performed
    to measure the velocity of all tracks using optical flow.

    Matching is done to optimize the given cost.

    Note:
        This implementation requires torch-kf. (pip install torch-kf)

    See `KalmanLinker` for the other attributes.

    Attributes:
        specs (KOFTLinkerParameters): Parameters specifications of the algorithm.
            See `KOFTLinkerParameters`.
        last_detections (byotrack.Detections): The last detections used in update.
            Optionnaly used to extract flows at the detection positions and not the track state.
            Required for `motion_model`
        n_initial (int): Number of newly started tracks on the last update. Used to correclty
            initialized these tracks with the velocity update.

    """

    progress_bar_description = "KOFT linking"

    def __init__(
        self,
        specs: KOFTLinkerParameters,
        optflow: Optional[byotrack.OpticalFlow] = None,
        features_extractor: Optional[byotrack.FeaturesExtractor] = None,
        save_all=False,
    ) -> None:
        super().__init__(specs, optflow, features_extractor, save_all)

        self.specs: KOFTLinkerParameters
        assert self.optflow is not None, "KOFT requires an optical flow algorithm"
        self.optflow: OnlineFlowExtractor

        self.last_detections = byotrack.Detections(data={"position": torch.empty((0, 2))})
        self.n_initial = 0

    def reset(self) -> None:
        super().reset()
        self.last_detections = byotrack.Detections(data={"position": torch.empty((0, 2))})
        self.n_initial = 0

    def motion_model(self) -> None:
        # Not initialized yet
        if self.active_states is None or self.kalman_filter is None:
            return

        if self.optflow.flow_map is None:
            raise RuntimeError("The motion model cannot be applied without a computed flow")

        dim = self.kalman_filter.measure_dim // 2

        # Second update using velocity just before the prediction time
        points = self.active_states.mean[:, :dim, 0]
        if self.specs.extract_flows_on_detections:
            for i, track in enumerate(self.active_tracks):
                if track.detection_ids[-1] != -1:
                    points[i] = self.last_detections.position[track.detection_ids[-1]]

        if not self.specs.always_measure_velocity:  # KOFT--
            mask = torch.full((len(self.active_tracks),), True)
            for i, track in enumerate(self.active_tracks):
                if track.detection_ids[-1] == -1:
                    mask[i] = False

            points = points[mask]

        velocities = torch.tensor(self.optflow.optflow.flow_at(self.optflow.flow_map, points.numpy()))[..., None]

        if not self.specs.always_measure_velocity:  # KOFT--
            self.active_states[mask] = self.kalman_filter.update(
                self.active_states[mask],
                velocities,
                measurement_matrix=self.kalman_filter.measurement_matrix[dim:],
                measurement_noise=self.kalman_filter.measurement_noise[dim:, dim:],
            )
        else:  # KOFT
            self.active_states = self.kalman_filter.update(
                self.active_states,
                velocities,
                measurement_matrix=self.kalman_filter.measurement_matrix[dim:],
                measurement_noise=self.kalman_filter.measurement_noise[dim:, dim:],
            )

        # Correct initial tracks: The velocity of initial tracks is initialize with computed flow
        # and the flow_std uncertainty
        if self.n_initial:
            self.active_states.mean[-self.n_initial :, dim : dim * 2] = velocities[-self.n_initial :]
            self.active_states.covariance[-self.n_initial :, dim : dim * 2, dim : dim * 2] = (
                self.kalman_filter.measurement_noise[dim:, dim:]
            )
        # Use the Kalman filter to predict the current states of each active tracks
        self.active_states = self.kalman_filter.predict(self.active_states)

        # Project states for association
        self.projections = self.kalman_filter.project(self.active_states)

    def cost(self, _: np.ndarray, detections: byotrack.Detections) -> Tuple[torch.Tensor, float]:
        if self.active_states is None or self.kalman_filter is None:
            self.kalman_filter = torch_kf.ckf.constant_kalman_filter(
                self.specs.detection_std,
                self.specs.process_std,
                detections.dim,
                self.specs.kalman_order,
                approximate=True,  # Approximate so that a flow precisely means the velocity modeled here
            )
            # Doubles the measurement space to measure velocity
            self.kalman_filter.measurement_matrix = torch.eye(detections.dim * 2, self.kalman_filter.state_dim)
            self.kalman_filter.measurement_noise = torch.eye(detections.dim * 2)
            self.kalman_filter.measurement_noise[: detections.dim, : detections.dim] *= self.specs.detection_std**2
            self.kalman_filter.measurement_noise[detections.dim :, detections.dim :] *= self.specs.flow_std**2

            self.active_states = torch_kf.GaussianState(
                torch.empty((0, self.kalman_filter.state_dim, 1)),
                torch.empty((0, self.kalman_filter.state_dim, self.kalman_filter.state_dim)),
            )
            self.projections = self.kalman_filter.project(self.active_states)

        if self.projections is None:
            raise RuntimeError("Projections should already be initialized.")

        if self.specs.cost == Cost.EUCLIDEAN:
            return (
                torch.cdist(self.projections.mean[:, : detections.dim, 0], detections.position),
                self.specs.association_threshold,
            )

        if self.specs.cost == Cost.EUCLIDEAN_SQ:
            return (
                torch.cdist(self.projections.mean[:, : detections.dim, 0], detections.position).pow_(2),
                self.specs.association_threshold**2,
            )

        # Restrict projections onto positions
        # NOTE: Following KOFT official implem, we use (cov-1)[:2, :2] instead of (cov[:2, :2])-1 for the precision.
        projections = torch_kf.GaussianState(
            self.projections.mean[:, : detections.dim],
            self.projections.covariance[:, : detections.dim, : detections.dim],
            (
                self.projections.precision[:, : detections.dim, : detections.dim]
                if self.projections.precision is not None
                else None
            ),
        )

        if self.specs.cost == Cost.MAHALANOBIS:
            return (
                projections[:, None].mahalanobis(detections.position[None, ..., None]),
                self.specs.association_threshold,
            )
        if self.specs.cost == Cost.MAHALANOBIS_SQ:
            return (
                projections[:, None].mahalanobis_squared(detections.position[None, ..., None]),
                self.specs.association_threshold**2,
            )

        # LIKELIHOOD: cost = -log likelihood
        cost = -projections[:, None].log_likelihood(detections.position[None, ..., None])
        return cost, -torch.log(torch.tensor(self.specs.association_threshold)).item()

    def post_association(self, _: np.ndarray, detections: byotrack.Detections, links: torch.Tensor):
        if self.active_states is None or self.kalman_filter is None or self.projections is None:
            raise RuntimeError("The linker should already be initialized.")

        self.last_detections = detections  # Save detections (May be required)

        # Update the state of associated tracks (unassociated tracks keep the predicted state)
        self.active_states[links[:, 0]] = self.kalman_filter.update(
            self.active_states[links[:, 0]],
            detections.position[links[:, 1]][..., None],
            torch_kf.GaussianState(
                self.projections.mean[links[:, 0], : detections.dim],
                self.projections.covariance[links[:, 0], : detections.dim, : detections.dim],
                None,  # /!\ inv(cov[:2,:2]) != inv(cov)[:2, :2] =>
            ),
            self.kalman_filter.measurement_matrix[: detections.dim],
            self.kalman_filter.measurement_noise[: detections.dim, : detections.dim],
        )

        # Update active track handlers
        active_mask = self.update_active_tracks(links)

        # Create new track handlers for unmatched detections
        unmatched = self.handle_extra_detections(detections, links)
        unmatched_measures = detections.position[unmatched]
        self.n_initial = unmatched_measures.shape[0]

        # Build the initial states for tracks:
        # We initialize the position using the detection position and the measurement std as covariance.
        # For velocity, we also will have a measurement at prediction time, it will be therefore initialized
        # correctly at this moment. If acceleration or jerk is modeled, we set them at 0 with a large covariance.
        # In SKT, having a too large uncertainty on non measured states will produce very uncertain projections
        # which are either unlinkable with likelihood or too linkable with mahalanobis.
        # In KOFT as velocity is measured before the prediction/projection step, the projection is never fully
        # uncertain, therefore we can initialize them with as much uncertainty as required.
        initial_state = torch_kf.GaussianState(
            torch.zeros(len(unmatched_measures), self.kalman_filter.state_dim, 1),
            torch.eye(self.kalman_filter.state_dim)[None].expand(
                len(unmatched_measures), self.kalman_filter.state_dim, self.kalman_filter.state_dim
            )
            * self.kalman_filter.process_noise.max()
            * 100,
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
            self.all_positions.append(self.active_states.mean[:, : detections.dim, 0])

        if self.save_all or self.specs.track_building == TrackBuilding.SMOOTHED:
            self.all_states.append(self.active_states)
