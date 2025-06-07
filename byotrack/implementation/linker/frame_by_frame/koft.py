# pylint: disable=duplicate-code

import dataclasses
from typing import Optional, Tuple, Union
import warnings

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

    Note:
        The merging and splitting features is still experimental.

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
        process_std (Union[float, torch.Tensor]): Expected process noise. See `torch_kf.ckf.constant_kalman_filter`, the
            process is modeled as constant order-th derivative motion. This quantify how much the supposely "constant"
            order-th derivative can change between two consecutive frames. A common rule of thumb is to use
            3 * process_std ~= max_t(| dx^(order)(t+1) - dx^(order)(t)|). It can be provided for each dimension).
            Default: 1.5 pixels
        kalman_order (int): Order of the Kalman filter to use. 0 is for brownian motion (it predicts a 0 velocity)
            1 for directed brownian motion, 2 for accelerated brownian motions, etc...
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
        extract_flows_on_detections (bool): If True it extracts the optical flow at the detection location if possible.
            Otherwise it extract the flow from the curent estimate of the track position.
            Default: False
        always_measure_velocity (bool): Update velocity for all tracks even non-linked ones.
            If set to False, it implements KOFT-- from the paper. This is sub-optimal, you should keep it True.
            Default: True
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

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
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
        anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        cost: Union[str, Cost] = Cost.EUCLIDEAN,
        track_building: Union[str, TrackBuilding] = TrackBuilding.FILTERED,
        split_factor: float = 0.0,
        merge_factor: float = 0.0,
        extract_flows_on_detections=False,
        always_measure_velocity=True,
        online_process_std=0.0,
        initial_std_factor=10.0,
    ):
        super().__init__(
            association_threshold=association_threshold,
            detection_std=detection_std,
            process_std=process_std,
            kalman_order=kalman_order,
            n_valid=n_valid,
            n_gap=n_gap,
            association_method=association_method,
            anisotropy=anisotropy,
            cost=cost,
            track_building=track_building,
            split_factor=split_factor,
            merge_factor=merge_factor,
            online_process_std=online_process_std,
            initial_std_factor=initial_std_factor,
        )

        if isinstance(flow_std, float) and min(anisotropy) != max(anisotropy):
            warnings.warn(
                "A single flow_std is provided, but the images are anisotrope. Consider giving one std by dimension."
            )

        self.flow_std = flow_std
        self.extract_flows_on_detections = extract_flows_on_detections
        self.always_measure_velocity = always_measure_velocity

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

        self.optflow: OnlineFlowExtractor
        self.specs: KOFTLinkerParameters
        assert self.optflow is not None, "KOFT requires an optical flow algorithm"

        self.last_detections = byotrack.Detections(data={"position": torch.empty((0, 2))})

    def reset(self, dim=2) -> None:
        super().reset(dim)

        # Reset the KF initialized by super
        self.kalman_filter = torch_kf.ckf.constant_kalman_filter(
            self.specs.detection_std,
            self.specs.process_std,
            dim=dim,
            order=self.specs.kalman_order + (self.specs.kalman_order == 0),
            approximate=True,  # Approximate so that a flow precisely means the velocity modeled here
        )
        if self.specs.kalman_order == 0:
            # In order 0, we still model velocity, but we always predict it at 0
            self.kalman_filter.process_matrix[dim:, dim:] = 0

        # Doubles the measurement space to measure velocity
        self.kalman_filter.measurement_matrix = torch.eye(dim * 2, self.kalman_filter.state_dim)
        self.kalman_filter.measurement_noise = torch.eye(dim * 2)
        self.kalman_filter.measurement_noise[:dim, :dim] *= self.specs.detection_std**2
        self.kalman_filter.measurement_noise[dim:, dim:] *= self.specs.flow_std**2
        self.kalman_filter = self.kalman_filter.to(self.dtype)

        self.active_states = torch_kf.GaussianState(
            torch.empty((0, self.kalman_filter.state_dim, 1), dtype=self.dtype),
            torch.empty((0, self.kalman_filter.state_dim, self.kalman_filter.state_dim), dtype=self.dtype),
        )
        self.projections = self.kalman_filter.project(self.active_states)

        self.last_detections = byotrack.Detections(data={"position": torch.empty((0, dim))})

    def motion_model(self) -> None:
        if self.optflow.flow_map is None:
            raise RuntimeError("The motion model cannot be applied without a computed flow")

        dim = self.kalman_filter.measure_dim // 2

        # Second update using velocity just before the prediction time
        points = self.active_states.mean[:, :dim, 0].clone()
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

        # Replace the registered states after adding optical flow inside of it
        if self.save_all or self.specs.track_building == TrackBuilding.SMOOTHED:
            if self.all_states:
                self.all_states[-1] = self.active_states

        # Use the Kalman filter to predict the current states of each active tracks
        self.active_states = self.kalman_filter.predict(
            self.active_states, process_noise=self.process_noises if self.specs.online_process_std else None
        )

        # Project states for association
        self.projections = self.kalman_filter.project(self.active_states)

    def cost(self, _: np.ndarray, detections: byotrack.Detections) -> Tuple[torch.Tensor, float]:
        anisotropy = torch.tensor(self.specs.anisotropy)[-detections.dim :]

        if self.specs.cost == Cost.EUCLIDEAN:
            return (
                torch.cdist(
                    self.projections.mean[:, : detections.dim, 0] * anisotropy, detections.position * anisotropy
                ),
                self.specs.association_threshold,
            )

        if self.specs.cost == Cost.EUCLIDEAN_SQ:
            return (
                torch.cdist(
                    self.projections.mean[:, : detections.dim, 0] * anisotropy, detections.position * anisotropy
                ).pow_(2),
                self.specs.association_threshold**2,
            )

        # Restrict projections onto positions
        # NOTE: Following KOFT initial implem, we use (cov-1)[:2, :2] instead of (cov[:2, :2])-1 for the precision.
        #       It works slightly better, but it needs to be investigated.
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

    def post_association(self, _: np.ndarray, detections: byotrack.Detections, active_mask: torch.Tensor):
        self.last_detections = detections  # Save detections (May be required)
        positions = detections.position.to(self.dtype)

        # Update the state of associated tracks (unassociated tracks keep the predicted state)
        updated = self.kalman_filter.update(
            self.active_states[self._links[:, 0]],
            positions[self._links[:, 1]][..., None],
            projection=torch_kf.GaussianState(
                self.projections.mean[self._links[:, 0], : detections.dim],
                self.projections.covariance[self._links[:, 0], : detections.dim, : detections.dim],
                None,  # /!\ inv(cov[:2,:2]) != inv(cov)[:2, :2]
            ),
            measurement_matrix=self.kalman_filter.measurement_matrix[: detections.dim],
            measurement_noise=self.kalman_filter.measurement_noise[: detections.dim, : detections.dim],
        )

        if self.specs.online_process_std:
            # XXX: Should be computed after velocity update ?
            # Update the process_noise based on the prediction errors.
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
        # As in KOFT, the velocities are measured, it could be set to a higher value.
        # For the Brownian motion, the process_std directly gives the initial uncertainty on velocity
        std_factor = self.specs.initial_std_factor if self.specs.kalman_order != 0 else 1.0
        initial_state = torch_kf.GaussianState(
            torch.zeros(len(unmatched_measures), self.kalman_filter.state_dim, 1, dtype=self.dtype),
            (torch.eye(self.kalman_filter.state_dim, dtype=self.dtype) * (std_factor * self.specs.process_std) ** 2)[
                None
            ]
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
