from __future__ import annotations

import dataclasses
import sys
import warnings
from typing import TYPE_CHECKING

import scipy
import torch
import torch_kf
import torch_kf.ckf
import tqdm.auto as tqdm

import byotrack
from byotrack.api.detections import statistics
from byotrack.implementation.linker.frame_by_frame.base import AssociationMethod
from byotrack.implementation.linker.frame_by_frame.kalman_linker import (
    Cost,
    KalmanLinker,
    KalmanLinkerParameters,
    TrackBuilding,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import numpy as np

    from byotrack.implementation.linker.frame_by_frame.base import OnlineFlowExtractor

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


@dataclasses.dataclass
class KOFTLinkerParameters(KalmanLinkerParameters):
    """Parameters of KOFTLinker.

    Note:
        Most parameters can be estimated automatically from the detections using `estimate`.

    Attributes:
        association_threshold (float): This is the main hyperparameter, it defines the threshold on the distance used
            not to link tracks with detections. A low threshold will typically reduce wrong assignments and ID-switches,
            but may increase track fragmentation. Higher values will reduce track fragmentation, but miss-detected
            tracks may be linked to a wrong detection.
            Depending on `cost`, it is either expressed the maximum Euclidean distance (pixels), or the maximum
            Mahalanobis distance, or the minimum likelihood (probability).
            Default: -1.0 (automatically estimated, see `estimate`.)
        detection_std (float | torch.Tensor): Expected measurement noise (in pixel) on the detection process.
            The detection process is modeled with a Gaussian noise with this given std. You may provide a different
            noise for each dimension. See `torch_kf.ckf.constant_kalman_filter`.
            Default: 0.0 (automatically estimated, see `estimate`.)
        flow_std (float | torch.Tensor): Expected measurement noise (in pixel) on the optical flow process.
            The optical flow process is modeled with a Gaussian noise with this given std. You may provide a different
            noise for each dimension.
            Default: 0.0 (coarse automatic estimation, see `estimate`)
        process_std (float | torch.Tensor): Expected process noise (in pixel). See `torch_kf.ckf.constant_kalman_filter`
            The process is modeled as constant order-th derivative motion with a Gaussian noise. This quantify how much
            the supposedly "constant" order-th derivative can change between two consecutive frames.
            A common rule of thumb is to use 4 * process_std ~= max_t(| dx^(order)(t+1) - dx^(order)(t)|) (see
            `estimate_process_std_from_tracks`). It can be provided for each dimension.
            Default: 0.0 (coarse automatic estimation, see `estimate`)
        kalman_order (int): Order of the Kalman filter to use.
            0 for brownian motions, 1 for directed brownian motion, 2 for accelerated brownian motions, etc...
            Default: 1
        n_valid (int): Number of detections required to validate the track after its creation. If a track is missed
            during its first n_valid frames, it is dropped. This provides robustness to false positive detections.
            With no false positives, it can be set to 1 (a detection always belongs to a track).
            Highers values allow to remove non time-consistent false positives, but may prune real tracks that have
            been miss-detected.
            Default: 3
        n_gap (int): Number of consecutive frames without any association (miss-detected) before the track termination.
            This provides robustness to false negative detections. Without any false negatives, it can be set to 0.
            Higher values allow to support larger gaps in the track, but may lead to wrong assignments.
            Default: 3
        association_method (AssociationMethod): The frame-by-frame association to use. See `AssociationMethod`.
            It can be provided as a string. Choice: GREEDY, OPT_HARD, OPT_SMOOTH, SPARSE_OPT_HARD, SPARSE_OPT_SMOOTH.
            Default: SPARSE_OPT_SMOOTH
        anisotropy (tuple[float, float, float]): Anisotropy of images (Ratio of the pixel sizes
            for each axis, depth first). This will be used to scale distances. Note that it will only impact
            EUCLIDEAN[_SQ] costs; for probabilistic cost, anisotropy should be already integrated
            within the stds of the kalman filter (providing one std for each dimension).
            Default: (1., 1., 1.)
        cost_method (CostMethod): The cost method to use. See `CostMethod`.
            It can be provided as a string. Choice: EUCLIDEAN, EUCLIDEAN_SQ, MAHALANOBIS, MAHALANOBIS_SQ, LIKELIHOOD.
            This also defines the unit of `association_threshold` (in pixels for Euclidean, no units for Mahalanobis,
            and a probability for likelihood).
            Default: LIKELIHOOD
        track_building (TrackBuilding): How the linker will build the final tracks. See `TrackBuilding`.
            Either from detections, or from filtered/smoothed positions computed by the
            Kalman filter. It can be provided as a string. Choice: DETECTION, FILTERED, SMOOTHED.
            Default: SMOOTHED
        split_factor (float): Allow splitting of tracks, using a second association step.
            The association threshold in this case is `split_factor * association_threshold`.
            Default: 0.0 (No splits)
        merge_factor (float): Allow merging of tracks, using a second association step.
            The association threshold in this case is `merge_factor * association_threshold`.
            Default: 0.0 (No merges)
        extract_flows_on_detections (bool): If True it extracts the optical flow at the detection location if possible.
            Otherwise it extract the flow from the current estimate of the track position.
            Default: False
        always_measure_velocity (bool): Update velocity for all tracks even non-linked ones.
            If set to False, it implements KOFT-- from the paper. This is sub-optimal, you should keep it True.
            Default: True
        online_process_std (float): Recomputes the process std online following "A. Genovesio, et al, 2004, October.
            Adaptive gating in Gaussian Bayesian multi-target tracking. ICIP'04. (Vol. 1, pp. 147-150). IEEE."
            Each track has its own process std depending on the errors made in the past. It automatically adjusts to
            process errors, allowing to increase the validation gate. Should be used in conjunction with MAHALANOBIS
            or LIKELIHOOD `cost_method`. As this may be detrimental, it is disabled by default.
            Default: 0.0 (Process_std is constant)
        initial_std_factor (float): The uncertainties on initial velocities/accelerations are set
            to initial_std_factor * process_std. See `KalmanLinker.build_initial_covariance`.
            Having a small factor will prevent handling correctly starting tracks with large initial velocity
            on their first frames.
            Typical values lies between 3.0 to 100.0.
            Default: 5.0

    """

    def __init__(  # noqa: PLR0913
        self,
        association_threshold: float = -1.0,
        *,
        detection_std: float | torch.Tensor = 0.0,
        flow_std: float | torch.Tensor = 0.0,
        process_std: float | torch.Tensor = 0.0,
        kalman_order: int = 1,
        n_valid=3,
        n_gap=3,
        association_method: str | AssociationMethod = AssociationMethod.SPARSE_OPT_SMOOTH,
        anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
        cost: str | Cost = Cost.LIKELIHOOD,
        track_building: str | TrackBuilding = TrackBuilding.SMOOTHED,
        split_factor: float = 0.0,
        merge_factor: float = 0.0,
        extract_flows_on_detections=False,
        always_measure_velocity=True,
        online_process_std=0.0,
        initial_std_factor=5.0,
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
                "A single flow_std is provided, but the images are anisotrope. Consider giving one std by dimension.",
                stacklevel=2,
            )

        self.flow_std = flow_std
        self.extract_flows_on_detections = extract_flows_on_detections
        self.always_measure_velocity = always_measure_velocity

    flow_std: float | torch.Tensor = 0.0
    extract_flows_on_detections: bool = False
    always_measure_velocity: bool = True

    @override
    def check(self):
        super().check()

        if torch.as_tensor(self.flow_std).min() <= 0:
            raise ValueError("`flow_std` should be greater than 0. Consider calling `estimate_flow_std_from_tracks`.")

    @override
    def estimate(self, detections_sequence: Sequence[byotrack.Detections]) -> KOFTLinkerParameters:
        """Estimate parameters from the given detections.

        Estimation is triggered by providing negative dummy values for positive parameters. The dummy values are
        then replaced by their estimate.

        Estimators:
        * detection_std: `average_radius` / 2 (i.e. localization is rarely predicted outside the target)
        * process_std: `average_radius` (i.e. unmodeled motion is ~the size of targets)
                       (Consider using `estimate_process_std_from_tracks`)
        * flow_std: `average_radius` (i.e. the flow can be wrong up to twice the target size)
                       (Consider using `estimate_flow_std_from_tracks`)
        * association_threshold: `steady_state_covariance` * 3 (See `estimate_association_threshold`).
        * anisotropy: Computed from `statistics.anisotropy`.
        * split_factor: 1.0 if the number of detection increase by more than 30% over the full sequence.
        * merge_factor: 1.0 if the number of detection decrease by more than 30% over the full sequence.

        Args:
            detections_sequence (Sequence[byotrack.Detections]): Detections for the current sequence.

        Returns:
            KOFTLinkerParameters: self with updated parameters.
        """
        if torch.as_tensor(self.flow_std).min() <= 0:
            warnings.warn(
                "`flow_std` estimation is coarse. Consider using `estimate_flow_std_from_tracks`.", stacklevel=2
            )
            avg_radius = statistics.average_radius(detections_sequence)
            self.flow_std = avg_radius

        super().estimate(detections_sequence)

        return self

    def estimate_flow_std_from_tracks(
        self,
        video: Sequence[np.ndarray] | np.ndarray,
        optflow: byotrack.OpticalFlow,
        tracks: Collection[byotrack.Track],
        quantile: float = 0.99993,
    ) -> None:
        """Estimate `flow_std` based on the errors made by the flow versus ground-truth tracks.

        Modify in place `flow_std`. It sets the flow_std so that it fits with the maximum flow errors on the
        annotations.

        NOTE: Without annotations, you may set the flow_std according to the following method:
              Manually check how the flow moves over your targets (see `InteractiveFlowVisualizer`) and
              estimate a coarse maximum error (in pixel) between two consevutive frames. Then flow_std can be set
              as this maximum error divided by 4.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): Sequence of T frames (array) on which to measure the flow_std.
                Each array is expected to have a shape ([D, ]H, W, C).
            optflow (byotrack.OptFlow): Optical flow algorithm that will be used in KOFT. The flow_std will be measured
                for this optical flow.
            tracks (Collection[byotrack.Track]): Partial ground-truth tracks. If these are manually annotated tracks,
                consider using a RTSSmoother to reduce the annotation noise.
            quantile (float): Quantile to extract the maximum value. Can be reduced to ignore some false positive links.
                Default: 0.99993
        """
        start = min(track.start for track in tracks)
        end = min(max(track.start + len(track) for track in tracks), len(video))
        points = byotrack.Track.tensorize(tracks, frame_range=(start, end))
        predicted = torch.full_like(points, torch.nan)
        valid = ~torch.isnan(points).any(dim=-1)

        src = optflow.preprocess(video[start])
        for frame_id in tqdm.trange(1, end - start):
            dst = optflow.preprocess(video[start + frame_id])
            flow_map = optflow.compute(src, dst)

            predicted[frame_id, valid[frame_id - 1]] = torch.from_numpy(
                optflow.transform(flow_map, points[frame_id - 1, valid[frame_id - 1]].numpy())
            )

            src = dst

        errors = (predicted[1:] - points[1:]).norm(dim=-1)  # ~ Chi distribution
        errors = errors[~torch.isnan(errors)]  # Remove NaNs

        # Ensure quantile leaves one out
        n_samples = len(errors)
        n_outliers = (1 - quantile) * n_samples
        if n_outliers < 1:
            quantile = max(0.5, 1 - 1 / n_samples)

        self.flow_std = errors.quantile(quantile).item() / scipy.stats.chi.ppf(quantile, points.shape[-1])

        # How should anisotropy be handled ? Errors are probably not scaled with anisotropy ?
        # We could check errors per axis but probably not very robust

    @override
    def build_filter(self, dim: int) -> torch_kf.KalmanFilter:
        kalman_filter = torch_kf.ckf.constant_kalman_filter(
            self.detection_std,
            self.process_std,
            dim=dim,
            order=self.kalman_order + (self.kalman_order == 0),
            approximate=True,  # Approximate so that a flow precisely means the velocity modeled here
        )
        if self.kalman_order == 0:
            # In order 0, we still model velocity, but we always predict it at 0
            kalman_filter.process_matrix[dim:, dim:] = 0

        # Doubles the measurement space to measure velocity
        kalman_filter.measurement_matrix = torch.eye(dim * 2, kalman_filter.state_dim)
        kalman_filter.measurement_noise = torch.eye(dim * 2)
        kalman_filter.measurement_noise[:dim, :dim] *= self.detection_std**2
        kalman_filter.measurement_noise[dim:, dim:] *= self.flow_std**2

        return kalman_filter


class KOFTLinker(KalmanLinker):
    """Kalman and Optical Flow Tracking [9].

    Motion is modeled with a Kalman filter of a specified order >= 1 (See `torch_kf.ckf`)
    Positions are measured through the detection process. A second update step is performed
    to measure the velocity of all tracks using optical flow.

    Matching is done to optimize the given cost.

    See `KalmanLinker` for the other attributes.

    Attributes:
        specs (KOFTLinkerParameters): Parameters specifications of the algorithm.
            See `KOFTLinkerParameters`.
        last_detections (byotrack.Detections): The last detections used in update.
            Optionally used to extract flows at the detection positions and not the track state.
            Required for `motion_model`

    """

    progress_bar_description = "KOFT linking"

    def __init__(
        self,
        specs: KOFTLinkerParameters,
        optflow: byotrack.OpticalFlow | None = None,
        features_extractor: byotrack.FeaturesExtractor | None = None,
        *,
        save_all=False,
    ) -> None:
        super().__init__(specs, optflow, features_extractor, save_all=save_all)

        self.optflow: OnlineFlowExtractor
        self.specs: KOFTLinkerParameters
        if self.optflow is None:
            raise ValueError("KOFT requires an optical flow algorithm")

        self.last_detections: byotrack.Detections = byotrack.PointDetections(torch.empty((0, 2), dtype=torch.float32))

    @override
    def reset(self, dim=2) -> None:
        super().reset(dim)

        # Reset the KF initialized by super
        self.kalman_filter = self.specs.build_filter(dim)
        self.kalman_filter = self.kalman_filter.to(self.dtype)

        self.active_states = torch_kf.GaussianState(
            torch.empty((0, self.kalman_filter.state_dim, 1), dtype=self.dtype),
            torch.empty((0, self.kalman_filter.state_dim, self.kalman_filter.state_dim), dtype=self.dtype),
        )
        self.projections = self.kalman_filter.project(self.active_states)

        self.last_detections = byotrack.PointDetections(torch.empty((0, dim), dtype=torch.float32))

    @override
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
            mask = torch.full((len(self.active_tracks),), fill_value=True)
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
        if (self.save_all or self.specs.track_building == TrackBuilding.SMOOTHED) and self.all_states:
            self.all_states[-1] = self.active_states

        # Use the Kalman filter to predict the current states of each active tracks
        self.active_states = self.kalman_filter.predict(
            self.active_states, process_noise=self.process_noises if self.specs.online_process_std else None
        )

        # Project states for association
        self.projections = self.kalman_filter.project(self.active_states)

    @override
    def cost(self, frame: np.ndarray | None, detections: byotrack.Detections) -> tuple[torch.Tensor, float]:
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

    @override
    def post_association(
        self, frame: np.ndarray | None, detections: byotrack.Detections, active_mask: torch.Tensor
    ) -> None:
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
        # initial_std_factor * process_std:
        # Small factors (~1) are not suited for tracking objects that appears with a large initial velocity.
        # Large factors (>> 1) will create association problem on the second frame, as the predicted
        # position inherit from this uncertainty.
        # As in KOFT, the velocities are measured, it could be set to a higher value.
        # For the Brownian motion, the process_std directly gives the initial uncertainty on velocity
        initial_state = torch_kf.GaussianState(
            torch.zeros(len(unmatched_measures), self.kalman_filter.state_dim, 1, dtype=self.dtype),
            torch.stack([self.build_initial_covariance(detections.dim)] * (1 + len(unmatched_measures)))[1:],
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
