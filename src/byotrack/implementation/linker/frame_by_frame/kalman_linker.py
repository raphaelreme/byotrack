from __future__ import annotations

import dataclasses
import enum
import sys
import warnings
from typing import TYPE_CHECKING

import scipy
import torch
import torch_kf
import torch_kf.ckf

import byotrack
from byotrack.api.detections import statistics
from byotrack.implementation.linker.frame_by_frame.nearest_neighbor import (
    AssociationMethod,
    FrameByFrameLinker,
    FrameByFrameLinkerParameters,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import numpy as np

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class Cost(enum.Enum):
    """The cost to use for association.

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
    """How to build the final tracks.

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
    """Parameters of KalmanLinker.

    Note:
        Most parameters can be estimated automatically from the detections using `estimate`.

    Attributes:
        association_threshold (float): This is the main hyperparameter, it defines the threshold on the distance used
            not to link tracks with detections. A low threshold will typically reduce wrong assignments and ID-switches,
            but may increase track fragmentation. Higher values will reduce track fragmentation, but miss-detected
            tracks may be linked to a wrong detection.
            Depending on `cost`, it is either expressed the maximum Euclidean distance (pixels), or the maximum
            Mahalanobis distance, or the minimum likelihood (probability).
            Default: -1.0 (to be estimated, see `estimate`.)
        detection_std (float | torch.Tensor): Expected measurement noise (in pixel) on the detection process.
            The detection process is modeled with a Gaussian noise with this given std. You may provide a different
            noise for each dimension. See `torch_kf.ckf.constant_kalman_filter`.
            Default: 0.0 (to be estimated, see `estimate`.)
        process_std (float | torch.Tensor): Expected process noise (in pixel). See `torch_kf.ckf.constant_kalman_filter`
            The process is modeled as constant order-th derivative motion with a Gaussian noise. This quantify how much
            the supposedly "constant" order-th derivative can change between two consecutive frames.
            A common rule of thumb is to use 4 * process_std ~= max_t(| dx^(order)(t+1) - dx^(order)(t)|) (see
            `estimate_process_std_from_tracks`). It can be provided for each dimension.
            Default: 0.0 (to be estimated, see `estimate`)
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
        online_process_std (float): Recomputes the process std online following "A. Genovesio, et al, 2004, October.
            Adaptive gating in Gaussian Bayesian multi-target tracking. ICIP'04. (Vol. 1, pp. 147-150). IEEE."
            Each track has its own process std depending on the errors made in the past. It automatically adjusts to
            process errors, allowing to increase the validation gate. Should be used in conjunction with MAHALANOBIS
            or LIKELIHOOD `cost_method`. As this may be detrimental, it is disabled by default.
            Default: 0.0 (Process_std is constant)
        initial_std_factor (float): The uncertainties on initial velocities/accelerations are set
            to initial_std_factor * process_std. See `KalmanLinker.build_initial_covariance`.
            Having a small factor will prevent handling correctly starting tracks with large initial velocity
            on their first frames. But large values will lead to large uncertainty on the first prediction, making
            it hard to associate to a detection with MAHALANOBIS or LIKELIHOOD methods.
            Typical values lies between 3.0 to 10.0.
            Default: 5.0

    """

    def __init__(  # noqa: PLR0913
        self,
        association_threshold: float = -1.0,
        *,
        detection_std: float | torch.Tensor = 0.0,
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
        online_process_std: float = 0.0,
        initial_std_factor: float = 5.0,
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

        if isinstance(detection_std, float) and min(anisotropy) != max(anisotropy) and detection_std > 0:
            warnings.warn(
                "A single `detection_std` is provided, but images are anisotrope. Consider giving std per dimension.",
                stacklevel=2,
            )
        if isinstance(process_std, float) and min(anisotropy) != max(anisotropy) and process_std > 0:
            warnings.warn(
                "A single `process_std` is provided, but images are anisotrope. Consider giving std per dimension.",
                stacklevel=2,
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

    detection_std: float | torch.Tensor = 0.0
    process_std: float | torch.Tensor = 0.0
    kalman_order: int = 1
    cost: Cost = Cost.LIKELIHOOD
    track_building: TrackBuilding = TrackBuilding.SMOOTHED
    online_process_std: float = 0.0
    initial_std_factor: float = 5.0

    @override
    def check(self):
        super().check()
        if self.kalman_order < 0:
            raise ValueError("`kalman_order` should be greater than or equal to 0.")

        if self.online_process_std < 0:
            raise ValueError("`online_process_std` should be greater than or equal to 0.")

        if self.initial_std_factor <= 0:
            raise ValueError("`initial_std_factor` should be greater than 0.")

        if torch.as_tensor(self.detection_std).min() <= 0:
            raise ValueError("`detection_std` should be greater than 0. Consider calling `estimate`.")

        if torch.as_tensor(self.process_std).min() <= 0:
            raise ValueError("`process_std` should be greater than 0. Consider calling `estimate`.")

    @override
    def estimate(self, detections_sequence: Sequence[byotrack.Detections]) -> KalmanLinkerParameters:
        """Estimate parameters from the given detections.

        Estimation is triggered by providing negative dummy values for positive parameters. The dummy values are
        then replaced by their estimate.

        Estimators:

        * detection_std: `average_radius` / 2 (i.e. localization is rarely predicted outside the target)
        * process_std: `average_radius` (i.e. unmodeled motion is ~the size of targets)\
            (Consider using `estimate_process_std_from_tracks` instead)
        * association_threshold: `steady_state_covariance` * 3 (See `estimate_association_threshold`).
        * anisotropy: Computed from `statistics.anisotropy`.
        * split_factor: 1.0 if the number of detection increase by more than 30% over the full sequence.
        * merge_factor: 1.0 if the number of detection decrease by more than 30% over the full sequence.

        Args:
            detections_sequence (Sequence[byotrack.Detections]): Detections for the current sequence.

        Returns:
            NearestNeighborParameters: self with updated parameters.
        """
        estimate_association_threshold = False
        if self.association_threshold <= 0.0:  # Do not estimate threshold in super(), we will do it here anyway.
            self.association_threshold = 1.0
            estimate_association_threshold = True

        super().estimate(detections_sequence)

        if self.kalman_order < 0:
            warnings.warn("No estimation available for parameter `kalman_order`. Defaults to 1.", stacklevel=2)
            self.kalman_order = 1

        if self.online_process_std < 0.0:
            warnings.warn("No estimation available for parameter `online_process_std`. Defaults to 0.0.", stacklevel=2)
            self.online_process_std = 0.0

        if self.initial_std_factor <= 0.0:
            warnings.warn("No estimation available for parameter `initial_std_factor`. Defaults to 5.0.", stacklevel=2)
            self.initial_std_factor = 5.0

        avg_radius: float | None = None
        if torch.as_tensor(self.detection_std).min() <= 0.0:
            avg_radius = statistics.average_radius(detections_sequence, self.anisotropy)
            self.detection_std = torch.tensor([avg_radius / 2.0 / ani for ani in self.anisotropy], dtype=torch.float32)
            self.detection_std = self.detection_std[-detections_sequence[0].dim :]

        if torch.as_tensor(self.process_std).min() <= 0.0:
            avg_radius = (  # Recompute only if needed
                statistics.average_radius(detections_sequence, self.anisotropy) if avg_radius is None else avg_radius
            )

            warnings.warn(
                "`process_std` estimation is coarse. Consider setting it manually "
                "or from a few labeled tracks with `estimate_process_std_from_tracks`.",
                stacklevel=2,
            )

            self.process_std = torch.tensor([avg_radius / ani for ani in self.anisotropy], dtype=torch.float32)
            self.process_std = self.process_std[-detections_sequence[0].dim :]

        if estimate_association_threshold:
            self.estimate_association_threshold(detections_sequence[0].dim)

        return self

    def estimate_association_threshold(self, dim: int, mahalanobis_threshold: float = 3.0) -> None:
        """Estimate `association_threshold` based on the steady state covariance of the filter.

        Modify in place `association_threshold` so that it cuts at `mahalanobis_threshold` when the filter
        is in its steady state (i.e. after a few non-missed assignments).

        Args:
            dim (int): Dimension of the video (2D or 3D).
            mahalanobis_threshold (float): Threshold on the mahalanobis distance in the steady state.
                Default: 3.0
        """
        if torch.as_tensor(self.detection_std).min() <= 0:
            raise ValueError("`detection_std` should be greater than 0 to estimate `association_threshold`.")

        if torch.as_tensor(self.process_std).min() <= 0:
            raise ValueError("`process_std` should be greater than 0 to estimate `association_threshold`.")

        if self.cost in (Cost.MAHALANOBIS, Cost.MAHALANOBIS_SQ):
            self.association_threshold = mahalanobis_threshold

        kalman_filter = self.build_filter(dim)

        state = torch_kf.GaussianState(
            torch.zeros(kalman_filter.measure_dim, 1),
            kalman_filter.steady_state_covariance(predicted=True, projected=True),
        )
        # Cutoff is at maha times the steady_state_covariance.
        cutoff = state.covariance[0, 0].sqrt() * mahalanobis_threshold

        if self.cost in (Cost.EUCLIDEAN, Cost.EUCLIDEAN_SQ):
            self.association_threshold = cutoff.item()
        else:
            measure = torch.zeros((kalman_filter.measure_dim, 1))
            measure[0, 0] = cutoff
            self.association_threshold = state.likelihood(measure).item()

    def estimate_process_std_from_tracks(self, tracks: Collection[byotrack.Track], quantile=0.99993) -> None:
        """Estimate `process_std` based on the given tracks.

        It modifies in place `process_std` so that it roughly fits the maximum unmodeled motion.

        NOTE: Without annotations, you may set the process_std according to the following rule:
              For **kalman_order=0**, it can be set to the maximum velocity (in pixel) divided by 4, where you
              manually estimate the velocity visually (a rough estimation is enough).
              For **kalman_order=1**, it can be similarly be set to the maximum variation of velocity between
              consecutive frames divided by 4, with a rough visual estimation of the velocity variation.

        Args:
            tracks (Collection[byotrack.Track]): Partial ground-truth tracks. Note for a given `kalman_order`,
                only the given tracks with length >= kalman_order + 2 will be used.
                If these are manually annotated tracks, consider using a RTSSmoother to reduce the annotation noise.
            quantile (float): Quantile to extract the maximum value. Can be reduced to ignore some false positive links.
                Default: 0.99993
        """
        if any(ani <= 0 for ani in self.anisotropy):
            raise ValueError("`anisotropy` should be greater than 0 to estimate `process_std`.")

        points = byotrack.Track.tensorize(tracks)

        dim = points.shape[-1]
        points *= torch.tensor(self.anisotropy)[-dim:]  # Get isotrope positions

        for _ in range(self.kalman_order + 1):
            points = points[1:] - points[:-1]  # Compute the derivative at the right order

        unexpected_motion = points.norm(dim=-1)  # ~ Chi distribution
        unexpected_motion = unexpected_motion[~unexpected_motion.isnan()]  # Remove NaNs

        # Ensure quantile leaves one out
        n_samples = len(unexpected_motion)
        n_outliers = (1 - quantile) * n_samples
        if n_outliers < 1:
            # We assume that hard samples are given, if only a few annotations (So at least 90% of the distrib)
            quantile = max(0.9, 1 - 1 / n_samples)

        self.process_std = unexpected_motion.quantile(quantile).item() / scipy.stats.chi.ppf(quantile, dim)
        self.process_std = torch.tensor([self.process_std / ani for ani in self.anisotropy], dtype=torch.float32)
        self.process_std = self.process_std[-dim:]

    def build_filter(self, dim: int) -> torch_kf.KalmanFilter:
        """Build the Kalman filter used by the Linker.

        See `torch_kf.ckf.constant_kalman_filter`.
        """
        return torch_kf.ckf.constant_kalman_filter(
            self.detection_std,
            self.process_std,
            dim=dim,
            order=self.kalman_order,
            approximate=True,
        )


class KalmanLinker(FrameByFrameLinker):
    """Frame by frame linker using Kalman filters.

    Motion is modeled with a Kalman filter of a specified order (See `torch_kf.ckf`)
    Matching is done to optimize the given cost. If optical flow is provided, it is used
    online to warp the predicted state positions of the kalman filter. This will work, but it
    is sub-optimal: consider using `KOFTLinker` that exploits in a finer way optical flow
    inside Kalman filters.

    This is an implementation of Simple Kalman Tracking (SKT) from KOFT [9].

    See `FrameByFrameLinker` for the other attributes.

    Attributes:
        specs (KalmanLinkerParameters): Parameters specifications of the algorithm.
            See `KalmanLinkerParameters`.
        kalman_filter (torch_kf.KalmanFilter): The Kalman filter.
        active_states (torch_kf.GaussianState): The Kalman filter estimation for each track.
            Shape: mean=(N, D * (order + 1), 1), covariance=(N, D * (order + 1), dim * (order + 1))
            dtype: float
        projections (torch_kf.GaussianState): The Kalman filter projection for each track.
            Shape: mean=(N, D, 1), covariance=(N, D, D), precision=(N, D, D)
            dtype: float
        process_noises (torch.Tensor): The Kalman filter process noise for each track.
            Only used when online_process_std > 0.0. It allows to compute an adaptative process_std
            and therefore gating for each track.
            Shape: (N, D, 1), dtype: float
        all_states (list[torch_kf.GaussianState]): The Kalman filter estimation for each track at each seen
            frame. States are only registered when save_all=True or if you build tracks from RTS smoothing.
            Shape: mean=(N, D * (order + 1), 1), covariance=(N, D * (order + 1), dim * (order + 1))
            dtype: float

    """

    progress_bar_description = "Kalman filter linking"

    def __init__(
        self,
        specs: KalmanLinkerParameters,
        optflow: byotrack.OpticalFlow | None = None,
        features_extractor: byotrack.FeaturesExtractor | None = None,
        *,
        save_all=False,
    ) -> None:
        super().__init__(specs, optflow, features_extractor, save_all=save_all)

        self.specs: KalmanLinkerParameters
        self.kalman_filter = torch_kf.ckf.constant_kalman_filter(1.0, 1.0, dim=2, order=self.specs.kalman_order)
        self.dtype = torch.float32

        self.active_states = torch_kf.GaussianState(
            torch.empty((0, self.kalman_filter.state_dim, 1)),
            torch.empty((0, self.kalman_filter.state_dim, self.kalman_filter.state_dim)),
        )
        self.projections = self.kalman_filter.project(self.active_states)
        self.process_noises = torch.empty((0, self.kalman_filter.state_dim, self.kalman_filter.state_dim))

        self.all_states: list[torch_kf.GaussianState] = []

    @override
    def reset(self, dim=2) -> None:
        super().reset(dim)

        self.kalman_filter = self.specs.build_filter(dim)
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

    @override
    def collect(self) -> list[byotrack.Track]:
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
            is_defined = torch.full((len(self.all_states), len(tracks_handlers)), fill_value=False)

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

            return self._remove_single_split(tracks)

        return super().collect()

    @override
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

    @override
    def cost(self, frame: np.ndarray | None, detections: byotrack.Detections) -> tuple[torch.Tensor, float]:
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

    @override
    def post_association(
        self, frame: np.ndarray | None, detections: byotrack.Detections, active_mask: torch.Tensor
    ) -> None:
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
        # initial_std_factor * process_std:
        # Small factors (~1) are not suited for tracking objects that appears with a large initial velocity.
        # Large factors (>> 1) will create association problem on the second frame, as the predicted
        # position inherit from this uncertainty.
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

    def build_initial_covariance(self, dim: int) -> torch.Tensor:
        """Build the diagonal initial covariance matrix.

        The position is initially unknown, leading to a belief (given by the first detection) set
        to the position of the first detection, with detection_std uncertainty.

        The velocity (and higher order derivatives) are assumed to be 0.0 with a relatively high uncertainty:
        initial_std_factor * process_std.

        Note that having a large initial_std_factor (>10) may decrease performances, as the first prediction
        will be impacted and largely uncertain, leading to low probabilities for every associations. In KOFT,
        as the velocity is measured before the first prediction, the initial_std_factor can be increased to
        reduce this bias toward a nul initial velocity. We found that initial_std_factor=0.0 is a good
        trade off in practice.
        """
        process_std = torch.broadcast_to(torch.as_tensor(self.specs.process_std, dtype=self.dtype), (dim,)).clone()

        # In the case of Brownian motion, the initial covariance if fully rewritten in SKT (no impact)
        # But in KOFT, Brownian motion models velocity with an initial velocity centered on 0 and with
        # an uncertainty given by the process_std, therefore we don't use initial_std_factor
        if self.specs.kalman_order > 0:
            process_std *= self.specs.initial_std_factor

        # Process std is squared and set for each order of the process (pos, vel, acc, ...)
        covariance = torch.diag(torch.cat([process_std**2] * (self.kalman_filter.state_dim // dim)))

        # Then, it is overwritten for the position, to be set to measurement_std
        measurement_std = torch.broadcast_to(torch.as_tensor(self.specs.detection_std, dtype=self.dtype), (dim,))
        torch.diagonal(covariance)[:dim] = measurement_std**2

        return covariance
