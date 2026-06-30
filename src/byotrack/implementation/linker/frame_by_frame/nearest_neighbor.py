from __future__ import annotations

import dataclasses
import sys
import warnings
from typing import TYPE_CHECKING

import torch

from byotrack.implementation.linker.frame_by_frame.base import (
    AssociationMethod,
    FrameByFrameLinker,
    FrameByFrameLinkerParameters,
)

if TYPE_CHECKING:
    import numpy as np

    import byotrack

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


@dataclasses.dataclass
class NearestNeighborParameters(FrameByFrameLinkerParameters):
    """Parameters of NearestNeighborLinker.

    Note:
        Most parameters can be estimated automatically from the detections using `estimate`.

    Attributes:
        association_threshold (float): This is the main hyperparameter, it defines the threshold on the distance used
            not to link tracks with detections. A low threshold will typically reduce wrong assignments and ID-switches,
            but may increase track fragmentation. Higher values will reduce track fragmentation, but miss-detected
            tracks may be linked to a wrong detection.
            Default: -1.0 (automatically estimated, see `estimate`.)
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
            It can be provided as a string. (Choice: GREEDY, OPT_HARD, OPT_SMOOTH, SPARSE_OPT_HARD, SPARSE_OPT_SMOOTH)
            Default: SPARSE_OPT_SMOOTH
        anisotropy (tuple[float, float, float]): Anisotropy of images (Ratio of the pixel sizes
            for each axis, depth first). This will be used to scale distances.
            Default: (1., 1., 1.)
        ema (float): Optional exponential moving average to reduce detection noise. Detection positions are smoothed
            using this EMA. Should be smaller than 1. It use: x_{t+1} = ema x_{t} + (1 - ema) det(t)
            As motion is not modeled, EMA may introduce lag that will hinder tracking. It is more effective with
            optical flow to compensate motions, in this case, a typical value is 0.5, to average the previous position
            with the current measured one. For more advanced modelisation, see `KalmanLinker`.
            Default: 0.0 (No EMA)
        fill_gap (bool): Fill the gap of missed detections using a forward optical flow
            propagation (Only when optical flow is provided). We advise to rather use a
            ForwardBackward interpolation using the same optical flow: it will produce
            smoother interpolations.
            Default: False
        split_factor (float): Allow splitting of tracks, using a second association step.
            The association threshold in this case is `split_factor * association_threshold`.
            Default: 0.0 (No splits)
        merge_factor (float): Allow merging of tracks, using a second association step.
            The association threshold in this case is `merge_factor * association_threshold`.
            Default: 0.0 (No merges)

    """

    def __init__(  # noqa: PLR0913
        self,
        association_threshold: float = -1.0,
        *,
        n_valid=3,
        n_gap=3,
        association_method: str | AssociationMethod = AssociationMethod.SPARSE_OPT_SMOOTH,
        anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ema=0.0,
        fill_gap=False,
        split_factor: float = 0.0,
        merge_factor: float = 0.0,
    ):
        super().__init__(
            association_threshold=association_threshold,
            n_valid=n_valid,
            n_gap=n_gap,
            association_method=association_method,
            anisotropy=anisotropy,
            split_factor=split_factor,
            merge_factor=merge_factor,
        )
        self.ema = ema
        self.fill_gap = fill_gap

        if self.ema >= 1.0:
            raise ValueError("`ema` should be lower than 1.0.")

    ema: float = 0.0
    fill_gap: bool = False

    @override
    def check(self):
        super().check()
        if not 0 <= self.ema < 1.0:
            raise ValueError("`ema` should in [0, 1[.")

    @override
    def estimate(self, detections_sequence) -> NearestNeighborParameters:
        """Estimate parameters from the given detections.

        Estimation is triggered by providing negative dummy values for positive parameters. The dummy values are
        then replaced by their estimate.

        Estimators:
        * association_threshold: max(3 * `statistics.average_radius`, `statistics.average_min_dist`)
        * anisotropy: Computed from `statistics.anisotropy`.
        * split_factor: 1.0 if the number of detection increase by more than 30% over the full sequence.
        * merge_factor: 1.0 if the number of detection decrease by more than 30% over the full sequence.

        Args:
            detections_sequence (Sequence[byotrack.Detections]): Detections for the current sequence.

        Returns:
            NearestNeighborParameters: self with updated parameters.
        """
        super().estimate(detections_sequence)

        if self.ema < 0:
            warnings.warn("No estimation available for parameter `ema`. Defaults to 0.0.", stacklevel=2)
            self.ema = 0.0

        return self


class NearestNeighborLinker(FrameByFrameLinker):
    """Frame by frame linker by associating with the closest detections.

    Motion is not modeled, but if an optical flow method is provided, it
    will be used to compensate motion online. Matching is done from a simple Euclidean
    distance. This can be easily changed by inheriting this class and overriding the `cost` method.

    See `FrameByFrameLinker` for the other attributes.

    Attributes:
        specs (NearestNeighborParameters): Parameters specifications of the algorithm.
            See `NearestNeighborParameters`.
        active_positions (torch.Tensor): The positions of actives tracks, if undetected it is estimated by
            optical flow propagation, or simply propagated if no optical flow is given.
            Shape: (N, D), dtype: float32

    """

    progress_bar_description = "Nearest Neighbor linking"

    def __init__(
        self,
        specs: NearestNeighborParameters,
        optflow: byotrack.OpticalFlow | None = None,
        features_extractor: byotrack.FeaturesExtractor | None = None,
        *,
        save_all=False,
    ) -> None:
        super().__init__(specs, optflow, features_extractor, save_all=save_all)
        self.specs: NearestNeighborParameters
        self.active_positions = torch.zeros(0, 2)

        if self.specs.fill_gap and not self.optflow:
            warnings.warn("Optical flow has not been provided. Gap cannot be filled", stacklevel=2)

    @override
    def reset(self, dim=2) -> None:
        super().reset(dim)
        self.active_positions = torch.zeros(0, dim)

    @override
    def motion_model(self) -> None:
        if self.optflow and self.optflow.flow_map is not None:
            self.active_positions = torch.tensor(
                self.optflow.optflow.transform(self.optflow.flow_map, self.active_positions.numpy())
            )

    @override
    def cost(self, frame: np.ndarray | None, detections: byotrack.Detections) -> tuple[torch.Tensor, float]:
        anisotropy = torch.tensor(self.specs.anisotropy)[-detections.dim :]

        return (
            torch.cdist(self.active_positions * anisotropy, detections.position * anisotropy),
            self.specs.association_threshold,
        )

    @override
    def post_association(
        self, frame: np.ndarray | None, detections: byotrack.Detections, active_mask: torch.Tensor
    ) -> None:
        # Update tracks positions with detections
        # Optionally using an EMA to reduce detections noise
        self.active_positions[self._links[:, 0]] -= (1.0 - self.specs.ema) * (
            self.active_positions[self._links[:, 0]] - detections.position[self._links[:, 1]]
        )

        # Merge still active positions and new ones
        self.active_positions = torch.cat(
            (self.active_positions[active_mask], detections.position[self._unmatched_detections])
        )

        self.all_positions.append(self.active_positions.clone())

        if not self.specs.fill_gap:  # Erase optical flow predictions from the stored positions
            for i, track in enumerate(self.active_tracks):
                if track.detection_ids[-1] == -1:  # Not truly detected
                    self.all_positions[-1][i, :] = torch.nan
