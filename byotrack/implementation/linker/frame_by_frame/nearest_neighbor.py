import dataclasses
from typing import Optional, Tuple, Union
import warnings

import numpy as np
import torch

import byotrack
from .base import AssociationMethod, FrameByFrameLinker, FrameByFrameLinkerParameters


@dataclasses.dataclass
class NearestNeighborParameters(FrameByFrameLinkerParameters):
    """Parameters of NearestNeighborLinker

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

    def __init__(
        self,
        association_threshold: float = 5.0,
        *,
        n_valid=3,
        n_gap=3,
        association_method: Union[str, AssociationMethod] = AssociationMethod.OPT_SMOOTH,
        anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        ema=0.0,
        fill_gap=False,
        split_factor: float = 0.0,
        merge_factor: float = 0.0,
    ):
        super().__init__(  # pylint: disable=duplicate-code
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

    ema: float = 0.0
    fill_gap: bool = False


class NearestNeighborLinker(FrameByFrameLinker):
    """Frame by frame linker by associating with the closest detections

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
        optflow: Optional[byotrack.OpticalFlow] = None,
        features_extractor: Optional[byotrack.FeaturesExtractor] = None,
        save_all=False,
    ) -> None:
        super().__init__(specs, optflow, features_extractor, save_all)
        self.specs: NearestNeighborParameters
        self.active_positions = torch.zeros(0, 2)

        if self.specs.fill_gap and not self.optflow:
            warnings.warn("Optical flow has not been provided. Gap cannot be filled")

    def reset(self, dim=2) -> None:
        super().reset(dim)
        self.active_positions = torch.zeros(0, dim)

    def motion_model(self) -> None:
        if self.optflow and self.optflow.flow_map is not None:
            self.active_positions = torch.tensor(
                self.optflow.optflow.transform(self.optflow.flow_map, self.active_positions.numpy())
            )

    def cost(self, _: np.ndarray, detections: byotrack.Detections) -> Tuple[torch.Tensor, float]:
        anisotropy = torch.tensor(self.specs.anisotropy)[-detections.dim :]

        return (
            torch.cdist(self.active_positions * anisotropy, detections.position * anisotropy),
            self.specs.association_threshold,
        )

    def post_association(self, _: np.ndarray, detections: byotrack.Detections, active_mask: torch.Tensor):
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
