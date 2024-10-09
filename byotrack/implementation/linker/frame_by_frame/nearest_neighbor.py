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

    """

    def __init__(
        self,
        association_threshold: float = 5.0,
        *,
        n_valid=3,
        n_gap=3,
        association_method: Union[str, AssociationMethod] = AssociationMethod.OPT_SMOOTH,
        ema=0.0,
        fill_gap=False,
    ):
        super().__init__(  # pylint: disable=duplicate-code
            association_threshold=association_threshold,
            n_valid=n_valid,
            n_gap=n_gap,
            association_method=association_method,
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
        active_positions (Optional[torch.Tensor]): The positions of actives tracks,
            if undetected it is estimated by optical flow propagation.
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
        self.active_positions: Optional[torch.Tensor] = None

        if self.specs.fill_gap and not self.optflow:
            warnings.warn("Optical flow has not been provided. Gap cannot be filled")

    def reset(self) -> None:
        super().reset()
        self.active_positions = None

    def motion_model(self) -> None:
        if self.optflow and self.optflow.flow_map is not None:
            if self.active_positions is not None:
                self.active_positions = torch.tensor(
                    self.optflow.optflow.transform(self.optflow.flow_map, self.active_positions.numpy())
                )

    def cost(self, _: np.ndarray, detections: byotrack.Detections) -> Tuple[torch.Tensor, float]:
        if self.active_positions is None:
            self.active_positions = torch.empty((0, detections.position.shape[1]))

        return torch.cdist(self.active_positions, detections.position), self.specs.association_threshold

    def post_association(self, _: np.ndarray, detections: byotrack.Detections, links: torch.Tensor):
        if self.active_positions is None:
            self.active_positions = torch.empty((0, detections.position.shape[1]))

        # Update tracks positions with detections
        # Optionally using an EMA to reduce detections noise
        self.active_positions[links[:, 0]] -= (1.0 - self.specs.ema) * (
            self.active_positions[links[:, 0]] - detections.position[links[:, 1]]
        )

        # Update active track handlers
        active_mask = self.update_active_tracks(links)

        # Create new track handlers for unmatched detections
        unmatched = self.handle_extra_detections(detections, links)

        # Merge still active positions and new ones
        self.active_positions = torch.cat((self.active_positions[active_mask], detections.position[unmatched]))

        self.all_positions.append(self.active_positions.clone())

        if not self.specs.fill_gap:  # Erase optical flow predictions from the stored positions
            for i, track in enumerate(self.active_tracks):
                if track.detection_ids[-1] == -1:  # Not truly detected
                    self.all_positions[-1][i, :] = torch.nan
