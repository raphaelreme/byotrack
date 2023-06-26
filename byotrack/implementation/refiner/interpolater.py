from typing import Callable, Collection, Iterable, List, Union

import numpy as np
import torch

import byotrack

from . import propagation


class ForwardBackwardInterpolater(byotrack.Refiner):  # pylint: disable=too-few-public-methods
    """Interpolate tracks to fill out NaN values by averaging a forward estimation and a backward estimation.

    We propose two modes for position propagation: constant and tps. Constant estimation propagates the
    last known position, whereas tps uses active tracks knowledge to estimate the elastic motion.

    Forward and backward propagation are smoothly merged together in a weighted average. See `propagation` module

    Attributes:
        method (str | Callable[..., torch.Tensor]): Method for directed propagation
            ("constant", "tps" or your own callable)
            Default: "constant"
        full (bool): Interpolate track on all frames or only on the track range
            Default: False (Will not extend partial tracks, but only fill out NaN values inside of them)
        **kwargs: Additional arguments given to the method (For instance, `alpha` for tps)

    """

    def __init__(self, method: Union[str, Callable[..., torch.Tensor]] = "constant", full=False, **kwargs) -> None:
        super().__init__()
        self.method = method
        self.full = full
        self.kwargs = kwargs

    def run(self, video: Iterable[np.ndarray], tracks: Collection[byotrack.Track]) -> List[byotrack.Track]:
        tracks_matrix = byotrack.Track.tensorize(tracks)
        propagation_matrix = propagation.forward_backward_propagation(tracks_matrix, self.method, **self.kwargs)

        new_tracks = []
        for i, track in enumerate(tracks):
            if self.full:
                start = 0
                points = propagation_matrix[:, i]
            else:
                start = track.start
                points = propagation_matrix[start : start + len(track), i]
            new_tracks.append(byotrack.Track(start, points, track.identifier))

        return new_tracks
