from typing import Collection, List, Sequence, Union

import numpy as np
import torch

import byotrack

from . import propagation


class ForwardBackwardInterpolater(byotrack.Refiner):  # pylint: disable=too-few-public-methods
    """Interpolate tracks to fill out NaN values by averaging a forward estimation and a backward estimation.

    We propose three modes for position propagation: constant, tps and flow. Constant estimation propagates the
    last known position, tps interpolates elastically the motion of the active tracks to fill missing ones and
    flow uses optical flow to propagate the tracks.

    Forward and backward propagation are smoothly merged together in a weighted average. See `propagation` module

    Attributes:
        method (str | DirectedPropagate): Method to use for position propagation. Either "constant", "tps"
            or "flow" (see `constant_directed_propagate`, `tps_directed_propagate` or `optical_flow_directed_propagate`)
            or a user defined Callable following the `DirectedPropagate` protocol.
            The method will be called with the tracks_matrix, video, a boolean direction and additional kwargs.
        full (bool): If True, it will extrapolate positions for all the frames of the video
            If False, it will only interpolate NaN values inside each track domain, without extrapolation
            Default: False
        kwargs (Dict[str, Any]): Additional parameters given to `method`.
            For instance for "tps", you can set `alpha` (float) which is the regularization parameter of the
            interpolation (see `tps_propation` module and `torch_tps` librarie). We advise to use alpha > 5.0
            (alpha = 10 by default). For "flow" propagation, you should provide `optflow` (byotrack.OpticalFlow)
            to correctly estimate optical flows.

    """

    def __init__(self, method: Union[str, propagation.DirectedPropagate] = "constant", full=False, **kwargs) -> None:
        super().__init__()
        self.method = method
        self.full = full
        self.kwargs = kwargs

    def run(
        self, video: Union[Sequence[np.ndarray], np.ndarray], tracks: Collection[byotrack.Track]
    ) -> List[byotrack.Track]:
        if not tracks:
            return []

        tracks_matrix = byotrack.Track.tensorize(tracks, frame_range=(0, len(video)) if self.full else None)
        start, end = (0, len(video))
        if not self.full:
            start = min(track.start for track in tracks)
            end = max(track.start + len(track) for track in tracks)
            video = video[start:end]  # Clip video temporally so that it matches with the tracks_matrix
        propagation_matrix = propagation.forward_backward_propagation(tracks_matrix, video, self.method, **self.kwargs)

        new_tracks = []
        for i, track in enumerate(tracks):
            if self.full:
                track_start = 0
                points = propagation_matrix[:, i]
                detection_ids = torch.full((propagation_matrix.shape[0],), -1, dtype=torch.int32)
                detection_ids[track.start : track.start + len(track)] = track.detection_ids
            else:
                track_start = track.start
                points = propagation_matrix[track_start - start : track_start - start + len(track), i]
                detection_ids = track.detection_ids.clone()

            new_tracks.append(
                byotrack.Track(
                    track_start,
                    points,
                    track.identifier,
                    detection_ids,
                    merge_id=track.merge_id,
                    parent_id=track.parent_id,
                )
            )

        # Check produced tracks
        byotrack.Track.check_tracks(new_tracks, warn=True)

        return new_tracks
