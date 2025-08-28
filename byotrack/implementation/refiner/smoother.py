from typing import Collection, List, Sequence, Tuple, Union
import warnings

import numpy as np
import torch
import torch_kf
import torch_kf.ckf
import tqdm.auto as tqdm

import byotrack


# TODO: Add a KOFTSmoother ? that uses optical flow to smooth tracks as in KOFT
# TODO: Support for float64 and cuda ?


class RTSSmoother(byotrack.Refiner):  # pylint: disable=too-few-public-methods
    """Smooth tracks positions by running the optimal Rauch-Tung-Striebel smoothing.

    It will also fill out NaN values in tracks. (Except if a track starts with NaN values,
    the track will now starts at the first non-NaN value.)

    It models the tracks motion and position measurement using a Kalman filter framework (with Gaussian iid noise).
    In order to have consistent results, the tracks positions should be extracted as the positions of an independant
    detection process that is not already filtered/smoothed.

    See `torch_kf` and `filterpy` for more details.

    Note:
        This implementation requires torch-kf. (pip install torch-kf)

    Attributes:
        detection_std (Union[float, torch.Tensor]): Expected measurement noise on the detection process.
            The detection process is modeled with a Gaussian noise with this given std. (You can provide a different
            noise for each dimension). See `torch_kf.ckf.constant_kalman_filter`.
            Default: 3.0 pixels
        process_std (Union[float, torch.Tensor]): Expected process noise. See `torch_kf.ckf.constant_kalman_filter`, the
            process is modeled as constant order-th derivative motion. This quantify how much the supposely "constant"
            order-th derivative can change between two consecutive frames. A common rule of thumb is to use
            3 * process_std ~= max_t(| x^(order)(t) - x^(order)(t+1)|). It can be provided for each dimension).
            Default: 1.5 pixels / frame^order
        kalman_order (int): Order of the Kalman filter to use.
            0 for brownian motions, 1 for directed brownian motion, 2 for accelerated brownian motions, etc...
            Default: 1
        anisotropy (Tuple[float, float, float]): Deprecated and ignored. If the data is anisotrope, it should directly
            be incorporated inside detection_std and process_std. Will be removed in a future version.
        initial_std_factor (float): The initial state is created with an initial uncertainty
            set to the measurement_std of the Kalman Filter on measured states and this factor times
            the process_std for unmeasured states.
            Default: 10.0

    """

    description_forward = "RTS smoothing (forward)"
    description_backward = "RTS smoothing (backward)"

    def __init__(
        self,
        detection_std: Union[float, torch.Tensor] = 3.0,
        process_std: Union[float, torch.Tensor] = 1.5,
        kalman_order: int = 1,
        *,
        anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        initial_std_factor=10.0,
    ) -> None:
        super().__init__()
        self.detection_std = detection_std
        self.process_std = process_std
        self.kalman_order = kalman_order
        self.initial_std_factor = initial_std_factor

        if anisotropy != (1.0, 1.0, 1.0):
            warnings.warn(
                "'anisotropy' is deprecated and will be removed in a future version; it is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

    def run(  # pylint: disable=too-many-locals
        self, _: Union[Sequence[np.ndarray], np.ndarray], tracks: Collection[byotrack.Track]
    ) -> List[byotrack.Track]:
        if not tracks:
            return []

        dim = next(iter(tracks)).points.shape[-1]

        kalman_filter = torch_kf.ckf.constant_kalman_filter(
            self.detection_std,
            self.process_std,
            dim=dim,
            order=self.kalman_order,
        )

        # Starting time reference (positions[0] correspond to positions at start)
        start = min(track.start for track in tracks)

        positions = byotrack.Track.tensorize(tracks)
        is_defined = torch.full((positions.shape[0], positions.shape[1]), False)
        offsets: List[int] = []

        for i, track in enumerate(tracks):
            offset = int(torch.nonzero(~torch.isnan(track.points).any(dim=-1))[0, 0].item())
            if offset != 0:  # Handle tracks starting by NaN values
                warnings.warn("A track is starting with NaN values. It will be clipped.")

            is_defined[track.start + offset - start : track.start + len(track) - start, i] = True
            offsets.append(offset)

        # Initial state (see KalmanLinker)
        process_std = torch.broadcast_to(torch.as_tensor(self.process_std), (dim,)) * self.initial_std_factor
        process_std = torch.cat([process_std**2] * (kalman_filter.state_dim // dim))  # Expand to state_dim

        initial_state = torch_kf.GaussianState(
            torch.zeros(1, kalman_filter.state_dim, 1),
            torch.diag(process_std)[None],
        )
        initial_state.covariance[:, :dim, :dim] = kalman_filter.measurement_noise

        # Filtering
        states = torch_kf.GaussianState(
            torch.zeros(positions.shape[0], positions.shape[1], kalman_filter.state_dim, 1),
            torch.zeros(positions.shape[0], positions.shape[1], kalman_filter.state_dim, kalman_filter.state_dim),
        )

        was_defined = torch.full((positions.shape[1],), False)
        for frame_id in tqdm.trange(positions.shape[0], desc=self.description_forward):
            starting = ~was_defined & is_defined[frame_id]
            kept = was_defined & is_defined[frame_id]
            measured = ~torch.isnan(positions[frame_id]).any(dim=-1)

            # Prediction
            if frame_id > 0:
                states[frame_id, kept] = kalman_filter.predict(states[frame_id - 1, kept])

            # Initialization
            assert measured[starting].all()  # Always measured on start point
            states[frame_id, starting] = initial_state
            states.mean[frame_id, starting, :dim, 0] = positions[frame_id, starting]

            # Update
            states[frame_id, kept & measured] = kalman_filter.update(
                states[frame_id, kept & measured], positions[frame_id, kept & measured, :, None]
            )

            was_defined = is_defined[frame_id]

        # Smoothing: iterate backward (Update done for active t where t+1 is defined)
        for frame_id in tqdm.trange(positions.shape[0] - 2, -1, -1, desc=self.description_backward):
            kept = is_defined[frame_id + 1] & is_defined[frame_id]
            cov_at_process = states.covariance[frame_id, kept] @ kalman_filter.process_matrix.mT
            predicted_covariance = kalman_filter.process_matrix @ cov_at_process + kalman_filter.process_noise

            kalman_gain = cov_at_process @ predicted_covariance.inverse().mT
            states.mean[frame_id, kept] += kalman_gain @ (
                states.mean[frame_id + 1, kept] - kalman_filter.process_matrix @ states.mean[frame_id, kept]
            )
            states.covariance[frame_id, kept] += (
                kalman_gain @ (states.covariance[frame_id + 1, kept] - predicted_covariance) @ kalman_gain.mT
            )

        # Build new tracks from computed states
        new_tracks = []
        for i, track in enumerate(tracks):
            new_tracks.append(
                byotrack.Track(
                    track.start,
                    states.mean[track.start + offsets[i] - start : track.start + len(track) - start, i, :dim, 0],
                    track.identifier,
                    track.detection_ids[offsets[i] :],
                    merge_id=track.merge_id,
                    parent_id=track.parent_id,
                )
            )

        # Check produced tracks
        byotrack.Track.check_tracks(new_tracks, warn=True)

        return new_tracks
