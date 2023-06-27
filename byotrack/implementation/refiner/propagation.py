from typing import Callable, Optional, Union
import warnings

import torch
import torch_tps
import tqdm


def forward_backward_propagation(
    tracks_matrix: torch.Tensor, method: Union[str, Callable[..., torch.Tensor]], **kwargs
) -> torch.Tensor:
    """Fill all NaN values in the tracks matrix by merging a forward and backward propagation

    Esimate missing positions for any defined track and frame. First it computes forward estimations,
    then backward ones. If estimations overlap (for instance with missing positions inside a track), it will
    merge smoothly the estimations.

    We provide two implementations for directed propagation:
    Constant where we propagate the last known position of the track
    TPS where we use ThinPlateSpline using the active tracks to estimate the motion of other tracks

    Args:
        tracks_matrix (torch.Tensor): Tracks data in a single tensor (See `Track.tensorize`)
            Shape: (T, N, D), dtype: float32
        method (str | Callable[..., torch.Tensor]): Method to use for propagation. Either "tps", "constant"
            of a self defined function to be called.
        **kwargs (Any): Additional arguments given to the method

    Returns:
        torch.Tensor: Extrapolated point for each track and time
            Shape: (T, N, D), dtype: float32
    """
    directed_propagate: Callable[..., torch.Tensor]
    if isinstance(method, str):
        if method.lower() == "tps":
            directed_propagate = tps_directed_propagate
        elif method.lower() == "constant":
            directed_propagate = constant_directed_propagate
        else:
            raise ValueError(f"Unknown method {method}. We only provide two of them: constant and tps")
    else:
        directed_propagate = method

    forward_positions = directed_propagate(tracks_matrix, True, **kwargs)
    backward_positions = directed_propagate(tracks_matrix, False, **kwargs)

    return merge(tracks_matrix, forward_positions, backward_positions)


def merge(
    tracks_matrix: torch.Tensor, forward_positions: torch.Tensor, backward_positions: torch.Tensor
) -> torch.Tensor:
    """Merge forward and backward propagation into a single propagation matrix

    If both propagation defines a position estimation, a weighted average is performed.

    Args:
        tracks_matrix (torch.Tensor): Original tracks data (`Track.tensorize`)
            Shape: (T, N, D), dtype: float32
        forward_positions (torch.Tensor): Forward estimation of positions
            Shape: (T, N, D), dtype: float32
        backward_positions (torch.Tensor): backward estimation of positions
            Shape: (T, N, D), dtype: float32

    Returns:
        torch.tensor: Merged propagation matrix
            Shape: (T, N, D), dtype: float32
    """

    # 1- Take forward and complete by backward

    propagation_matrix = forward_positions.clone()
    propagation_matrix[torch.isnan(forward_positions)] = backward_positions[torch.isnan(forward_positions)]

    # 2- If both forward and backward, merge them by a weighted sum
    both = (
        ~torch.isnan(backward_positions).any(dim=-1)
        & ~torch.isnan(forward_positions).any(dim=-1)
        & torch.isnan(tracks_matrix).any(dim=-1)
    )
    if both.any():
        undefined = torch.isnan(tracks_matrix).any(dim=-1)
        forward_score = torch.zeros(undefined.shape, dtype=torch.int16)[..., None]
        backward_score = torch.zeros(undefined.shape, dtype=torch.int16)[..., None]

        for i in range(1, tracks_matrix.shape[0]):
            # Forward (resp. backward) computation for backward score (resp. forward)
            backward_score[i, undefined[i]] = backward_score[i - 1, undefined[i]] + 1

            i = tracks_matrix.shape[0] - i - 1
            forward_score[i, undefined[i]] = forward_score[i + 1, undefined[i]] + 1

        propagation_matrix[both] = (
            forward_score[both] * forward_positions[both] + backward_score[both] * backward_positions[both]
        ) / (forward_score[both] + backward_score[both])

    return propagation_matrix


def constant_directed_propagate(tracks_matrix: torch.Tensor, forward=True) -> torch.Tensor:
    """Propagate tracks matrix with the last known position in a single direction

    Args:
        tracks_matrix (torch.Tensor): Tracks data in a single tensor (See `Tracks.tensorize`)
            Shape: (T, N, D), dtype: float32
        forward (bool): Forward or backward propagation
            Default: True (Forward)

    Returns:
        torch.Tensor: Estimation of tracks point in a single direction
            Shape: (T, N, D), dtype: float32
    """
    tracks_matrix = tracks_matrix if forward else torch.flip(tracks_matrix, (0,))

    propagation_matrix = tracks_matrix.clone()  # (T, N, D)
    valid = ~torch.isnan(propagation_matrix).any(dim=-1)  # (T, N)

    for i in tqdm.trange(1, tracks_matrix.shape[0], desc=f"{'Forward' if forward else 'Backward'} propagation"):
        # Compute propagation mask (not valid and has a past)
        propagation_mask = ~valid[i] & ~torch.isnan(propagation_matrix[i - 1]).any(dim=-1)

        if propagation_mask.sum() == 0:
            continue  # No propagation to do

        # Propagate points
        propagation_matrix[i, propagation_mask] = propagation_matrix[i - 1, propagation_mask]

    return propagation_matrix if forward else torch.flip(propagation_matrix, (0,))


def tps_directed_propagate(
    tracks_matrix: torch.Tensor, forward=True, alpha=5.0, track_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Propagate tracks matrix using Thin Plate Spline (TPS) algorithm in a single direction

    .. note:: We use torch-tps which is very fast but not very accurate. For instance, shuflling the tracks
        may yield different results (<1px deviation). We have not noticed any real impact on EMC2 stitching
        performances. Alpha > 5.0 is advised (reduces the numerical errors + outliers resilience).


    Args:
        tracks_matrix (torch.Tensor): Tracks data in a single tensor (See `Tracks.tensorize`)
            Shape: (T, N, D), dtype: float32
        forward (bool): Forward or backward propagation
            Default: True (Forward)
        alpha (float): Regularization parameter of TPS
            Default: 5.0 (alpha > 5.0 is advised)
        track_mask (Optional[torch.Tensor]): Filter out some tracks to build control points
            Allow to drop uncertain tracks or to simulate propagation with less particles
            Shape: (N, ), dtype: bool

    Returns:
        torch.Tensor: Estimation of tracks point in a single direction
            Shape: (T, N, D), dtype: float32
    """
    tps = torch_tps.ThinPlateSpline(alpha, "cpu")  # Cf torch_tps: Faster on cpu than gpu

    tracks_matrix = tracks_matrix if forward else torch.flip(tracks_matrix, (0,))

    propagation_matrix = tracks_matrix.clone()  # (T, N, D)
    valid = ~torch.isnan(propagation_matrix).any(dim=-1)  # (T, N)

    for i in tqdm.trange(1, tracks_matrix.shape[0], desc=f"{'Forward' if forward else 'Backward'} propagation"):
        # Compute control mask (Valid before and now)
        control_mask = valid[i - 1] & valid[i]  # (N,)
        if track_mask is not None:
            control_mask &= ~track_mask  # Keep only unmasked tracks

        # Compute propagation mask (not valid and has a past)
        propagation_mask = ~valid[i] & ~torch.isnan(propagation_matrix[i - 1]).any(dim=-1)

        if propagation_mask.sum() == 0:
            continue  # No propagation to do

        if control_mask.sum() < 5:
            warnings.warn(f"Too few control points while propagating ({control_mask}.sum())")

        # Fit TPS on the control tracks from source and target frame
        tps.fit(tracks_matrix[i - 1, control_mask], tracks_matrix[i, control_mask])

        # Propagate points
        propagation_matrix[i, propagation_mask] = tps.transform(propagation_matrix[i - 1, propagation_mask])

    return propagation_matrix if forward else torch.flip(propagation_matrix, (0,))
