from typing import Optional
import warnings

import torch
import tqdm

import torch_tps


def propagate(tracks_matrix: torch.Tensor, alpha=0.0, track_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Propagate tracks matrix using Thin Plate Spline (TPS) algorithm.

    Esimate missing positions for any defined track and frame. First it computes forward estimations,
    then backward ones. If estimations overlap (for instance with missing positions inside a track), it will
    merge smoothly the estimations.

    Args:
        tracks_matrix (torch.Tensor): Tracks data in a single tensor (See `Tracks.tensorize`)
            Shape: (T, N, D), dtype: float32
        alpha (float): Regularization parameter of TPS
            Default: 0.0
        track_mask (Optional[torch.Tensor]): Filter out some tracks to build control points
            Allow to drop uncertain tracks or to simulate propagation with less particles
            Shape: (N, ), dtype: bool

    Returns:
        torch.Tensor: Extrapolated point for each track and time
            Shape: (T, N, D), dtype: float32
    """
    forward_positions = directed_propagate(tracks_matrix, alpha, True, track_mask)
    backward_positions = directed_propagate(tracks_matrix, alpha, False, track_mask)

    # Merge:
    # 1- Take forward and complete by backward
    # 2- If both, merge them by a weighted sum

    propagation_matrix = forward_positions
    propagation_matrix[torch.isnan(forward_positions)] = backward_positions[torch.isnan(forward_positions)]

    # Newly computed point by both forward and backward
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


def directed_propagate(
    tracks_matrix: torch.Tensor, alpha=0.0, forward=True, track_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Propagate tracks matrix using Thin Plate Spline (TPS) algorithm in a single direction

    .. note:: We use torch-tps which is very fast but not very accurate. For instance, shuflling the tracks
        may yield different results (<1px deviation). We have not noticed any real impact on EMC2 stitching
        performances. Alpha > 5.0 is advised (reduces the numerical errors + outliers resilience).


    Args:
        tracks_matrix (torch.Tensor): Tracks data in a single tensor (See `Tracks.tensorize`)
            Shape: (T, N, D), dtype: float32
        alpha (float): Regularization parameter of TPS
            Default: 0.0
        forward (bool): Forward or backward propagation
            Default: True (Forward)
        track_mask (Optional[torch.Tensor]): Filter out some tracks to build control points
            Allow to drop uncertain tracks or to simulate propagation with less particles
            Shape: (N, ), dtype: bool

    Returns:
        torch.Tensor: Extrapolated point for each track and time
            Shape: (T, N, D), dtype: float32
    """
    n_frames, _, _ = tracks_matrix.shape
    tps = torch_tps.ThinPlateSpline(alpha, "cpu")  # Cf torch_tps: Faster on cpu than gpu

    tracks_matrix = tracks_matrix if forward else torch.flip(tracks_matrix, (0,))

    propagation_matrix = tracks_matrix.clone()  # (T, N, D)
    valid = ~torch.isnan(propagation_matrix).any(dim=-1)  # (T, N)

    for i in tqdm.trange(1, n_frames, desc=f"{'Forward' if forward else 'Backward'} propagation"):
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
