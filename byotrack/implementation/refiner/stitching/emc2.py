from typing import Collection, Iterable

import numba  # type: ignore
import numpy as np

import byotrack

from . import dist_stitcher
from .. import propagation


@numba.njit(parallel=True)
def _fast_emc2_dist(propagation_matrix: np.ndarray, skip_mask: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Fast implementation of EMC2 distance

    Compute the mininum distance between each track propagation (only in the temporal gap between them)

    Args:
        propagation_matrix (np.ndarray): Tracks data with propagation
            Shape: (T, N, D), dtype: np.float32
        skip_mask (np.ndarray): Skip computation if True
            Shape: (N, N), dtype: bool
        ranges (np.ndarray): Range for each track (start, end)
            Shape: (N, 2), dtype: int
    """
    n_tracks = propagation_matrix.shape[1]
    emc_dist = np.full((n_tracks, n_tracks), np.inf, dtype=np.float32)

    for i in numba.prange(n_tracks):  # pylint: disable=not-an-iterable
        # Could set some verbosity (without parallel)
        # with numba.objmode():
        #     print(f"Computing emc2 dist {i+1} / {n_tracks}", flush=True, end="\r")
        for j in numba.prange(n_tracks):  # pylint: disable=not-an-iterable
            if skip_mask[i, j]:
                continue

            start = ranges[i, 1] - 1  # End of i (taken)
            end = ranges[j, 0]  # Start of j (taken)
            if end < start:
                start, end = end, start

            diff = propagation_matrix[start : end + 1, i] - propagation_matrix[start : end + 1, j]
            emc_dist[i, j] = (diff**2).sum(axis=-1).min()

    return np.sqrt(emc_dist)


class EMC2Stitcher(dist_stitcher.DistStitcher):
    """Implements Elastic Motion Correction and Concatenation (EMC2)

    Stitch tracks using motion corrected positions following the paper:
    T. Lagache, A. Hanson, J. Perez-Ortega, et al., “Tracking calcium dynamics from
    individual neurons in behaving animals”, PLoS computational biology, vol. 17, pp. e1009432, 10 2021.

    Attributes:
        alpha (float): Thin Plate Spline regularization (See `tps_propation` module and `torch_tps` librarie)
            We advise to use alpha > 5.0. It improves outliers resiliences and helps reducing numerical errors.
            Default: 10.0
        eta (float): Soft threshold in LAP solving (See `DistStitcher`)
            Default: 5.0 (Pixels)
        max_overlap (int): Cannot stitch tracks that overlap more than `max_overlap`
            Default: 5
        max_dist (float): Cannot stich track i and track j if the last position of i and
            first position of j are farther than `max_dist` (ignored if max_dist <= 0)
            Default: 100.0
        max_gap (int): Cannot stich track i and track j if i ended more
            than `max_gap` frame before j started (ignored if max_gap <= 0)
            Default: 250

    """

    def __init__(self, alpha: float = 10.0, eta: float = 5.0) -> None:
        super().__init__(self.compute_dist, eta)
        self.alpha = alpha
        self.max_overlap = 5
        self.max_dist = 100.0
        self.max_gap = 250

    def compute_dist(self, _: Iterable[np.ndarray], tracks: Collection[byotrack.Track]) -> np.ndarray:
        """Compute EMC2 distance between tracks

        Compute the mininum distance between each track propagation (only in the temporal gap between them).

        Args:
            video (Iterable[np.ndarray]): Unused, added for API purposes
            tracks (Collection[Track]): Input tracks

        Returns:
            np.ndarray: Distance between each track. (Contains np.inf)
                Shape: (N, N), dtype: float32
        """

        skip_mask = self.skip_computation(tracks, self.max_overlap, self.max_dist, self.max_gap)
        propagation_matrix = propagation.forward_backward_propagation(
            byotrack.Track.tensorize(tracks), "tps", alpha=self.alpha
        )
        ranges = np.array([(track.start, track.start + len(track)) for track in tracks])
        ranges -= ranges.min()  # The track tensor has an offset of ranges.min() (Usually 0)

        return _fast_emc2_dist(propagation_matrix.numpy(), skip_mask.numpy(), ranges)
