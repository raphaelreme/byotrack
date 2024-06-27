from typing import Tuple

import numba  # type: ignore
import numpy as np


@numba.njit()
def _fast_build_links(indices: np.ndarray, shape: Tuple[int, int]):
    """Extract the links from the sorted indices"""
    n = min(shape)
    i_valid = np.full((shape[0],), True)
    j_valid = np.full((shape[1],), True)

    links = []

    for i, j in indices:  # O(nm), but much faster has we stop before having seen most indices
        if i_valid[i] and j_valid[j]:
            i_valid[i] = False
            j_valid[j] = False
            links.append((i, j))

            if len(links) >= n:
                break

    return np.array(links)


def greedy_assignment_solver(dist: np.ndarray, eta: float = np.inf):
    """Solve assignement problem in a greedy way

    Iteratively select the minimum cost, then deleting its row/column.

    Stops when the cost matrix is empty (no more rows or columns) or if the selected
    cost is higher than `eta`

    Args:
        dist (np.ndarray): Distance matrix
                Shape: (N, M), dtype: float
        eta (float): Hard thresholding
            Default: inf (No thresholding)

    Returns:
        np.ndarray: Links (i, j)
            Shape: (L, 2), dtype: uint16
    """
    if min(dist.shape) == 0:
        return np.zeros((0, 2), dtype=np.uint16)

    # Create a sorted list of indices to investigate => O(nm log(nm))
    valid = dist <= eta
    indices = np.indices(dist.shape).transpose((1, 2, 0))[valid]
    indices = indices[np.argsort(dist[valid])]
    if indices.size == 0:  # No links can be done
        return np.zeros((0, 2), dtype=np.uint16)

    links = _fast_build_links(indices, dist.shape[:2])

    return np.array(links, dtype=np.uint16).reshape(-1, 2)  # Ensure uint16 and (L, 2)
