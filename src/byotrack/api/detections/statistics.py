"""Statistics utilities for sequences of detections.

Provides functions to estimate aggregate properties (mass, radius, nearest-neighbor distance,
anisotropy) over a sequence of per-frame :class:`byotrack.Detections` objects.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    import byotrack


def average_mass(detections_sequence: Sequence[byotrack.Detections]) -> float:
    """Return the average mass (pixel surface or volume) per detection over a sequence of frames.

    Args:
        detections_sequence (Sequence[byotrack.Detections]): Sequence of per-frame detections.

    Returns:
        float: Average number of pixels (2D) or voxels (3D) per detection. Returns 0.0 if the
            sequence is empty or contains no detections.

    """
    total_size = 0
    count = 0
    for detections in detections_sequence:
        if len(detections) <= 0:
            continue

        count += len(detections)
        total_size += int(detections.mass.sum().item())  # XXX: May overflow with large volumes

    return total_size / (count + (count == 0))


def average_radius(
    detections_sequence: Sequence[byotrack.Detections], anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """Return the average radius of detections over a sequence of frames.

    Assumes each detection is roughly spherical (3D) or circular (2D), and derives the
    radius from the average mass using the corresponding volume formula:

    - 2D: ``mass = π * R²``  =>  ``R = sqrt(mass / π)``
    - 3D: ``mass = 4/3 * π * R³``  =>  ``R = (3 * mass / (4 * π)) ** (1/3)``

    Args:
        detections_sequence (Sequence[byotrack.Detections]): Sequence of per-frame detections.
        anisotropy (tuple[float, float, float]): Anisotropy factors ``(ani_z, ani_y, ani_x)``
            used to scale the average mass before computing the radius. These factors convert
            voxel coordinates to isotropic ones.
            Default: (1.0, 1.0, 1.0) (no scaling).

    Returns:
        float: Average detection radius in isotropic coordinates. Returns 0.0 if the sequence
            is empty.

    """
    avg_mass = average_mass(detections_sequence)

    if len(detections_sequence) == 0:
        return 0.0

    avg_mass *= math.prod(anisotropy)  # Scale by anisotropy

    if detections_sequence[0].dim == 3:  # noqa: PLR2004
        return (avg_mass * 3 / 4 / math.pi) ** (1 / 3)  # area = 4/3 pi R^3

    return math.sqrt(avg_mass / math.pi)  # pi R^2


def average_min_dist(
    detections_sequence: Sequence[byotrack.Detections], anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """Return the average minimal distance between two detections in the same frame.

    For each frame with at least two detections, computes the per-detection minimum distance
    to its nearest neighbor, then takes the median across all detections in that frame (to
    reduce the influence of outliers). The result is averaged over all eligible frames.

    Args:
        detections_sequence (Sequence[byotrack.Detections]): Sequence of per-frame detections.
            Frames with fewer than two detections are ignored.
        anisotropy (tuple[float, float, float]): Anisotropy factors ``(ani_z, ani_y, ani_x)``
            used to scale detection positions before computing distances. Only the last ``dim``
            elements of the tuple are applied.
            Default: (1.0, 1.0, 1.0) (no scaling).

    Returns:
        float: Average (over frames) of the per-frame median nearest-neighbor distance.
            Returns 0.0 if no frame contains at least two detections.

    """
    anisotropy_pt = torch.tensor(anisotropy)
    sum_min_dist = 0.0
    count = 0
    for detections in detections_sequence:
        if len(detections) <= 1:
            continue

        position = detections.position * anisotropy_pt[-detections.dim :]
        dist = torch.cdist(position, position)
        dist[torch.arange(detections.length), torch.arange(detections.length)] = torch.inf
        sum_min_dist += dist.min(dim=1).values.median().item()  # Taking a median per frame to remove outliers
        count += 1

    return sum_min_dist / (count + (count == 0))


def anisotropy(detections_sequence: Sequence[byotrack.Detections], *, only_depth=True) -> tuple[float, float, float]:
    """Return the average anisotropy found in the detections based on their size.

    It makes the assumption that objects do not have a preferential direction and therefore their average size should
    be isotrope.

    The anisotropy is defined as the scaling factors (ani_z, ani_y, ani_x) to scale original coordinates
    to isotrope ones.

    This always takes the last dimension (X) as reference and therefore anisotropy = (ani_z, ani_y, 1).

    If `only_depth` is set to true, the two last dimensions (YX) are used as references and only the axis Z is
    anisotrope: anisotropy = (ani_z, 1, 1).
    """
    if len(detections_sequence) == 0:
        return (1.0, 1.0, 1.0)

    dim = detections_sequence[0].dim

    # Average detection size along each axis
    sizes = sum(
        (detections.bbox[:, detections.dim :].to(torch.float32).mean(dim=0) for detections in detections_sequence),
        start=torch.zeros(1),
    ) / len(detections_sequence)

    if (sizes == 0).any():
        return (1.0, 1.0, 1.0)

    anisotropy = tuple(sizes[-1] / s for s in sizes)

    if only_depth and max(anisotropy[1:]) / min(anisotropy[1:]) > 1.5:  # noqa: PLR2004
        warnings.warn("Computing anisotropy only for Z-axis, but X and Y axes seems to be anistrope.", stacklevel=2)

    if dim == 2:  # noqa: PLR2004
        if only_depth:
            return (1.0, 1.0, 1.0)
        return (1.0, *anisotropy)

    if only_depth:
        depth_anisotropy = float(sizes[1:].mean() / sizes[0])
        return (depth_anisotropy, 1.0, 1.0)

    return anisotropy
