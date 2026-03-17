"""Helpers to handle Sinetra dataset [11].

We only provide functions to load annotations.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import torch

import byotrack

if TYPE_CHECKING:
    import os


def load_metadata(path: str | os.PathLike) -> dict[str, torch.Tensor]:
    """Load all the metadata of the ground-truths.

    It loads the outputted metadata from SINETRA (position, size, rotation and intensity weight)
    See: https://github.com/raphaelreme/SINETRA

    JSON Format:

    .. code-block:: python

        {
            "mu": torch.Tensor (T, N, D)  # Position
            "std": torch.Tensor (T, N, D)  # Size
            "theta": torch.Tensor (T, N, 1 or D)  # Rotation (1 in 2D, 3 in 3D)
            "weight": torch.Tensor (T, N)  # Weight
        }

    Args:
        path (str | os.PathLike): Path to the generated folder or to the `video_data.pt` file.

    Returns:
        dict[str, torch.Tensor]: Metadata (position, size, rotation, and intensity weight)
    """
    path = pathlib.Path(path)

    if path.is_dir():
        return torch.load(path / "video_data.pt", weights_only=True)

    return torch.load(path, weights_only=True)


def load_tracks(path: pathlib.Path) -> list[byotrack.Track]:
    """Load ground-truth tracks, built from the metadata.

    This is quite simple, it uses only the positional metadata ("mu") to build tracks.
    Each track is defined from frame 0 to the end in current implementation of SINETRA.

    Args:
        path (str | os.PathLike): Path to the generated folder or to the `video_data.pt` file.

    Returns:
        list[byotrack.Track]: Ground-truth tracks

    """
    ground_truth = load_metadata(path)

    return [byotrack.Track(0, ground_truth["mu"][:, i], i) for i in range(ground_truth["mu"].shape[1])]
