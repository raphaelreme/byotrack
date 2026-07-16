"""SegmentationDetections: detections represented by a full instance-segmentation mask."""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Any

import numba  # type: ignore[import-untyped]
import numpy as np
import torch

import byotrack
from byotrack.api.detections.detections import (
    Detections,
    cached,
    draw_disk_2d,
    draw_disk_3d,
    labels_of,
    relabel_consecutive,
)
from byotrack.api.detections.detections import compress as compression
from byotrack.api.detections.detections import decompress as decompression
from byotrack.api.detections.point_detections import _expand_radius

if TYPE_CHECKING:
    from collections.abc import Callable

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


def _check_segmentation(segmentation: torch.Tensor) -> None:
    if len(segmentation.shape) not in (2, 3):
        raise ValueError("Segmentation tensor is expected to be 2D or 3D")
    if segmentation.numel() == 0:
        raise ValueError("Segmentation tensor is expected to have at least one pixel")


@numba.njit(cache=byotrack.NUMBA_CACHE)
def _compute_mass(segmentation: np.ndarray) -> np.ndarray:
    """Extract the number of pixels of each object in the segmentation.

    Assumes labels are consecutive from 1 to N.

    Args:
        segmentation (np.ndarray): Segmentation mask (consecutive labels 1..N)
            Shape: ([D, ]H, W), dtype: int

    Returns:
        np.ndarray: Mass for each object
            Shape: (N,), dtype: int32

    """
    n = segmentation.max()
    mass = np.zeros(n, dtype=np.int32)

    # Ravel in 1D
    segmentation = segmentation.reshape(-1)

    for i in range(segmentation.shape[0]):
        instance = segmentation[i]
        if instance != 0:
            mass[instance - 1] += 1

    return mass


@numba.njit(parallel=False, cache=byotrack.NUMBA_CACHE)
def _position_from_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Return the center (mean) of each instance in the segmentation.

    Assumes labels are consecutive from 1 to N.
    """
    n = segmentation.max()
    dim = len(segmentation.shape)

    m_0 = np.zeros(n, dtype=np.uint)
    m_1 = np.zeros((n, dim), dtype=np.uint)

    for index in np.ndindex(*segmentation.shape):
        instance = segmentation[index]
        if instance != 0:
            instance -= 1
            m_0[instance] += 1
            for i in range(dim):
                m_1[instance, i] += index[i]

    return m_1.astype(np.float32) / m_0.reshape(-1, 1)


@numba.njit(parallel=False, cache=byotrack.NUMBA_CACHE)
def _median_from_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Return the center (median) of each instance in the segmentation.

    Assumes labels are consecutive from 1 to N.
    """
    flat_segmentation = segmentation.reshape(-1)

    n = segmentation.max()
    median = np.zeros((n, len(segmentation.shape)), dtype=np.float32)
    counts = np.zeros(n, dtype=np.uint)

    if n == 0:
        return median

    for i in range(flat_segmentation.shape[0]):
        instance = flat_segmentation[i]
        if instance != 0:
            counts[instance - 1] += 1

    m = np.max(counts)

    # Reset counts and allocate position
    counts[:] = 0
    positions = np.empty((n, m, len(segmentation.shape)), dtype=np.uint)

    for index in np.ndindex(*segmentation.shape):
        instance = segmentation[index]
        if instance != 0:
            instance -= 1
            positions[instance, counts[instance]] = index
            counts[instance] += 1

    # Compute medians
    for instance in range(n):
        if counts[instance] == 0:
            median[instance] = np.nan
            continue

        for axis in range(len(segmentation.shape)):
            median[instance, axis] = np.median(positions[instance, : counts[instance], axis])

    return median


@numba.njit(cache=byotrack.NUMBA_CACHE)
def _bbox_from_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Compute bounding boxes for each instance in the segmentation.

    Assumes labels are consecutive from 1 to N.
    """
    # NOTE: this is a bit slower than previous version in 2D, but fine
    n = segmentation.max()
    dim = len(segmentation.shape)

    bbox = np.zeros((n, 2 * dim), dtype=np.int32)
    mini = np.full((n, dim), np.iinfo(np.int32).max, dtype=np.int32)
    maxi = np.zeros((n, dim), dtype=np.int32)

    for index in np.ndindex(*segmentation.shape):
        instance = segmentation[index]
        if instance != 0:
            instance -= 1
            for i in range(dim):
                mini[instance, i] = min(mini[instance, i], index[i])
                maxi[instance, i] = max(maxi[instance, i], index[i])

    # Keep 0 for undefined element
    defined = mini[:, 0] != np.iinfo(np.int32).max

    bbox[defined, :dim] = mini[defined]
    bbox[defined, dim:] = maxi[defined] - mini[defined] + 1

    return bbox


@numba.njit(parallel=True, cache=byotrack.NUMBA_CACHE)
def _filter_segmentation(segmentation: np.ndarray, to_delete: np.ndarray) -> None:
    """Filter the segmentation mask in place.

    Args:
        segmentation (np.ndarray): Instance segmentation mask (consecutive) to filter.
            Shape: ([D, ]H, W), dtype: int.
        to_delete (np.ndarray): Boolean array indicating which labels to remove.
            Shape: (N,), dtype: bool.
    """
    segmentation = segmentation.reshape(-1)
    for i in numba.prange(segmentation.size):
        instance = segmentation[i]
        if instance != 0 and to_delete[instance - 1]:
            segmentation[i] = 0


def _default_position_method(method: str | Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable position extractor from a string key or pass through callable."""
    if callable(method):
        return method
    if method == "mean":
        return _position_from_segmentation
    if method == "median":
        return _median_from_segmentation
    raise ValueError(f"Unknown position_method '{method}'. Use 'mean', 'median', or a callable.")


class SegmentationDetections(Detections):
    """Detections represented by a full instance-segmentation mask.

    The primary data is a 2-D or 3-D integer tensor of shape ``([D, ]H, W)``.
    Labels can be any non-negative integers (consecutive or not); 0 is always
    background. Non-consecutive labels are stored in :attr:`labels` so the
    original mapping is preserved; internally the mask is relabelled to the
    consecutive sequence 1, ..., N for efficient computation.

    All other properties (position, bbox, mass) are derived lazily.

    Attributes:
        segmentation (torch.Tensor): Instance segmentation mask. Labels are consecutive, from 1 to the
            length (N), where the detection i is labeled i+1. 0 is assigned to each background pixel.
            Shape: ([D, ]H, W), dtype: int32.
        labels (torch.Tensor): Labels of the detections. Defaults to the labels of the input segmentation mask so that
            ``labeled_segmentation`` remap the input segmentation. As 0 is the background, labels drawn
            on segmentation start at 1 (off-by-one).
            Shape: (N,), dtype: int32.

    """

    def __init__(
        self,
        segmentation: torch.Tensor,
        *,
        confidence: torch.Tensor | None = None,
        position_method: str | Callable = "median",
        cache: bool = True,
        compress: bool = byotrack.ZSTD_SEG,
    ) -> None:
        """Create SegmentationDetections.

        Args:
            segmentation (torch.Tensor): Instance segmentation mask. Labels can be any non-negative
                integers; 0 is background. Note that large integer may cause performance issues.
                Shape: ([D, ]H, W), dtype: int32.
            confidence (torch.Tensor | None): Per-detection confidence score. Defaults to all-ones.
                Shape: (N,), dtype: float32.
            position_method (str | Callable): How to extract centre positions from the mask.
                ``"mean"`` or ``"median"`` (default), or any callable with signature
                ``(segmentation: np.ndarray) -> np.ndarray`` returning float32 (N, dim).
            shape (tuple[int, ...] | None): Image shape ([D, ]H, W).
                Inferred from ``ceil(max(position) + radius) + 1`` if not given.
            cache (bool): Cache lazily-computed properties. Default: True.
            compress (bool): Compress the segmentation mask in memory using ZSTD.
                Defaults to the ``ZSTD_SEG`` environment variable value.

        """
        self._segmentation = segmentation.to(torch.int32)
        _check_segmentation(self._segmentation)
        if self._segmentation.min() < 0:
            raise ValueError("segmentation should be non-negative.")

        labels = labels_of(self._segmentation)

        if len(labels) > 0 and labels[-1] != len(labels) - 1:  # Non-consecutive labels
            relabel_consecutive(self._segmentation, inplace=True)

        self.length = len(labels)
        self.dim = segmentation.ndim
        self.shape = tuple(segmentation.shape)

        super().__init__(confidence=confidence, labels=labels, cache=cache, compress=compress)

        self._position_fn = _default_position_method(position_method)

        if self._compress:
            self._segmentation = compression(self._segmentation.reshape(-1))

    @property
    @cached
    @override
    def position(self) -> torch.Tensor:
        return torch.from_numpy(self._position_fn(self.segmentation.numpy()))

    @property
    @cached
    @override
    def bbox(self) -> torch.Tensor:
        return torch.from_numpy(_bbox_from_segmentation(self.segmentation.numpy()))

    @property
    @cached
    @override
    def segmentation(self) -> torch.Tensor:
        if self._compress:
            return decompression(self._segmentation).reshape(self.shape)

        return self._segmentation

    @property
    @cached
    @override
    def mass(self) -> torch.Tensor:
        return torch.from_numpy(_compute_mass(self.segmentation.numpy()))

    @override
    def add_disks(
        self,
        positions: torch.Tensor,
        radius: float | torch.Tensor = 2.0,
        *,
        labels: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        overwrite: bool = False,
    ) -> SegmentationDetections:
        length = positions.shape[0]
        radius = _expand_radius(radius, length, self.dim)

        if labels is None:
            start_label = self.labels.max().item() + 1 if self.length else 0
            labels = torch.arange(start_label, start_label + length, dtype=torch.int32)
        else:
            labels = labels.to(torch.int32)

        confidence = confidence.to(torch.float32) if confidence is not None else torch.ones(length, dtype=torch.float32)

        segmentation = self._segmentation
        segmentation = decompression(segmentation).reshape(self.shape) if self._compress else segmentation.clone()

        # Draw disks with temporary consecutive ids continuing after the existing ones.
        temp_ids = np.arange(self.length, self.length + length)
        draw_disk = draw_disk_3d if self.dim == 3 else draw_disk_2d  # noqa: PLR2004
        new_positions = positions.to(torch.float32).numpy()
        draw_disk(segmentation.numpy(), new_positions, radius.numpy(), temp_ids, overwrite=overwrite)

        detections = SegmentationDetections(
            segmentation,
            position_method=self._position_fn,
            cache=self._use_cache,
            compress=self._compress,
        )

        # Some disks may have failed to draw (out of bounds, or occluded), or have occluded previous labels:
        # `detections.labels` holds the surviving subset of labels, which can be used to filter out dropped labels
        detections._confidence = torch.cat([self.confidence, confidence])[detections.labels]
        detections._labels = torch.cat([self.labels, labels])[detections.labels]

        return detections

    @override
    def filter(self, kept: torch.Tensor) -> SegmentationDetections:
        segmentation = self._segmentation
        if segmentation.ndim == 1:
            segmentation = decompression(segmentation).reshape(self.shape)
        else:
            segmentation = segmentation.clone()

        # Filter inplace
        _filter_segmentation(segmentation.numpy(), (~kept).numpy())

        detections = SegmentationDetections(
            segmentation,
            confidence=self._confidence[kept] if self._confidence is not None else None,
            position_method=self._position_fn,
            cache=self._use_cache,
            compress=self._compress,
        )

        # Reset the labels
        if self._labels is not None:
            detections._labels = self._labels[kept]

        return detections

    def _to_dict(self) -> dict[str, object]:
        d = super()._to_dict()
        d["_type"] = "segmentation"

        d["segmentation"] = self._segmentation

        if self._compress:
            d["shape"] = self.shape

        if self._position_fn.__name__ == "_position_from_segmentation":
            d["position_method"] = "mean"
        elif self._position_fn.__name__ == "_median_from_segmentation":
            d["position_method"] = "median"
        else:
            warnings.warn(
                "Specific ``position_fn`` cannot be saved. Saving with default ``median`` mode.", stacklevel=2
            )
            d["position_method"] = "median"

        return d

    @staticmethod
    def _from_dict(
        data: dict[str, Any], *, cache: bool = True, compress: bool = byotrack.ZSTD_SEG
    ) -> SegmentationDetections:
        segmentation: torch.Tensor = data["segmentation"]
        labels = data.get("labels")
        confidence = data.get("confidence")
        position_method = data.get("position_method", "median")

        # Handle compressed segmentation
        if segmentation.ndim == 1:
            shape = data["shape"]  # Shape needs to be included if compressed
            segmentation = decompression(segmentation).reshape(shape)

        detections = SegmentationDetections(
            segmentation, confidence=confidence, position_method=position_method, cache=cache, compress=compress
        )

        # Restore labels if any
        detections._labels = labels

        return detections
