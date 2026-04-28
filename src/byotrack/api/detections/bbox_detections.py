"""BBoxDetections: detections represented as axis-aligned bounding boxes."""

from __future__ import annotations

import sys
from typing import Any

import numba  # type: ignore[import-untyped]
import numpy as np
import torch

import byotrack
from byotrack.api.detections.detections import Detections, cached

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


def _check_bbox(bbox: torch.Tensor) -> None:
    if len(bbox.shape) != 2:  # noqa: PLR2004
        raise ValueError("Bbox tensor is expected to be of shape (N, 2xdim)")
    if bbox.shape[1] not in (4, 6):
        raise ValueError("Bbox tensor should have 4 or 6 values: ([front, ]top, left, [depth, ]height, width)")


@numba.njit(cache=byotrack.NUMBA_CACHE)
def _segmentation_from_bbox_2d(bbox: np.ndarray, shape: tuple[int, int], labels: np.ndarray) -> np.ndarray:
    # Switch to start/stop bbox instead of start/size bbox
    bbox = bbox.copy()
    bbox[:, 2:] += bbox[:, :2]
    bbox.clip(0, out=bbox)  # Clip bbox to prevent negative indices

    segmentation = np.zeros(shape, dtype=np.int32)
    for label, bbox_ in enumerate(bbox):
        segmentation[bbox_[0] : bbox_[2], bbox_[1] : bbox_[3]] = labels[label] + 1

    return segmentation


@numba.njit(cache=byotrack.NUMBA_CACHE)
def _segmentation_from_bbox_3d(bbox: np.ndarray, shape: tuple[int, int, int], labels: np.ndarray) -> np.ndarray:
    # Switch to start/stop bbox instead of start/size bbox
    bbox = bbox.copy()
    bbox[:, 3:] += bbox[:, :3]
    bbox.clip(0, out=bbox)  # Clip bbox to prevent negative indices

    segmentation = np.zeros(shape, dtype=np.int32)
    for label, bbox_ in enumerate(bbox):
        segmentation[bbox_[0] : bbox_[3], bbox_[1] : bbox_[4], bbox_[2] : bbox_[5]] = labels[label] + 1

    return segmentation


class BBoxDetections(Detections):
    """Detections represented as axis-aligned bounding boxes.

    The primary data is a tensor of bounding boxes ``(N, 2*dim)`` in the format
    ``([front, ]top, left, [depth, ]height, width)`` — i.e. start coordinates
    followed by sizes in index coordinates.

    All other properties (position, segmentation, mass) are derived lazily.

    Note:
        Bounding boxes use integer index coordinates.
        All box sizes must be strictly positive; a zero-size box raises ``ValueError``
        at construction time.

    Attributes:
        bbox (torch.Tensor): Bounding boxes ``([k, ]i, j, [dk, ], di, dj)`` of each detection in index coordinates.
            Shape: (N, 2*dim), dtype: int32.

    """

    def __init__(
        self,
        bbox: torch.Tensor,
        *,
        confidence: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        shape: tuple[int, ...] | None = None,
        cache: bool = True,
        compress: bool = byotrack.ZSTD_SEG,
    ) -> None:
        """Create BBoxDetections.

        Args:
            bbox (torch.Tensor): Bounding boxes ``([k, ]i, j, [dk, ], di, dj)`` of each detection in index coordinates.
                Shape: (N, 2*dim), dtype: int32.
            confidence (torch.Tensor | None): Per-detection confidence score. Defaults to all-ones.
                Shape: (N,), dtype: float32.
            labels (torch.Tensor | None): Labels of the detections. Defaults to consecutive labels from 0 to N-1.
                If given, stored and used to draw ``labeled_segmentation``. As 0 is the background, labels drawn
                on segmentation start at 1 (off-by-one).
                Shape: (N,), dtype: int32.
            shape (tuple[int, ...] | None): Image shape ([D, ]H, W).
                Inferred from ``max(bbox_end)`` if not given.
            cache (bool): Cache lazily-computed properties. Default: True.
            compress (bool): Compress the segmentation mask in memory using ZSTD.
                Defaults to the ``ZSTD_SEG`` environment variable value.

        """
        self._bbox = bbox.to(torch.int32)

        _check_bbox(self._bbox)

        self.length = bbox.shape[0]
        self.dim = bbox.shape[1] // 2

        if bbox.numel() and bbox[:, self.dim :].min() <= 0:
            raise ValueError("bbox should only have positive sizes.")

        super().__init__(confidence=confidence, labels=labels, cache=cache, compress=compress)

        # Shape: infer from data if not given
        if shape is not None:
            if len(shape) != self.dim:
                raise ValueError(f"shape has {len(shape)} dimensions but position has {self.dim}")
            self.shape = shape
        else:
            self.shape = self._infer_shape()

    def _infer_shape(self) -> tuple[int, ...]:
        if self.length == 0:
            return (1, 1, 1) if self.dim == 3 else (1, 1)  # noqa: PLR2004

        ends = self._bbox[:, : self.dim] + self._bbox[:, self.dim :]
        return tuple(int(v) for v in ends.max(dim=0).values.tolist())

    @property
    @cached
    @override
    def position(self) -> torch.Tensor:
        return self._bbox[:, : self.dim] + (self._bbox[:, self.dim :] - 1) / 2

    @property
    @cached
    @override
    def bbox(self) -> torch.Tensor:
        return self._bbox

    @property
    @cached
    @override
    def segmentation(self) -> torch.Tensor:
        labels = np.arange(self.length)  # Let's use consecutive labels in ``segmentation``.

        if self.dim == 2:  # noqa: PLR2004
            segmentation = _segmentation_from_bbox_2d(self._bbox.numpy(), self.shape, labels)
        else:
            segmentation = _segmentation_from_bbox_3d(self._bbox.numpy(), self.shape, labels)

        return torch.from_numpy(segmentation)

    @property
    @cached
    @override
    def mass(self) -> torch.Tensor:
        return self._bbox[:, self.dim :].prod(dim=-1)

    @override
    def filter(self, kept: torch.Tensor) -> BBoxDetections:
        return BBoxDetections(
            self._bbox[kept],
            confidence=self._confidence[kept] if self._confidence is not None else None,
            labels=self._labels[kept] if self._labels is not None else None,
            shape=self.shape,
            cache=self._use_cache,
            compress=self._compress,
        )

    def _to_dict(self) -> dict[str, Any]:
        d = super()._to_dict()
        d["_type"] = "bbox"
        d["bbox"] = self._bbox

        if self.shape != self._infer_shape():
            d["shape"] = self.shape

        return d

    @staticmethod
    def _from_dict(data: dict[str, Any], *, cache: bool = True, compress: bool = byotrack.ZSTD_SEG) -> BBoxDetections:
        bbox = data["bbox"]
        confidence = data.get("confidence")
        labels = data.get("labels")
        shape = data.get("shape")
        return BBoxDetections(bbox, confidence=confidence, labels=labels, shape=shape, cache=cache, compress=compress)
