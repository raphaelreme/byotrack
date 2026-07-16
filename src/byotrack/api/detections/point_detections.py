"""PointDetections: detections represented as center positions with optional spot radii."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch

import byotrack
from byotrack.api.detections.detections import Detections, cached, draw_disk_2d, draw_disk_3d

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


def _check_position(position: torch.Tensor) -> None:
    if len(position.shape) != 2:  # noqa: PLR2004
        raise ValueError("Position tensor is expected to be of shape (N, dim)")
    if position.shape[1] not in (2, 3):
        raise ValueError("Position tensor should have 2 or 3 values: ([k, ]i, j)")


def _expand_radius(radius: float | torch.Tensor, length: int, dim: int) -> torch.Tensor:
    """Expand a radius spec (scalar or tensor expandable to (length, dim)) to (length, dim)."""
    if isinstance(radius, (int, float)):
        radius = torch.full((length, dim), radius, dtype=torch.float32)
    else:
        radius = torch.expand_copy(radius.to(torch.float32), (length, dim))

    if radius.numel() and radius.min() < 0:
        raise ValueError("Radius should be non-negative.")

    return radius


class PointDetections(Detections):
    """Detections represented as center positions with an optional spot radius.

    The primary data is a tensor of center positions ``(N, dim)``.
    All other properties (bbox, segmentation, mass) are derived lazily from the
    positions and radius.

    Attributes:
        position (torch.Tensor): Position ``([k, ]i, j)`` of each detection in index coordinates.
            Shape: (N, dim), dtype: float32.
        radius (torch.Tensor): Per detection and axis spot radius for segmentation / bbox conversion.
            Shape: (N, dim), dtype: float32.

    """

    def __init__(
        self,
        position: torch.Tensor,
        *,
        radius: float | torch.Tensor = 2.0,
        confidence: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        shape: tuple[int, ...] | None = None,
        cache: bool = True,
        compress: bool = byotrack.ZSTD_SEG,
    ) -> None:
        """Create PointDetections.

        Args:
            position (torch.Tensor): Position ``([k, ]i, j)`` of each detection in index coordinates.
                Shape: (N, dim), dtype: float32.
            radius (float | torch.Tensor): Per detection and axis spot radius for segmentation / bbox conversion.
                Either a scalar applied to all detections, or a float32 tensor expandable to (N, dim).
                Default: 2.0.
            confidence (torch.Tensor | None): Per-detection confidence score. Defaults to all-ones.
                Shape: (N,), dtype: float32.
            labels (torch.Tensor | None): Labels of the detections. Defaults to consecutive labels from 0 to N-1.
                If given, stored and used to draw ``labeled_segmentation``. As 0 is the background, labels drawn
                on segmentation start at 1 (off-by-one).
                Shape: (N,), dtype: int32.
            shape (tuple[int, ...] | None): Image shape ([D, ]H, W).
                Inferred from ``ceil(max(position) + radius) + 1`` if not given.
            cache (bool): Cache lazily-computed properties. Default: True.
            compress (bool): Compress the segmentation mask in memory using ZSTD.
                Defaults to the ``ZSTD_SEG`` environment variable value.

        """
        self._position = position.to(torch.float32, copy=True)

        _check_position(self._position)
        if torch.isnan(self._position).any():
            raise ValueError("Found ill-defined NaN position.")

        self.length = position.shape[0]
        self.dim = position.shape[1]

        super().__init__(confidence=confidence, labels=labels, cache=cache, compress=compress)

        self._radius = _expand_radius(radius, self.length, self.dim)

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

        maxi = (self._position + self._radius).max(dim=0).values
        return tuple(int(v) for v in (maxi + 1).ceil().int().tolist())

    @property
    @cached
    @override
    def position(self) -> torch.Tensor:
        return self._position

    @property
    @cached
    @override
    def bbox(self) -> torch.Tensor:
        bbox = torch.zeros((self.length, 2 * self.dim), dtype=torch.int32)

        starts = (self._position - self._radius).floor().int()
        ends = (self._position + self._radius + 1).ceil().int()

        bbox[:, : self.dim] = starts
        bbox[:, self.dim :] = ends - starts

        return bbox

    @property
    @cached
    @override
    def segmentation(self) -> torch.Tensor:
        segmentation = np.zeros(self.shape, dtype=np.int32)
        labels = np.arange(self.length)  # Let's use consecutive labels in ``segmentation``.

        # NOTE: Let's round position before drawing => enforce at least one pixel per point.
        if self.dim == 2:  # noqa: PLR2004
            draw_disk_2d(segmentation, self._position.numpy().round(), self._radius.numpy(), labels, overwrite=True)
        else:
            draw_disk_3d(segmentation, self._position.numpy().round(), self._radius.numpy(), labels, overwrite=True)

        return torch.from_numpy(segmentation)

    @property
    @cached
    @override
    def mass(self) -> torch.Tensor:
        if self.dim == 2:  # noqa: PLR2004
            mass = (torch.pi * self._radius.prod(dim=-1)).round().to(torch.int32)
        else:
            mass = ((4 / 3) * torch.pi * self._radius.prod(dim=-1)).round().to(torch.int32)

        return mass.clip(min=1)

    @override
    def filter(self, kept: torch.Tensor) -> PointDetections:
        return PointDetections(
            self._position[kept],
            radius=self._radius[kept],
            confidence=self._confidence[kept] if self._confidence is not None else None,
            labels=self._labels[kept] if self._labels is not None else None,
            shape=self.shape,
            cache=self._use_cache,
            compress=self._compress,
        )

    @override
    def add_disks(
        self,
        positions: torch.Tensor,
        radius: float | torch.Tensor = 2.0,
        *,
        labels: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        overwrite: bool = False,
    ) -> PointDetections:
        length = positions.shape[0]
        radius = _expand_radius(radius, length, self.dim)

        if labels is None:
            start_label = self.labels.max().item() + 1 if self.length else 0
            labels = torch.arange(start_label, start_label + length, dtype=torch.int32)
        else:
            labels = labels.to(torch.int32)

        confidence = confidence.to(torch.float32) if confidence is not None else torch.ones(length, dtype=torch.float32)

        return PointDetections(
            torch.cat([self._position, positions.to(torch.float32)]),
            radius=torch.cat([self._radius, radius]),
            confidence=torch.cat([self.confidence, confidence]),
            labels=torch.cat([self.labels, labels]),
            shape=self.shape,
            cache=self._use_cache,
            compress=self._compress,
        )

    def _to_dict(self) -> dict[str, Any]:
        d = super()._to_dict()
        d["_type"] = "point"
        d["position"] = self._position

        if (self._radius[0, 0] == self._radius).all():
            d["radius"] = self._radius[0, 0].item()
        else:
            d["radius"] = self._radius

        if self.shape != self._infer_shape():
            d["shape"] = self.shape

        return d

    @staticmethod
    def _from_dict(data: dict[str, Any], *, cache: bool = True, compress: bool = byotrack.ZSTD_SEG) -> PointDetections:
        position = data["position"]
        radius = data.get("radius", 2.0)
        confidence = data.get("confidence")
        labels = data.get("labels")
        shape = data.get("shape")
        return PointDetections(
            position, radius=radius, confidence=confidence, labels=labels, shape=shape, cache=cache, compress=compress
        )
