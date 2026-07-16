"""Detections API: abstract base class, low-level utilities, and auto-detect factory."""

from __future__ import annotations

import copy
import functools
import pathlib
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, overload

import numba  # type: ignore[import-untyped]
import numpy as np
import torch

import byotrack
from byotrack import utils

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Sequence

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd


def _check_confidence(confidence: torch.Tensor, length: int) -> None:
    if len(confidence.shape) != 1:
        raise ValueError("confidence is expected to be of shape (N,)")

    if confidence.shape[0] != length:
        raise ValueError(f"confidence length ({confidence.shape[0]}) do not match detection length ({length})")

    if confidence.numel() and confidence.min() < 0:
        raise ValueError("confidence should be non-negative.")


def _check_labels(labels: torch.Tensor, length: int) -> None:
    if len(labels.shape) != 1:
        raise ValueError("labels is expected to be of shape (N,)")

    if labels.shape[0] != length:
        raise ValueError(f"labels length ({labels.shape[0]}) do not match detection length ({length})")

    if labels.numel() and labels.min() < 0:
        raise ValueError("labels should be non-negative.")


@numba.njit(cache=byotrack.NUMBA_CACHE)
def draw_disk_2d(
    segmentation: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    labels: np.ndarray,
    *,
    overwrite=True,
) -> np.ndarray:
    """Draw disks on a 2D segmentation mask.

    Modify ``segmentation`` inplace.

    Args:
        segmentation (np.ndarray): 2D segmentation mask. Empty or pre-filled with other instances.
            Shape: (H, W), dtype: int
        positions (np.ndarray): Position of the disks to draw.
            Shape: (N, 2), dtype: float
        radii (np.ndarray): Radii of the disks to draw.
            Shape: (N, 2), dtype: float
        labels (np.ndarray): Instance labels for each disk. As usual, an offset of 1 is applied,
            i.e. the ith disk is drawn with labels[i] + 1.
            Shape: (N,), dtype: int
        overwrite (bool): Allow disk to overwrite the pre-filled segmentation.
            Default: True

    Returns:
        np.ndarray: The modified segmentation.

    """
    shape = segmentation.shape
    best_dist = np.full(shape, np.inf, dtype=np.float32)

    for label in range(positions.shape[0]):
        pos = positions[label]
        r = radii[label]

        # Compute the disk bbox
        i_min = max(0, int(pos[0] - r[0]))
        i_max = min(shape[0], int(pos[0] + r[0]) + 2)
        j_min = max(0, int(pos[1] - r[1]))
        j_max = min(shape[1], int(pos[1] + r[1]) + 2)

        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                if not overwrite and segmentation[i, j] != 0 and best_dist[i, j] == np.inf:
                    continue

                dist = ((i - pos[0]) / r[0]) ** 2 + ((j - pos[1]) / r[1]) ** 2
                if dist < best_dist[i, j] and dist <= 1:
                    segmentation[i, j] = labels[label] + 1
                    best_dist[i, j] = dist

    return segmentation


@numba.njit(cache=byotrack.NUMBA_CACHE)
def draw_disk_3d(
    segmentation: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    labels: np.ndarray,
    *,
    overwrite=True,
) -> np.ndarray:
    """Draw disks on a 3D segmentation mask.

    Modify ``segmentation`` inplace.

    Args:
        segmentation (np.ndarray): 3D segmentation mask. Empty or pre-filled with other instances.
            Shape: (D, H, W), dtype: int
        positions (np.ndarray): Position of the disks to draw.
            Shape: (N, 3), dtype: float
        radii (np.ndarray): Radii of the disks to draw.
            Shape: (N, 3), dtype: float
        labels (np.ndarray): Instance labels for each disk. As usual, an offset of 1 is applied,
            i.e. the ith disk is drawn with labels[i] + 1.
            Shape: (N,), dtype: int
        overwrite (bool): Allow disk to overwrite the pre-filled segmentation.
            Default: True

    Returns:
        np.ndarray: The modified segmentation.
    """
    shape = segmentation.shape
    best_dist = np.full(shape, np.inf, dtype=np.float32)

    for label in range(positions.shape[0]):
        pos = positions[label]
        r = radii[label]

        # Compute the disk bbox
        k_min = max(0, int(pos[0] - r[0]))
        k_max = min(shape[0], int(pos[0] + r[0]) + 2)
        i_min = max(0, int(pos[1] - r[1]))
        i_max = min(shape[1], int(pos[1] + r[1]) + 2)
        j_min = max(0, int(pos[2] - r[2]))
        j_max = min(shape[2], int(pos[2] + r[2]) + 2)

        for k in range(k_min, k_max):
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    if not overwrite and segmentation[k, i, j] != 0 and best_dist[k, i, j] == np.inf:
                        continue

                    dist = ((k - pos[0]) / r[0]) ** 2 + ((i - pos[1]) / r[1]) ** 2 + ((j - pos[2]) / r[2]) ** 2
                    if dist < best_dist[k, i, j] and dist <= 1.0:
                        segmentation[k, i, j] = labels[label] + 1
                        best_dist[k, i, j] = dist

    return segmentation


@numba.njit(parallel=True, cache=byotrack.NUMBA_CACHE)
def fast_relabel(segmentation: np.ndarray, labels: np.ndarray) -> None:
    """Inplace fast relabel with given mapping.

    It keeps the background as is (0 => 0) and assumes labeling from 0 to N-1 (offset of 1 in segmentation).
    In practice, it maps instance i to mapping[i-1] + 1.

    Args:
        segmentation (np.ndarray): Instance segmentation frame.
            Shape: ([D, ]H, W), dtype: int
        labels (np.ndarray): Mapping to the targeted labels.
            Shape: (N,), dtype: int
    """
    segmentation = segmentation.reshape(-1)

    for i in numba.prange(segmentation.size):
        if segmentation[i]:
            segmentation[i] = labels[segmentation[i] - 1] + 1


@numba.njit(parallel=True, cache=byotrack.NUMBA_CACHE)
def _fast_unique(segmentation: np.ndarray) -> np.ndarray:
    """Fast np.unique for segmentation masks.

    About 30-60x faster than np.unique/torch.unique. Assumes seg.max() is small
    relative to the number of pixels, which is always true in practice.
    """
    segmentation = segmentation.reshape(-1)
    unique = np.zeros(segmentation.max() + 1, dtype=np.bool_)

    for i in numba.prange(segmentation.size):
        unique[segmentation[i]] = True

    return np.arange(segmentation.max() + 1, dtype=segmentation.dtype)[unique]


@overload
def labels_of(segmentation: torch.Tensor) -> torch.Tensor: ...


@overload
def labels_of(segmentation: np.ndarray) -> np.ndarray: ...


def labels_of(segmentation: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Extract the sorted labels of an instance segmentation mask.

    The background is excluded, and labels starts at 0 (offset of 1 with the segmentation mask
    where the background is 0 and the first label is 1).

    Args:
        segmentation (torch.Tensor | np.ndarray): Segmentation mask
            Shape: ([D, ]H, W), dtype: int

    Returns:
        torch.Tensor | np.ndarray: Sorted labels inside the segmentation mask.
            Shape: (N,), dtype: int

    """
    labels = _fast_unique(segmentation.numpy() if isinstance(segmentation, torch.Tensor) else segmentation)

    # Remove background if any
    if labels[0] == 0:
        labels = labels[1:]

    # Offset of 1
    labels -= 1

    if isinstance(segmentation, torch.Tensor):
        return torch.from_numpy(labels)

    return labels


@numba.njit(parallel=True, cache=byotrack.NUMBA_CACHE)
def _fast_relabel_consecutive(segmentation: np.ndarray) -> None:
    """Inplace relabel to consecutive labels."""
    segmentation = segmentation.reshape(-1)
    unique = np.zeros(segmentation.max() + 1, dtype=segmentation.dtype)

    for i in numba.prange(segmentation.size):
        unique[segmentation[i]] = 1

    unique[unique > 0] = np.arange(unique.sum(), dtype=segmentation.dtype)

    for i in numba.prange(segmentation.size):
        if segmentation[i]:
            segmentation[i] = unique[segmentation[i]]


@overload
def relabel_consecutive(segmentation: torch.Tensor, *, inplace: bool = ...) -> torch.Tensor: ...


@overload
def relabel_consecutive(segmentation: np.ndarray, *, inplace: bool = ...) -> np.ndarray: ...


def relabel_consecutive(segmentation: torch.Tensor | np.ndarray, *, inplace: bool = True) -> torch.Tensor | np.ndarray:
    """Relabel a segmentation mask so that labels are consecutive.

    For N instances, labels are 0 for background and [1:N] for each instance.

    Args:
        segmentation (torch.Tensor | np.ndarray): Segmentation mask
            Shape: ([D, ]H, W), dtype: int
        inplace (bool): Modify in place. Default: True

    Returns:
        torch.Tensor | np.ndarray: Relabeled segmentation (same object if inplace=True)

    """
    seg_np: np.ndarray
    if isinstance(segmentation, torch.Tensor):
        if not inplace:
            segmentation = segmentation.clone()
        seg_np = segmentation.numpy()
    else:
        if not inplace:
            segmentation = segmentation.copy()

        seg_np = segmentation

    _fast_relabel_consecutive(seg_np)

    return segmentation


# XXX: PyTorch allocation is quite bad with small tensors (that can result from zipping for instance).
#      Small tensors are typically allocated from cached segmentations, preventing freeing
#      the memory required to store the segmentations. It is quite unclear yet how this works.
#      Our current weird fix is to allocate the segmentation mask on numpy and then using from_numpy
#      which seems to avoid this issue. One could also use fixes like glibc.malloc_trim.
#      See https://github.com/pytorch/pytorch/issues/165319


def compress(tensor: torch.Tensor, level: int = 3) -> torch.Tensor:
    """Compress a tensor using zstd."""
    compressed = zstd.compress(tensor.numpy().tobytes(), level=level)
    return torch.frombuffer(bytearray(compressed), dtype=torch.uint8)


def decompress(tensor: torch.Tensor, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """Decompress a zstd-compressed tensor."""
    decompressed = zstd.decompress(tensor.numpy().tobytes())
    return torch.frombuffer(bytearray(decompressed), dtype=dtype)


_D = TypeVar("_D", bound="Detections")


def cached(fn: Callable[[_D], torch.Tensor]) -> Callable[[_D], torch.Tensor]:
    """Enable detections properties caching.

    Reads and writes ``self._cache`` (a ``dict[str, torch.Tensor]``) using the wrapped method's ``__name__``
    as the key.  Caching is skipped when ``self._use_cache`` is ``False`` or if ``self`` has a backing
    attribute ``f'_{fn.__name__}'`` (i.e. the property is simply derived from a dedicated storage).

    Note that ``segmentation`` is compressed before caching if ``self._compress`` is set.

    Args:
        fn: Method to wrap.  Its ``__name__`` is used as the cache key.

    Returns:
        Wrapped callable that checks / populates ``self._cache``.

    """
    name = fn.__name__

    @functools.wraps(fn)
    def wrapper(self: _D) -> torch.Tensor:
        result = self._cache.get(name)

        if result is None:
            result = fn(self)

            if self._use_cache and getattr(self, f"_{name}", None) is None:
                if name == "segmentation" and self._compress:
                    self._cache[name] = compress(result.reshape(-1))
                else:
                    self._cache[name] = result

        elif name == "segmentation" and self._compress:
            result = decompress(result).reshape(self.shape)

        return result

    # Preserve @abstractmethod so that abstract base definitions remain abstract.
    if getattr(fn, "__isabstractmethod__", False):
        wrapper.__isabstractmethod__ = True  # type: ignore[attr-defined]

    return wrapper


# Add something about caching
class Detections(ABC):
    """Abstract base class for frame-level detections.

    Represents the set of detected objects in a single video frame.
    Three concrete implementations exist:

    * :class:`PointDetections`: center positions + optional spot radius
    * :class:`BBoxDetections`: axis-aligned bounding boxes
    * :class:`SegmentationDetections`: full instance-segmentation mask

    All implementations expose the same properties: ``position``, ``bbox``,
    ``segmentation``, ``confidence``, ``length``, ``dim``, ``shape``, ``mass``.

    The easiest way to create a :class:`Detections` object from raw array-like data
    is via :func:`as_detections`, which automatically selects the right subclass based
    on the shape and dtype of the input.

    Note:
        All positions/bounding boxes use the index coordinate system (not xyz).
        In ByoTrack the depth axis (k) comes before height (i, row) and width (j, column):
        ``(k, i, j)`` in 3-D, ``(i, j)`` in 2-D.

    Note:
        Properties are derived lazily on first access and cached (controlled by the ``cache``
        constructor argument).  For example, :class:`PointDetections` caches ``bbox`` and ``segmentation``
        the first time they are requested; subsequent accesses return the cached tensor without recomputation.

    Note:
        The ``segmentation`` tensor can additionally be stored in compressed form (ZSTD) to
        reduce memory usage, controlled by the ``compress`` constructor argument
        (defaults to the ``ZSTD_SEG`` environment variable).

    Attributes:
        length (int): Number of detections (N).
        dim (int): Spatial dimension, i.e. 2 or 3.
        shape (tuple[int, ...]): Image shape ([D, ]H, W).
        position (torch.Tensor): Center positions.
            Shape: (N, dim), dtype: float32.
        bbox (torch.Tensor): Bounding boxes ([front, ]top, left, [depth, ]height, width),
            Shape: (N, 2*dim), dtype: int32.
        segmentation (torch.Tensor): Instance segmentation mask. Labels are consecutive, from 1 to the
            length (N), where the detection i is labeled i+1. 0 is assigned to each background pixel.
            Shape: ([D, ]H, W), dtype: int32.
        confidence (torch.Tensor): Per-detection confidence score. Defaults to all-ones.
            Shape: (N,), dtype: float32.
        labels (torch.Tensor): Labels of the detections. Defaults to consecutive labels from 0 to N-1.
            If given, stored and used to draw ``labeled_segmentation``. As 0 is the background, labels drawn
            on segmentation start at 1 (off-by-one).
            Shape: (N,), dtype: int32.
        mass (torch.Tensor): Per-detection pixel count (or approximation).
            Shape: (N,), dtype: int32.
        labeled_segmentation (torch.Tensor): Segmentation mask using the ``labels``. As for the ``segmentation``,
            detection i is labeled with labels[i] + 1. For :class:`SegmentationDetections`, it provides a mapping to
            the original segmentation map which has been relabeled consecutively internally.
            Shape: ([D, ]H, W), dtype: int32.
        metadata (dict[str, torch.Tensor]): Arbitrary per-detection tensors stored by external
            components (e.g. :class:`byotrack.FeaturesExtractor`). Not persisted by :meth:`save`.

    """

    length: int
    dim: int
    shape: tuple[int, ...]

    def __init__(
        self,
        *,
        confidence: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        cache: bool = True,
        compress: bool = byotrack.ZSTD_SEG,
    ) -> None:
        """Create a Detections.

        Args:
            confidence (torch.Tensor | None): Per-detection confidence score. Defaults to all-ones.
                Shape: (N,), dtype: float32.
            labels (torch.Tensor | None): Labels of the detections. Defaults to consecutive labels from 0 to N-1.
                If given, stored and used to draw ``labeled_segmentation``. As 0 is the background, labels drawn
                on segmentation start at 1 (off-by-one).
                Shape: (N,), dtype: int32.
            cache (bool): Cache lazily-computed properties. Default: True.
            compress (bool): Compress the segmentation mask in memory using ZSTD.
                Defaults to the ``ZSTD_SEG`` environment variable value.

        """
        self._confidence = None if confidence is None else confidence.to(torch.float32, copy=True)
        self._labels = None if labels is None else labels.to(torch.int32, copy=True)

        if self._confidence is not None:
            _check_confidence(self._confidence, self.length)

        if self._labels is not None:
            _check_labels(self._labels, self.length)

        self._use_cache = cache
        self._compress = compress

        self._cache: dict[str, torch.Tensor] = {}
        self.metadata: dict[str, torch.Tensor] = {}

    @property
    @cached
    @abstractmethod
    def position(self) -> torch.Tensor: ...  # noqa: D102

    @property
    @cached
    @abstractmethod
    def bbox(self) -> torch.Tensor: ...  # noqa: D102

    @property
    @cached
    @abstractmethod
    def segmentation(self) -> torch.Tensor: ...  # noqa: D102

    @property
    @cached
    @abstractmethod
    def mass(self) -> torch.Tensor: ...  # noqa: D102

    @property
    def confidence(self) -> torch.Tensor:  # noqa: D102
        if self._confidence is None:
            return torch.ones(self.length, dtype=torch.float32)

        return self._confidence

    @property
    def labels(self) -> torch.Tensor:  # noqa: D102
        if self._labels is None:
            return torch.arange(self.length, dtype=torch.int32)

        return self._labels

    @property  # Let's not cache this property
    def labeled_segmentation(self) -> torch.Tensor:  # noqa: D102
        if not torch.equal(self.labels, torch.arange(self.length)):
            segmentation = self.segmentation.clone()
            fast_relabel(segmentation.numpy(), self.labels.numpy())
            return segmentation

        return self.segmentation

    @abstractmethod
    def filter(self, kept: torch.Tensor) -> byotrack.Detections:
        """Filter the detections based on a boolean tensor.

        Args:
            kept (torch.Tensor): Detection to keep.
                Shape: (N,), dtype: bool

        Returns:
            byotrack.Detections: Filtered detections.
        """

    def relabel(self, labels: torch.Tensor) -> byotrack.Detections:
        """Return a copy of the detections with new labels.

        Args:
            labels (torch.Tensor): New labels for each detection.
                Shape: (N,), dtype: int32

        Returns:
            byotrack.Detections: A shallow copy of self with the new labels.
        """
        labels = labels.to(torch.int32, copy=True)
        _check_labels(labels, self.length)

        # Let's use a shallow copy, with a copy of the _cache and the metadata
        clone = copy.copy(self)
        clone._cache = dict(self._cache)  # noqa: SLF001
        clone.metadata = dict(self.metadata)
        clone._labels = labels  # noqa: SLF001
        return clone

    @abstractmethod
    def add_disks(
        self,
        positions: torch.Tensor,
        radius: float | torch.Tensor = 2.0,
        *,
        labels: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        overwrite: bool = False,
    ) -> byotrack.Detections:
        """Return a copy of the detections with additional disk-shaped detections.

        Useful to materialize a detection at a position where none was found (e.g. a track with
        a known position but no matching detection).

        Args:
            positions (torch.Tensor): Center positions of the disks to add.
                Shape: (M, dim), dtype: float32
            radius (float | torch.Tensor): Per-disk and axis radius. Either a scalar applied to all disks,
                or a float32 tensor expandable to (M, dim). Follows the same convention as
                :attr:`PointDetections.radius`.
                Default: 2.0
            labels (torch.Tensor | None): Labels for the new disks. Defaults to consecutive labels
                continuing after the last existing label.
                Shape: (M,), dtype: int32
            confidence (torch.Tensor | None): Confidence for the new disks. Defaults to ones.
                Shape: (M,), dtype: float32
            overwrite (bool): Only relevant for :class:`SegmentationDetections`: allow disks to overwrite
                pre-existing detection pixels. Ignored by :class:`PointDetections` and :class:`BBoxDetections`.
                Default: False

        Returns:
            byotrack.Detections: A new Detections with the disks added. Note that some disks may not appear
                in the result (e.g. fully out of frame, or fully occluded with ``overwrite=False`` for
                :class:`SegmentationDetections`).
        """

    def __len__(self) -> int:  # noqa: D105
        return self.length

    def _to_dict(self) -> dict[str, Any]:
        """Serialize primary data to a dict.

        Child class should overwrite this method and include a ``"type"`` key.
        """
        d: dict[str, Any] = {}
        if self._confidence is not None:
            d["confidence"] = self._confidence

        if self._labels is not None:
            d["labels"] = self._labels

        return d

    def save(self, path: str | os.PathLike) -> None:
        """Save detections to a file using ``torch.save``.

        Args:
            path (str | os.PathLike): Output path (expected ``.pt`` extension).

        """
        torch.save(self._to_dict(), path)

    @staticmethod
    def load(path: str | os.PathLike, *, cache: bool = True, compress: bool = byotrack.ZSTD_SEG) -> Detections:
        """Load detections from a file written by :meth:`save`.

        Dispatches to the appropriate subclass based on the ``"_type"`` key.

        Args:
            path (str | os.PathLike): Input path.
            cache (bool): Cache lazily-computed properties. Default: True.
            compress (bool): Compress the segmentation mask in memory using ZSTD.
                Defaults to the ``ZSTD_SEG`` environment variable value.

        Returns:
            Detections: The loaded detections object.

        """
        # Import here to avoid circular imports
        from byotrack.api.detections.bbox_detections import BBoxDetections  # noqa: PLC0415
        from byotrack.api.detections.point_detections import PointDetections  # noqa: PLC0415
        from byotrack.api.detections.segmentation_detections import SegmentationDetections  # noqa: PLC0415

        data: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        detection_type: str = data.pop("_type")

        dispatch: dict[str, Callable[..., Detections]] = {
            "point": PointDetections._from_dict,  # noqa: SLF001
            "bbox": BBoxDetections._from_dict,  # noqa: SLF001
            "segmentation": SegmentationDetections._from_dict,  # noqa: SLF001
        }

        if detection_type not in dispatch:
            raise ValueError(f"Unknown Detections type '{detection_type}' in saved file.")

        return dispatch[detection_type](data, cache=cache, compress=compress)

    @staticmethod
    def save_multi_frames_detections(detections_sequence: Sequence[Detections], path: str | os.PathLike) -> None:
        """Save a sequence of per-frame detections as ``{path}/0.pt``, ``1.pt``, ...

        Args:
            detections_sequence (Sequence[Detections]): Detections for each frame.
            path (str | os.PathLike): Output folder (created if absent).

        """
        path = pathlib.Path(path)
        path.mkdir(parents=True)

        for i, detections in enumerate(detections_sequence):
            detections.save(path / f"{i}.pt")

    @staticmethod
    def load_multi_frames_detections(path: str | os.PathLike) -> list[Detections]:
        """Load a sequence of per-frame detections from a folder.

        Expects files named ``0.pt``, ``1.pt``, ..., ``N.pt`` in *path*.

        Args:
            path (str | os.PathLike): Input folder.

        Returns:
            list[Detections]: Detections for each frame (ordered by index).

        """
        path = pathlib.Path(path)
        files = utils.sorted_alphanumeric(path.iterdir())
        detections_sequence: list[Detections] = []
        for i, file in enumerate(f for f in files if f.suffix == ".pt"):
            if file.stem != f"{i}":
                raise KeyError(f"The {i}th file is not '{i}.pt'")
            detections_sequence.append(Detections.load(file))
        return detections_sequence


DetectionsLike: TypeAlias = Detections | np.ndarray | torch.Tensor


def as_detections(data: DetectionsLike, **kwargs: Any) -> Detections:
    """Convert array-like data to the appropriate :class:`Detections` subclass.

    Heuristic rules:

    1. :class:`Detections` instance -> returned unchanged.
    2. 2-D floating tensor/array of shape ``(N, 2)`` or ``(N, 3)`` -> :class:`PointDetections`.
    3. 2-D integer tensor/array of shape ``(N, 4)`` or ``(N, 6)`` -> :class:`BBoxDetections`.
    4. 2-D or 3-D integer tensor/array -> :class:`SegmentationDetections`.

    Args:
        data (DetectionsLike): Input data.
        **kwargs: Forwarded to the chosen subclass constructor.

    Returns:
        Detections: Wrapped detections object.

    Raises:
        ValueError: If the format cannot be determined.

    """
    from byotrack.api.detections.bbox_detections import BBoxDetections  # noqa: PLC0415
    from byotrack.api.detections.point_detections import PointDetections  # noqa: PLC0415
    from byotrack.api.detections.segmentation_detections import SegmentationDetections  # noqa: PLC0415

    if isinstance(data, Detections):
        return data

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    if not isinstance(data, torch.Tensor):
        # Let's try to convert it to np and then pytorch
        data = torch.from_numpy(np.asarray(data))

    # 2-D float (N, 2|3) -> positions
    if data.ndim == 2 and data.is_floating_point() and data.shape[1] in (2, 3):  # noqa: PLR2004
        return PointDetections(data.to(torch.float32), **kwargs)

    # 2-D int (N, 4|6) -> bboxes (checked before general segmentation to avoid ambiguity)
    if data.ndim == 2 and not data.is_floating_point() and data.shape[1] in (4, 6):  # noqa: PLR2004
        return BBoxDetections(data.to(torch.int32), **kwargs)

    # 2-D or 3-D integer -> segmentation mask
    if data.ndim in (2, 3) and not data.is_floating_point():
        return SegmentationDetections(data.to(torch.int32), **kwargs)

    raise ValueError(
        f"Cannot automatically determine Detections format from tensor of shape {tuple(data.shape)} "
        f"and dtype {data.dtype}. Construct PointDetections, BBoxDetections or SegmentationDetections explicitly."
    )
