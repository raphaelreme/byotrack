from __future__ import annotations

import pathlib
import sys
from typing import TYPE_CHECKING, cast, overload

import numba  # type: ignore[import-untyped]
import numpy as np
import torch

import byotrack
from byotrack import utils

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd


def _check_segmentation(segmentation: torch.Tensor) -> None:
    """Check segmentation validity (type, shape, labels)."""
    if len(segmentation.shape) not in (2, 3):
        raise ValueError("Segmentation tensor is expected to be 2D or 3D")
    if segmentation.dtype is not torch.int32:
        raise ValueError("Segmentation tensor is expected to have a torch.int32 dtype")


def _check_bbox(bbox: torch.Tensor) -> None:
    """Check box validity (type, shape)."""
    if len(bbox.shape) != 2:  # noqa: PLR2004
        raise ValueError("Bbox tensor is expected to be of shape (N, 2xdim)")
    if bbox.shape[1] not in (4, 6):
        raise ValueError("Bbox tensor should have 4 or 6 values: ([front, ]top, left, [depth, ]height, width)")
    if bbox.dtype is not torch.int32:
        raise ValueError("Bbox tensor is expected to have a torch.int32 dtype")


def _check_position(position: torch.Tensor) -> None:
    """Check position validity (type, shape)."""
    if len(position.shape) != 2:  # noqa: PLR2004
        raise ValueError("Position tensor is expected to be of shape (N, dim)")
    if position.shape[1] not in (2, 3):
        raise ValueError("Position tensor should have 2 or 3 values: ([k, ]i, j)")
    if position.dtype is not torch.float32:
        raise ValueError("Position tensor is expected to have a torch.float32 dtype")


def _check_confidence(confidence: torch.Tensor) -> None:
    """Check confidence validity (type, shape)."""
    if len(confidence.shape) != 1:
        raise ValueError("Confidence tensor is expected to be of shape (N,)")
    if confidence.dtype is not torch.float32:
        raise ValueError("Confidence tensor is expected to have a torch.float32 dtype")


@numba.njit(cache=byotrack.NUMBA_CACHE)
def _compute_mass(segmentation: np.ndarray) -> np.ndarray:
    """Extract the number of pixels of each object in the segmentation.

    Args:
        segmentation (np.ndarray): Segmentation mask

    Returns:
        np.ndarray: Mass for each object
            Shape (n,), dtype: int32

    """
    n = segmentation.max()
    mass = np.zeros(n, dtype=np.int32)

    # Ravel in 1D
    segmentation = segmentation.reshape(-1)

    for i in range(segmentation.shape[0]):
        instance = segmentation[i] - 1
        if instance != -1:
            mass[instance] += 1

    return mass


@numba.njit(parallel=False, cache=byotrack.NUMBA_CACHE)
def _position_from_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Return the center (mean) of each instance in the segmentation."""
    # A bit slower than previous version in 2D, but still fine

    n = segmentation.max()
    dim = len(segmentation.shape)

    m_0 = np.zeros(n, dtype=np.uint)
    m_1 = np.zeros((n, dim), dtype=np.uint)

    for index in np.ndindex(*segmentation.shape):
        instance = segmentation[index] - 1
        if instance != -1:
            m_0[instance] += 1
            for i in range(dim):
                m_1[instance, i] += index[i]

    return m_1.astype(np.float32) / m_0.reshape(-1, 1)


@numba.njit(parallel=False, cache=byotrack.NUMBA_CACHE)
def _median_from_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Return the center (median) of each instance in the segmentation."""
    # Flatten space axes
    flat_segmentation = segmentation.reshape(-1)

    n = segmentation.max()
    median = np.zeros((n, len(segmentation.shape)), dtype=np.float32)
    counts = np.zeros(n, dtype=np.uint)

    if n == 0:
        return median

    for i in range(flat_segmentation.shape[0]):
        instance = flat_segmentation[i] - 1
        if instance != -1:
            counts[instance] += 1

    m = np.max(counts)

    # Reset counts and allocate position
    counts[:] = 0
    positions = np.empty((n, m, len(segmentation.shape)), dtype=np.uint)

    for index in np.ndindex(*segmentation.shape):
        instance = segmentation[index] - 1
        if instance != -1:
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
    # A bit slower than previous version in 2D, but still fine

    n = segmentation.max()
    dim = len(segmentation.shape)

    bbox = np.zeros((n, 2 * dim), dtype=np.int32)
    mini = np.full((n, dim), np.iinfo(np.int32).max, dtype=np.int32)
    maxi = np.zeros((n, dim), dtype=np.int32)

    for index in np.ndindex(*segmentation.shape):
        instance = segmentation[index] - 1
        if instance != -1:
            for i in range(dim):
                mini[instance, i] = min(mini[instance, i], index[i])
                maxi[instance, i] = max(maxi[instance, i], index[i])

    # Keep 0 for undefined element
    defined = mini[:, 0] != np.iinfo(np.int32).max

    bbox[defined, :dim] = mini[defined]
    bbox[defined, dim:] = maxi[defined] - mini[defined] + 1

    return bbox


def _segmentation_from_bbox(bbox: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) == 2:  # noqa: PLR2004
        return _segmentation_from_bbox_2d(bbox, shape)
    if len(shape) == 3:  # noqa: PLR2004
        return _segmentation_from_bbox_3d(bbox, shape)
    raise RuntimeError(f"Cannot create a segmentation of dimension {len(shape)}")


@numba.njit(cache=byotrack.NUMBA_CACHE)
def _segmentation_from_bbox_2d(bbox: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    # Switch to start/stop bbox instead of start/size bbox
    bbox = bbox.copy()
    bbox[:, 2:] += bbox[:, :2]
    bbox.clip(0, out=bbox)  # Clip bbox to prevent negative indices

    segmentation = np.zeros(shape, dtype=np.int32)
    for label, bbox_ in enumerate(bbox):
        segmentation[bbox_[0] : bbox_[2], bbox_[1] : bbox_[3]] = label + 1

    return segmentation


@numba.njit(cache=byotrack.NUMBA_CACHE)
def _segmentation_from_bbox_3d(bbox: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    # Switch to start/stop bbox instead of start/size bbox
    bbox = bbox.copy()
    bbox[:, 3:] += bbox[:, :3]
    bbox.clip(0, out=bbox)  # Clip bbox to prevent negative indices

    segmentation = np.zeros(shape, dtype=np.int32)
    for label, bbox_ in enumerate(bbox):
        segmentation[bbox_[0] : bbox_[3], bbox_[1] : bbox_[4], bbox_[2] : bbox_[5]] = label + 1

    return segmentation


def _segmentation_from_position(position: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    position = position.round().int()
    valid = (position >= 0).all(dim=-1) & (position < torch.tensor(shape)).all(dim=-1)
    labels = torch.arange(1, position.shape[0] + 1, dtype=torch.int32)[valid]
    position = position[valid]

    segmentation = torch.zeros(shape, dtype=torch.int32)
    segmentation[position.T.tolist()] = labels
    return segmentation


@numba.njit(parallel=True, cache=byotrack.NUMBA_CACHE)
def _fast_unique(segmentation: np.ndarray) -> np.ndarray:
    """Fast np.unique/torch.unique.

    It is 30 to 60 times faster than its counterparts, but it assumes that seg.max() is small
    before the number of pixels of the image (which is always the case in practice).
    """
    segmentation = segmentation.reshape(-1)
    unique = np.zeros(segmentation.max() + 1, dtype=np.bool_)

    for i in numba.prange(segmentation.size):
        unique[segmentation[i]] = True

    return np.arange(segmentation.max() + 1)[unique]


@numba.njit(parallel=True, cache=byotrack.NUMBA_CACHE)
def _fast_relabel(segmentation: np.ndarray) -> None:
    """Inplace fast relabel.

    It assumes that seg.max() is small before the number of pixels of the image (which is always the case in practice).
    """
    segmentation = segmentation.reshape(-1)
    unique = np.zeros(segmentation.max() + 1, dtype=segmentation.dtype)

    for i in numba.prange(segmentation.size):
        unique[segmentation[i]] = 1

    unique[unique > 0] = np.arange(unique.sum(), dtype=segmentation.dtype)

    for i in numba.prange(segmentation.size):
        if segmentation[i]:
            segmentation[i] = unique[segmentation[i]]


@overload
def relabel_consecutive(segmentation: torch.Tensor, *, inplace=True) -> torch.Tensor: ...


@overload
def relabel_consecutive(segmentation: np.ndarray, *, inplace=True) -> np.ndarray: ...


def relabel_consecutive(segmentation: torch.Tensor | np.ndarray, *, inplace=True) -> torch.Tensor | np.ndarray:
    """Relabel a segmentation mask so that labels are consecutives.

    For N instances, labels are 0 for background and then [1:N] for each instance.

    Args:
        segmentation (torch.Tensor | np.ndarray): Segmentation mask
            Shape: ([D, ]H, W), dtype: int
        inplace (bool): Modify in place the segmentation mask
            Default: True

    Returns:
        torch.Tensor | np.ndarray: The same segmentation mask where labels are consecutive (from 1 to N)

    """
    seg_np: np.ndarray
    if isinstance(segmentation, torch.Tensor):
        if not inplace:
            segmentation = segmentation.clone()

        seg_np = segmentation.numpy()  # Shares the same data with segmentation that will be modify in place
    else:
        if not inplace:
            segmentation = segmentation.copy()

        seg_np = segmentation

    _fast_relabel(seg_np)

    return segmentation


def compress(tensor: torch.Tensor, level=3) -> torch.Tensor:
    """Compress a tensor using zstd (Experimental)."""
    compressed = zstd.compress(tensor.numpy().tobytes(), level=level)
    return torch.frombuffer(compressed, dtype=torch.uint8)


def decompress(tensor: torch.Tensor, dtype=torch.int32) -> torch.Tensor:
    """Decompress a tensor using zstd (Experimental)."""
    decompressed = zstd.decompress(tensor.numpy().tobytes())
    return torch.frombuffer(decompressed, dtype=dtype)


# XXX: PyTorch allocation is quite bad with small tensors (that can results from zipping for instance)
#      Small tensors are typically allocated from cached segmentations, preventing freeing
#      the memory required to store the segmentations. It is quite unclear yet how this works.
#      Our current weird fix is to allocate the segmentation mask on numpy and then using from_numpy
#      which seems to avoid this issue. Otherwise, one would probably have to rethink the all Detections class
#      Or use ugly fixes like glibc.malloc_trim (that seems to work on linux also).
#      See https://github.com/pytorch/pytorch/issues/165319


class Detections:
    """Detections for a given frame.

    Built from a data dict. The data has to contained one of "position",
    "bbox" or "segmentation" keys that respectively define the positions of instances center
    ([k, ]i, j), the bounding boxes of instances ([front, ]top, left, [depth, ]height, width) or the instance
    segmentation of the image ([D, ]H, W).

    It supports 2D and 3D detections. In 2D, the depth axis (k indice) is missing.

    Note:
        All positions/bounding boxes uses the index coordinates system (not xyz). In ByoTrack, the Z axis
        (or depth D) is before the height and the width.
        We usually use the following nomenclature for indices: (k (stack), i (row) and j (columns)).

    Positions are stored as floats (index coordinates): (k, i, j).

    Bounding boxes are stored as ints (index coordinates): (k_0, i_0, j_0, dk, di, dj). It defines
    all the pixels (k, i, j) such that k_0 <= k < k_0 + dk, i_0 <= i <= i_0 + di, j_0 <= j < j_0 + dj.

    The segmentation mask is stored as 2D or 3D integer tensor, where labels are consecutives from 1 to N+1.
    0 is for the background.

    Note:
        The i_th detection in the Detections has the label i+1 in the segmentation mask.

    Additional optional data is also expected like "confidence" or "shape" that respectively
    defines the confidence for each detection and the shape of the image (H, W).

    Any additional meta information on the detections can be also given.

    Defines position, bbox, segmentation, and confidence properties. Each of them are either from
    the data or extrapolated if missing (see _extrapolate_xxx).

    Attributes:
        data (dict[str, toch.Tensor]): Detections data.
        length (int): Number of detections
        dim (int): Dimension of the detections: 2d or 3d.
        frame_id (int): Optional frame id in the original video (-1 if no video)
            In ByoTrack, detections linking do not rely on this frame_id, but rather
            on the position inside the detections_sequence. It should only be used
            for debugging/visualization.
            Default: -1
        shape: (tuple[int, ...]): Shape of the image ([D, ]H, W). (Extrapolated if not given)
        position (torch.Tensor): Positions (k, i, j) of instances (center) inferred from the data
            Shape: (N, dim), dtype: float32
        bbox (torch.Tensor): Bounding boxes of instances inferred from the data
            ([front, ]top, left, [depth, ]height, width)
            Shape: (N, 2*dim), dtype: int32
        segmentation (torch.Tensor): Segmentation inferred from the data
            Shape: ([D, ]H, W), dtype: int32
        confidence (torch.Tensor): Confidence for each instance
            Shape: (N,), dtype: float32
        mass (torch.Tensor): Size of each object in pixel, inferred from the data.
            Shape: (N,), dtype: int32
        use_median_position (bool): Use median instead of mean to compute positions from segmentation.
            Default: True (Usually more robust)

    """

    # XXX: Don't relabel, and allow non consecutive/undefined detections? (needs to be handled by linkers)

    def __init__(self, data: dict[str, torch.Tensor], frame_id: int = -1, *, use_median_position=True) -> None:  # noqa: C901
        self.length = -1
        self.dim = -1
        self._use_median_position = use_median_position

        if "position" in data:
            _check_position(data["position"])
            self.length = data["position"].shape[0]
            self.dim = data["position"].shape[1]

        if "bbox" in data:
            _check_bbox(data["bbox"])
            length = data["bbox"].shape[0]
            dim = data["bbox"].shape[1] // 2

            if self.length not in (-1, length):
                raise ValueError("Inconsistent number of detections in `data`")
            if self.dim not in (-1, dim):
                raise ValueError("Dimensions are inconsistent in `data`")

            self.length = length
            self.dim = dim

        if "segmentation" in data:
            _check_segmentation(data["segmentation"])
            relabel_consecutive(data["segmentation"])  # Relabel inplace
            length = int(data["segmentation"].max())
            dim = data["segmentation"].ndim

            if self.length not in (-1, length):
                raise ValueError("Inconsistent number of detections in `data`")
            if self.dim not in (-1, dim):
                raise ValueError("Dimensions are inconsistent in `data`")

            self.length = length
            self.dim = dim

            # Compress seg
            if byotrack.ZSTD_SEG:
                data["shape"] = torch.tensor(data["segmentation"].shape)
                data["segmentation"] = compress(data["segmentation"].reshape(-1))

        if "condidence" in data:
            _check_confidence(data["confidence"])
            if data["confidence"].shape[0] != self.length:
                raise ValueError("Confidence tensor is expected to have N elements")

        self.data = data
        self._lazy_extrapolated_data: dict[str, torch.Tensor] = {}

        self.shape = self._extrapolate_shape()
        self.frame_id = frame_id

    @property
    def position(self) -> torch.Tensor:  # noqa: D102
        position = self.data.get("position", self._lazy_extrapolated_data.get("position"))
        if position is None:
            position = self._extrapolate_position()
            self._lazy_extrapolated_data["position"] = position

        return position

    @property
    def bbox(self) -> torch.Tensor:  # noqa: D102
        bbox = self.data.get("bbox", self._lazy_extrapolated_data.get("bbox"))
        if bbox is None:
            bbox = self._extrapolate_bbox()
            self._lazy_extrapolated_data["bbox"] = bbox

        return bbox

    @property
    def segmentation(self) -> torch.Tensor:  # noqa: D102
        segmentation = self.data.get("segmentation", self._lazy_extrapolated_data.get("segmentation"))
        if segmentation is None:
            segmentation = self._extrapolate_segmentation()
            self._lazy_extrapolated_data["segmentation"] = segmentation
            if byotrack.ZSTD_SEG:
                self._lazy_extrapolated_data["segmentation"] = compress(segmentation.reshape(-1))

        if byotrack.ZSTD_SEG:
            segmentation = decompress(segmentation).reshape(self.shape)

        return segmentation

    @property
    def confidence(self) -> torch.Tensor:  # noqa: D102
        confidence = self.data.get("confidence", self._lazy_extrapolated_data.get("confidence"))
        if confidence is None:
            confidence = self._extrapolate_confidence()
            self._lazy_extrapolated_data["confidence"] = confidence

        return confidence

    @property
    def mass(self) -> torch.Tensor:  # noqa: D102
        mass = self.data.get("mass", self._lazy_extrapolated_data.get("mass"))
        if mass is None:
            mass = self._extrapolate_mass()
            self._lazy_extrapolated_data["confidence"] = mass

        return mass

    @property
    def use_median_position(self) -> bool:  # noqa: D102
        return self._use_median_position

    @use_median_position.setter
    def use_median_position(self, value: bool) -> None:
        if value is self._use_median_position:
            return

        self._use_median_position = value

        # Invalidate computed positions
        self._lazy_extrapolated_data.pop("position")

    def _extrapolate_shape(self) -> tuple[int, ...]:
        """Extrapolate shape from data.

        Either given in data, or from the segmentation shape, or from the shape needed to fit positions/bboxes.

        (Stored in `self.shape`, not in `self._lazy_extrapolated_data`)

        """
        if "shape" in self.data:
            shape = tuple(map(int, self.data["shape"].tolist()))
            if len(shape) != self.dim:
                raise ValueError("Dimensions are inconsistent in `data`")
            return shape

        if "segmentation" in self.data:
            return self.data["segmentation"].shape

        if self.length == 0:
            return (1, 1, 1) if self.dim == 3 else (1, 1)  # No data to extrapolate it.  # noqa: PLR2004

        if "bbox" in self.data:
            maxi = self.data["bbox"][:, : self.dim] + self.data["bbox"][:, self.dim :]
            return tuple(map(int, maxi.max(dim=0).values))

        maxi = self.data["position"].max(dim=0).values.ceil().int() + 1
        return tuple(map(int, maxi))

    def _extrapolate_position(self) -> torch.Tensor:
        """Extrapolate position from data.

        Average the segmentation for each label if exists or uses the bbox

        """
        if "segmentation" in self.data:
            segmentation = self.data["segmentation"]

            if byotrack.ZSTD_SEG:
                segmentation = decompress(segmentation).reshape(self.shape)

            if self.use_median_position:
                return torch.from_numpy(_median_from_segmentation(segmentation.numpy()))
            return torch.from_numpy(_position_from_segmentation(segmentation.numpy()))

        position = self.data["bbox"][:, : self.dim] + (self.data["bbox"][:, self.dim :] - 1) / 2
        position[self.data["bbox"][:, self.dim :].min(dim=-1).values == 0] = torch.nan

        return position

    def _extrapolate_bbox(self) -> torch.Tensor:
        """Extrapolate bbox from data.

        Bounding box of the segmentation if exists, or return a bbox of shape 1 centered on the position

        """
        if "segmentation" in self.data:
            segmentation = self.data["segmentation"]

            if byotrack.ZSTD_SEG:
                segmentation = decompress(segmentation).reshape(self.shape)

            return torch.from_numpy(_bbox_from_segmentation(segmentation.numpy()))

        invalid = torch.isnan(self.data["position"]).any(dim=-1)

        bbox = torch.ones((self.length, 4), dtype=torch.int32)
        bbox[invalid] = 0
        bbox[~invalid, : self.dim] = self.data["position"][~invalid].round().int()
        return bbox

    def _extrapolate_segmentation(self) -> torch.Tensor:
        """Extrapolate segmentation from data.

        Fill the bounding boxes if exists, or fill one pixel on each position

        Warning: unable to handle overlapping bboxes/positions

        """
        if "bbox" in self.data:
            return torch.from_numpy(_segmentation_from_bbox(self.data["bbox"].numpy(), self.shape))

        return _segmentation_from_position(self.data["position"], self.shape)

    def _extrapolate_confidence(self) -> torch.Tensor:
        """Extrapolate confidence."""
        return torch.ones(self.length)

    def _extrapolate_mass(self) -> torch.Tensor:
        if "segmentation" in self.data:
            segmentation = self.data["segmentation"]

            if byotrack.ZSTD_SEG:
                segmentation = decompress(segmentation).reshape(self.shape)

            return torch.from_numpy(_compute_mass(segmentation.numpy()))

        if "bbox" in self.data:
            return self.data["bbox"][:, self.dim :].prod(dim=-1)

        return torch.ones(self.length, dtype=torch.int32)

    def __len__(self) -> int:  # noqa: D105
        return self.length

    def save(self, path: str | os.PathLike) -> None:
        """Save detections to a file using `torch.save`.

        Args:
            path (str | os.PathLike): Output path

        """
        torch.save({"frame_id": self.frame_id, **self.data}, path)

    @staticmethod
    def load(path: str | os.PathLike) -> Detections:
        """Load a detections for a given frame using `torch.load`.

        Args:
            path (str | os.PathLike): Input path

        """
        data = cast("dict", torch.load(path, map_location="cpu", weights_only=True))
        frame_id = cast("int", data.pop("frame_id"))

        # Handle compressed segmentation
        if "segmentation" in data:
            segmentation = cast("torch.Tensor", data["segmentation"])
            if segmentation.ndim == 1:  # Compressed
                assert segmentation.dtype == torch.uint8  # noqa: S101
                assert "shape" in data  # noqa: S101

                data["segmentation"] = decompress(segmentation).reshape(*data["shape"].tolist())

        return Detections(data, frame_id)

    @staticmethod
    def save_multi_frames_detections(detections_sequence: Sequence[Detections], path: str | os.PathLike) -> None:
        """Save detections for a sequence of frames.

        It will save the detections as::

            path/{frame_id}.pt
                 ...


        Args:
            detections_sequence (Sequence[Detections]): Detections for each frame
                Each detections should have a different frame_id
            path (str | os.PathLike): Output folder

        """
        path = pathlib.Path(path)
        path.mkdir(parents=True)

        for i, detections in enumerate(detections_sequence):
            detections.save(path / f"{i}.pt")

    @staticmethod
    def load_multi_frames_detections(path: str | os.PathLike) -> list[Detections]:
        """Load detections for a sequence of frames.

        Expect the following file structure::

            path/{0}.pt
                 ...
                 {i}.pt
                 ...
                 {n}.pt

        Args:
            path (str | os.PathLike): Input folder

        Returns:
            list[Detections]: Detections for each frame (sorted by frame id)

        """
        path = pathlib.Path(path)
        files = utils.sorted_alphanumeric(path.iterdir())

        detections_sequence: list[Detections] = []

        for i, file in enumerate(file for file in files if file.suffix == ".pt"):
            if file.stem != f"{i}":
                raise KeyError(f"The {i}th file is not '{i}.pt'")
            detections_sequence.append(Detections.load(file))

        return detections_sequence
