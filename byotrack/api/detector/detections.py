from __future__ import annotations

import os
import re
from typing import Collection, Dict, Iterable, List, Tuple, Union

import numba  # type: ignore
import numpy as np
import torch


def sorted_alphanumeric(data: Iterable[str]):
    """Sorts alphanumeriacally an iterable of strings

    "1" < "2" < "10" < "foo1" < "foo2" < "foo3"

    """

    def alphanum_key(key: str) -> List[Union[str, int]]:
        def convert(text: str) -> Union[str, int]:
            return int(text) if text.isdigit() else text.lower()

        return [convert(text) for text in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key)


def _check_segmentation(segmentation: torch.Tensor) -> None:
    """Check segmentation validity (type, shape, labels)"""
    assert len(segmentation.shape) == 2
    assert segmentation.dtype is torch.int32

    # Check that labels are in [0:N+1]
    assert torch.unique(segmentation).shape[0] == segmentation.max() + 1, "Labels are not consecutive"


def _check_bbox(bbox: torch.Tensor) -> None:
    """Check box validity (type, shape)"""
    assert len(bbox.shape) == 2
    assert bbox.shape[1] == 4, "Bbox should have 4 values (top, left, height, width)"
    assert bbox.dtype is torch.int32


def _check_position(position: torch.Tensor) -> None:
    """Check position validity (type, shape)"""
    assert len(position.shape) == 2
    assert position.shape[1] == 2, "Position should have 2 values (i, j)"
    assert position.dtype is torch.float32


def _check_confidence(confidence: torch.Tensor) -> None:
    """Check confidence validity (type, shape)"""
    assert len(confidence.shape) == 1
    assert confidence.dtype is torch.float32


@numba.njit
def _position_from_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Return the centre of each instance in the segmentation"""
    n = segmentation.max()
    positions = np.zeros((n, 2), dtype=np.float32)
    m_00 = np.zeros(n, dtype=np.uint)
    m_01 = np.zeros(n, dtype=np.uint)
    m_10 = np.zeros(n, dtype=np.uint)

    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[1]):
            instance = segmentation[i, j] - 1
            if instance != -1:
                m_00[instance] += 1
                m_01[instance] += i
                m_10[instance] += j

    positions[:, 0] = m_01 / m_00
    positions[:, 1] = m_10 / m_00
    return positions


@numba.njit
def _bbox_from_segmentation(segmentation: np.ndarray) -> np.ndarray:
    n = segmentation.max()
    bbox = np.zeros((n, 4), dtype=np.int32)
    mini = np.ones((n, 2), dtype=np.int32) * np.inf
    maxi = np.zeros((n, 2), dtype=np.int32)

    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[1]):
            instance = segmentation[i, j] - 1
            if instance != -1:
                mini[instance] = min(mini[instance][0], i), min(mini[instance][1], j)
                maxi[instance] = max(maxi[instance][0], i), max(maxi[instance][1], j)

    bbox[:, :2] = mini
    bbox[:, 2:] = maxi - mini + 1
    return bbox


@numba.njit
def _segmentation_from_bbox(bbox: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    segmentation = np.zeros(shape, dtype=np.int32)
    for label, bbox_ in enumerate(bbox):
        segmentation[bbox_[0] : bbox_[0] + bbox_[2], bbox_[1] : bbox_[1] + bbox_[3]] = label + 1

    return segmentation


def _segmentation_from_position(position: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    segmentation = torch.zeros(shape, dtype=torch.int32)
    segmentation[position.round().int().T.tolist()] = torch.arange(1, position.shape[0] + 1, dtype=torch.int32)
    return segmentation


def relabel_consecutive(segmentation: torch.Tensor) -> torch.Tensor:
    """Relabel a segmentation mask so that labels are consecutives

    For N instances, labels are 0 for background and then [1:N] for each instance.

    Args:
        segmentation (torch.Tensor): Segmentation mask
            Shape: (H, W), dtype: int32

    Returns:
        torch.Tensor: The same segmentation mask where labels are consecutive (from 0 to N)

    """
    labels: torch.Tensor = torch.unique(segmentation)
    max_label = labels.max().item()

    assert isinstance(max_label, int)

    if max_label + 1 == labels.shape[0]:
        return segmentation  # Nothing to do

    # Has to go to int64 has torch does not support indexing with int32 yet
    mapping = torch.zeros(max_label + 1, dtype=torch.int64)
    mapping[labels.to(torch.int64)] = torch.arange(labels.shape[0])

    return mapping[segmentation.to(torch.int64)].to(torch.int32)


class Detections:
    """Detections for a given frame

    Built from a data dict. The data has to contained one of "position",
    "bbox" or "segmentation" keys that respectively define the positions of instances center
    (i, j), the bounding boxes of instances (top, left, height, width) or the instance
    segmentation of the image (H, W).

    Positions are stored as floats (row first).
    Bounding boxes are stored as ints (row first), thus right = left + width - 1
    The labels of the segmentation mask have to be consecutive. You can make it consecutive
    using `relabel_consecutive`.

    Additional optional data is also expected like "confidence" or "shape" that respectively
    defines the confidence for each detection and the shape of the image (H, W).

    Any additional meta information on the detections can be also given.

    Defines position, bbox, segmentation, and confidence properties. Each of them are either from
    the data or extrapolated if missing (see _extrapolate_xxx).

    Attributes:
        data (Dict[str, toch.Tensor]): Detections data.
        length (int): Number of detections
        frame_id (int): Frame id in the video (-1 if no video)
            Default: -1
        shape: (Tuple[int, int]): Shape of the image (H, W). (Extrapolated if not given)
        position (torch.Tensor): Positions (i, j) of instances centre inferred from the data
            Shape: (N, 2), dtype: float32
        bbox (torch.Tensor): Bounding boxes of instances inferred from the data
            Shape: (N, 4), dtype: int32
        segmentation (torch.Tensor): Segmentation inferred from the data
            Shape: (H, W), dtype: int32
        confidence (torch.Tensor): Confidence for each instance
            Shape: (N,), dtype: float32

    """

    def __init__(self, data: Dict[str, torch.Tensor], frame_id: int = -1) -> None:
        self.length = -1

        if "position" in data:
            _check_position(data["position"])
            self.length = data["position"].shape[0]

        if "bbox" in data:
            _check_bbox(data["bbox"])
            length = data["bbox"].shape[0]

            assert self.length in (-1, self.length)
            self.length = length

        if "segmentation" in data:
            _check_segmentation(data["segmentation"])
            length = int(data["segmentation"].max())

            assert self.length in (-1, self.length)
            self.length = length

        assert self.length != 1, "Cannot built detections without any position, bbox or segmentation"

        if "condidence" in data:
            _check_confidence(data["confidence"])
            assert data["confidence"].shape[0] == self.length

        self.data = data
        self._lazy_extrapolated_data: Dict[str, torch.Tensor] = {}

        self.shape = self._extrapolate_shape()
        self.frame_id = frame_id

    @property
    def position(self) -> torch.Tensor:
        position = self.data.get("position", self._lazy_extrapolated_data.get("position"))
        if position is None:
            position = self._extrapolate_position()
            self._lazy_extrapolated_data["position"] = position

        return position

    @property
    def bbox(self) -> torch.Tensor:
        bbox = self.data.get("bbox", self._lazy_extrapolated_data.get("bbox"))
        if bbox is None:
            bbox = self._extrapolate_bbox()
            self._lazy_extrapolated_data["bbox"] = bbox

        return bbox

    @property
    def segmentation(self) -> torch.Tensor:
        segmentation = self.data.get("segmentation", self._lazy_extrapolated_data.get("segmentation"))
        if segmentation is None:
            segmentation = self._extrapolate_segmentation()
            self._lazy_extrapolated_data["segmentation"] = segmentation

        return segmentation

    @property
    def confidence(self) -> torch.Tensor:
        confidence = self.data.get("confidence", self._lazy_extrapolated_data.get("confidence"))
        if confidence is None:
            confidence = self._extrapolate_confidence()
            self._lazy_extrapolated_data["confidence"] = confidence

        return confidence

    def _extrapolate_shape(self) -> Tuple[int, int]:
        """Extrapolate shape from data

        Etiher given in data, or from the segmentation shape, or from the shape needed to fit positions/bboxes.

        (Stored in `self.shape`, not in `self._lazy_extrapolated_data`)

        """
        if "shape" in self.data:
            return (int(self.data["shape"][0]), int(self.data["shape"][1]))

        if "segmentation" in self.data:
            return (self.data["segmentation"].shape[0], self.data["segmentation"].shape[1])

        if "bbox" in self.data:
            maxi = self.data["bbox"][:, :2] + self.data["bbox"][:, 2:]
            shape = maxi.max(dim=0).values
            return (int(shape[0]), int(shape[1]))

        shape = self.data["position"].max(dim=0).values.ceil().int() + 1
        return (int(shape[0]), int(shape[1]))

    def _extrapolate_position(self) -> torch.Tensor:
        """Extrapolate position from data

        Average the segmentation for each label if exists or uses the bbox

        """
        if "segmentation" in self.data:
            return torch.tensor(_position_from_segmentation(self.data["segmentation"].numpy()))

        return self.data["bbox"][:, :2] + (self.data["bbox"][:, 2:] - 1) / 2

    def _extrapolate_bbox(self) -> torch.Tensor:
        """Extrapolate bbox from data

        Bounding box of the segmentation if exists, or return a bbox of shape 1 centered on the position

        """
        if "segmentation" in self.data:
            return torch.tensor(_bbox_from_segmentation(self.data["segmentation"].numpy()))

        bbox = torch.ones((self.length, 4), dtype=torch.int32)
        bbox[:, :2] = self.data["position"].round().int()
        return bbox

    def _extrapolate_segmentation(self) -> torch.Tensor:
        """Extrapolate segmentation from data

        Fill the bounding boxes if exists, or fill one pixel on each position

        Warning: unable to handle overlapping bboxes/positions

        """
        if "bbox" in self.data:
            return torch.tensor(_segmentation_from_bbox(self.data["bbox"].numpy(), self.shape))

        return _segmentation_from_position(self.data["position"], self.shape)

    def _extrapolate_confidence(self) -> torch.Tensor:
        """Extrapolate confidence"""
        return torch.ones(self.length)

    def __len__(self) -> int:
        return self.length

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Save detections to a file using `torch.save`

        Args:
            path (str | os.PathLike): Output path

        """
        torch.save({"frame_id": self.frame_id, **self.data}, path)

    @staticmethod
    def load(path: Union[str, os.PathLike]) -> Detections:
        """Load a detections for a given frame using `torch.load`

        Args:
            path (str | os.PathLike): Input path

        """
        data = torch.load(path, map_location="cpu")
        assert isinstance(data, dict)
        frame_id = data.pop("frame_id")

        return Detections(data, frame_id)

    @staticmethod
    def save_multi_frames_detections(
        detections_sequence: Collection[Detections], path: Union[str, os.PathLike]
    ) -> None:
        """Save detections for a sequence of frames

        It will save the detections as::

            path/{frame_id}.pt
                 ...


        Args:
            detections_sequence (Collection[Detections]): Detections for each frame
                Each detections should have a different frame_id
            path (str | os.PathLike): Output folder

        """
        os.makedirs(path)

        for detections in detections_sequence:
            detections.save(os.path.join(path, f"{detections.frame_id}.pt"))

    @staticmethod
    def load_multi_frames_detections(path: Union[str, os.PathLike]) -> Collection[Detections]:
        """Load detections for a sequence of frames

        Expect the following file structure::

            path/{frame_id}.pt
                 ...

        Args:
            path (str | os.PathLike): Input folder

        Returns:
            Collection[Detections]: Detections for each frame (sorted by frame id)

        """
        files = os.listdir(path)

        detections_sequence: List[Detections] = []

        for file in filter(lambda file: file[-3:] == ".pt", sorted_alphanumeric(files)):
            detections_sequence.append(Detections.load(os.path.join(path, file)))
            if detections_sequence[-1].frame_id == int(file[:-3]):
                raise ValueError(f"Detections {file} has a different saved frame_id")

        return detections_sequence
