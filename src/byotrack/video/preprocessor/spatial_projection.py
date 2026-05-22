from __future__ import annotations

import sys
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from byotrack.video.preprocessor import preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class SpatialProjection(preprocessor.VideoPreprocessor):
    """Spatial projection of the video along an axis.

    It allows to project a 3D video onto a 2D one.

    Attributes:
        axes_to_int (dict[str, int]): Mapping from axis name string (``"Z"``, ``"Y"``, ``"X"``,
            ``"D"``, ``"H"``, ``"W"``) to integer axis index (0, 1, 2).
        axis (int): Axis on which to project. Order is (Z, Y, X) ~ (D, H, W) ~ (0, 1, 2).
            Default: 0.
        method (Literal["mean", "min", "max", "select"]): Projection method.
            ``"mean"``, ``"min"`` and ``"max"`` aggregate over the axis with the appropriate function.
            ``"select"`` selects one slice of the volume.
            Default: ``"max"``.
        selected (int): Slice index used when method is ``"select"``.
            Default: 0.
    """

    axes_to_int: ClassVar[dict[str, int]] = {
        "D": 0,
        "H": 1,
        "W": 2,
        "Z": 0,
        "Y": 1,
        "X": 2,
    }

    def __init__(
        self,
        axis: str | int = "Z",
        method: Literal["mean", "min", "max", "select"] = "max",
        selected: int = 0,
    ):
        super().__init__()
        self.axis = self.axes_to_int.get(axis, 100) if isinstance(axis, str) else axis

        if self.axis < 0:
            self.axis = 3 + self.axis

        if self.axis not in (0, 1, 2):
            raise ValueError("Axis should be an integer in (0, 1, 2) (or an axis name in 'ZYX' / 'DHW').")

        self.method = method
        self.selected = selected

    @override
    def initialize(self, video: Sequence[np.ndarray] | np.ndarray) -> None:
        """Initialize the preprocessor for the given video.

        This will set the `shape` attribute correctly, or raise if not 3D.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): The video to preprocess.
                Sequence of T frames (array). Each array is expected to have a shape ([D, ]H, W, C).

        """
        super().initialize(video)

        if len(self.shape) != 4:  # noqa: PLR2004
            raise ValueError("SpatialProjection is only supported for 3D videos.")

        shape = self.shape[self.axis]
        if not -shape < self.selected <= shape:
            raise IndexError(f"index {self.selected} is out of bounds for axis {self.axis + 1} with size {shape}.")

        self._shape = tuple(s for axis, s in enumerate(self.shape) if axis != self.axis)

    @override
    def preprocess_frame(self, frame: np.ndarray, frame_id=0) -> np.ndarray:
        if self.method == "max":
            return frame.max(self.axis)
        if self.method == "min":
            return frame.min(self.axis)
        if self.method == "mean":
            return frame.mean(self.axis, dtype=frame.dtype)

        slices = [slice(None)] * self.axis + [self.selected]
        return frame[tuple(slices)]

    @override
    def preprocess_video(self, video: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        if not isinstance(video, np.ndarray):
            return super().preprocess_video(video)

        # Let's do it directly on the global np.ndarray
        axis = self.axis + 1

        if self.method == "max":
            return video.max(axis)
        if self.method == "min":
            return video.min(axis)
        if self.method == "mean":
            return video.mean(axis, dtype=video.dtype)

        slices = [slice(None)] * (axis) + [self.selected]
        return video[tuple(slices)]
