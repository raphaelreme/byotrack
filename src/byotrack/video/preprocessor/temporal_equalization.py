from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch_tps
import tqdm.auto as tqdm

from byotrack.video.preprocessor import preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class TemporalEqualizer(preprocessor.VideoPreprocessor):
    """Equalize slow temporal drifts of illumination in the video.

    Some acquisitions suffer from a slow, roughly monotonous variation of the global intensity
    through time (e.g., bleaching, light source drift, ...). This preprocessor measures the
    temporal evolution of a reference intensity `quantile` (assumed to stay roughly constant
    for the imaged sample), smooths/interpolates this (possibly noisy) evolution and divides
    every frame by its expected quantile value, so that this quantile becomes ~1.0 for the
    whole video.

    The quantile is only measured every `frame_step` frame (for speed) and then regressed
    through time with a thin plate spline (`self.tps`), regularized by `alpha`.

    Note:
        Dividing by the expected quantile does not bound the output values to a known range
        (Contrarily to `IntensityNormalizer`). This preprocessor should therefore be followed
        by an `IntensityNormalizer` (not necessarily on the same quantile) to rescale the video.

    Attributes:
        quantile (float): Reference intensity quantile whose temporal evolution is measured
            and equalized (Ex: 0.5 for the median).
        frame_step (int): Only one frame every `frame_step` is used to measure the quantile
            evolution through time, to speed up the computation.
            Default: 5
        tps (torch_tps.ThinPlateSpline): Regressor used to smooth and interpolate the measured
            quantiles through time. Built with the given `alpha` regularization factor
            (0.0 means exact interpolation of the measured quantiles).
            Default: alpha=50.0

    """

    def __init__(self, quantile: float, frame_step: int = 5, alpha: float = 50.0) -> None:
        super().__init__()
        self._dtype = np.dtype(np.float32)

        self.quantile = quantile
        self.frame_step = frame_step
        self.tps = torch_tps.ThinPlateSpline(alpha)

        self._quantiles: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self._ratios: np.ndarray = np.zeros((0, 1), dtype=np.float32)

    @override
    def initialize(self, video: Sequence[np.ndarray] | np.ndarray, frame_ids: list[int] | None = None) -> None:
        """Initialize the preprocessor for the given video.

        Reads the frame every `frame_step` and compute their `quantile`.
        Store a smoothed and interpolated version (via `self.tps`) of the quantile.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): The video to preprocess.
                Sequence of T frames (array). Each array is expected to have a shape ([D, ]H, W, C).
            frame_ids (list[int]): Optional indices of the frames in the video.
                Default: None (default to [0, ...,T-1])
        """
        super().initialize(video)
        frame_ids = frame_ids or list(range(len(video)))
        self._dtype = np.dtype(np.float32)  # Change the dtype of the video

        self._quantiles = np.array(
            [
                np.quantile(frame, self.quantile, axis=range(frame.ndim - 1))
                for frame in tqdm.tqdm(
                    video[:: self.frame_step], desc="IntensityTemporalEqualization - Computing quantiles..."
                )
            ]
        ).astype(np.float32, copy=False)

        self.tps.fit(
            torch.tensor(frame_ids, dtype=torch.float32)[:: self.frame_step], torch.from_numpy(self._quantiles)
        )
        self._ratios = self.tps.transform(torch.arange(0, max(frame_ids) + 1, dtype=torch.float32)).numpy()

    @override
    def preprocess_frame(self, frame: np.ndarray, frame_id=0) -> np.ndarray:
        frame = frame.astype(np.float32, copy=False)
        frame /= self._ratios[frame_id]
        return frame

    @override
    def preprocess_video(
        self, video: Sequence[np.ndarray] | np.ndarray, frame_ids: list[int] | None = None
    ) -> np.ndarray:
        if not isinstance(video, np.ndarray):
            return super().preprocess_video(video, frame_ids)

        frame_ids = frame_ids or list(range(len(video)))

        # Initialize for this video
        self.initialize(video, frame_ids)

        video = video.astype(np.float32, copy=False)
        video /= self._ratios[frame_ids][(slice(None),) + (None,) * (video.ndim - 2)]
        return video
