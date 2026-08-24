from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import scipy.ndimage as ndi
import skimage.transform

from byotrack.video.preprocessor import preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


ZoomOrder: TypeAlias = Literal[0, 1, 2, 3, 4, 5]
ZoomMode: TypeAlias = Literal[
    "reflect", "constant", "nearest", "mirror", "wrap", "grid-constant", "grid-mirror", "grid-wrap"
]


class Zoom(preprocessor.VideoPreprocessor):
    """Rescale each frame spatially by a given zoom factor.

    Wraps ``scipy.ndimage.zoom`` to resample every spatial axis ([D, ]H, W) of the video by the
    given per-axis factor. The channel axis is left untouched. Frame dtype is preserved.

    Note: An optional Gaussian pre-filter (`anti_aliasing`) can be enabled to reduce aliasing
        when down-scaling, with a sigma derived from `zoom`.

    Warning: When `anti_aliasing` is enabled, `preprocess_frame` modifies its input frame in
        place (the Gaussian filter is applied in place to avoid an extra large allocation, which
        matters for 3D videos).

    Attributes:
        zoom (tuple[float, ...]): Zoom factor for each spatial axis ([D, ]H, W).
            Values above 1 up-scale, values below 1 down-scale.
        order (ZoomOrder): Spline interpolation order used by `scipy.ndimage.zoom`
            (0: nearest neighbor, 1: linear, up to 5: quintic spline).
            Default: 1
        mode (ZoomMode): Boundary handling mode used by `scipy.ndimage`, used both by the
            optional anti-aliasing filter and by the zoom itself.
            Default: "reflect"
        cval (float): Constant value used when `mode` is "constant" (or "grid-constant").
            Default: 0.0
        anti_aliasing (bool): Whether to Gaussian-blur each frame before zooming, to reduce
            aliasing artifacts when down-scaling.
            Default: False
        anti_aliasing_sigma (tuple[float, ...]): Per-axis Gaussian sigma used when
            `anti_aliasing` is enabled. Computed from `zoom` (0 for axes that are not
            down-scaled).

    """

    def __init__(
        self,
        zoom: tuple[float, ...],
        *,
        order: ZoomOrder = 1,
        mode: ZoomMode = "reflect",
        cval: float = 0.0,
        anti_aliasing: bool = False,
    ) -> None:
        super().__init__()

        self.zoom = zoom
        self.order = order
        self.mode = mode
        self.cval = cval
        self.anti_aliasing = anti_aliasing
        self.anti_aliasing_sigma = tuple(max(0, (1 / zoom - 1) / 2) for zoom in self.zoom)

    @override
    def initialize(self, video: Sequence[np.ndarray] | np.ndarray, frame_ids: list[int] | None = None) -> None:
        """Initialize the preprocessor for the given video.

        This will update the `shape` attribute to reflect the output shape after zooming.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): The video to preprocess.
                Sequence of T frames (array). Each array is expected to have a shape ([D, ]H, W, C).
            frame_ids (list[int]): Optional indices of the frames in the video. Unused.
                Default: None

        """
        super().initialize(video)

        *input_shape, channel = self.shape  # Separate spatial & channel axes

        if len(self.zoom) != len(input_shape):
            raise ValueError(
                f"Zoom expects one zoom factor per spatial axis, got {len(self.zoom)} for a "
                f"video with {len(input_shape)} spatial axes."
            )

        self._shape = (
            *(round(shape * zoom) for shape, zoom in zip(input_shape, self.zoom, strict=True)),
            channel,
        )

    @override
    def preprocess_frame(self, frame: np.ndarray, frame_id=0) -> np.ndarray:
        if self.anti_aliasing:
            # Filtered in place (output=frame): allocating a second buffer would be costly for
            # large (e.g. 3D) frames. This mutates the caller-provided `frame`.
            frame = ndi.gaussian_filter(
                frame,
                self.anti_aliasing_sigma,
                output=frame,
                mode=self.mode,
                cval=self.cval,
                axes=tuple(range(frame.ndim - 1)),
            )

        output = np.empty(self.shape, dtype=frame.dtype)
        for channel in range(frame.shape[-1]):
            # Looping per channel is faster in practice than a single ndi.zoom call with the
            # channel axis batched in (zoom factor 1 on that axis).
            ndi.zoom(
                frame[..., channel],
                self.zoom,
                output=output[..., channel],
                order=self.order,
                mode=self.mode,
                cval=self.cval,
                grid_mode=True,
            )

        return output


class LocalMeanDownscaler(preprocessor.VideoPreprocessor):
    """Down-scale each frame by averaging non-overlapping blocks of pixels.

    Wraps ``skimage.transform.downscale_local_mean`` to reduce every spatial axis ([D, ]H, W) of
    the video by the given per-axis integer factor. The channel axis is left untouched.

    Note: Unlike `Zoom`, this only supports down-scaling by an integer factor, and always
        outputs `float32` (computed directly in `float32`, rather than the `float64` that
        ``downscale_local_mean`` would otherwise use for non-floating inputs, to limit
        memory/compute on large videos).

    Attributes:
        downscale (tuple[int, ...]): Down-scaling factor for each spatial axis ([D, ]H, W).

    """

    def __init__(self, downscale: tuple[int, ...]) -> None:
        super().__init__()
        self._dtype = np.dtype(np.float32)
        self.downscale = downscale

    @override
    def initialize(self, video: Sequence[np.ndarray] | np.ndarray, frame_ids: list[int] | None = None) -> None:
        """Initialize the preprocessor for the given video.

        This will update the `shape` attribute to reflect the output shape after downscaling.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): The video to preprocess.
                Sequence of T frames (array). Each array is expected to have a shape ([D, ]H, W, C).
            frame_ids (list[int]): Optional indices of the frames in the video. Unused.
                Default: None

        """
        super().initialize(video)
        self._dtype = np.dtype(np.float32)  # `downscale_local_mean` always returns a floating dtype

        *input_shape, channel = self.shape  # Separate spatial & channel axes

        if len(self.downscale) != len(input_shape):
            raise ValueError(
                f"LocalMeanDownscaler expects one downscale factor per spatial axis, got "
                f"{len(self.downscale)} for a video with {len(input_shape)} spatial axes."
            )

        self._shape = (
            *(math.ceil(shape / downscale) for shape, downscale in zip(input_shape, self.downscale, strict=True)),
            channel,
        )

    @override
    def preprocess_frame(self, frame: np.ndarray, frame_id=0) -> np.ndarray:
        # Cast to float32 first so the block mean is computed directly in float32 (skimage
        # otherwise upcasts non-floating inputs to float64, doubling memory/compute for ~ no gain).
        return skimage.transform.downscale_local_mean(frame.astype(np.float32, copy=False), (*self.downscale, 1))
