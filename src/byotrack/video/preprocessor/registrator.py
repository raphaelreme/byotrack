from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal

import scipy.ndimage as ndi
import skimage.registration

from byotrack.video.preprocessor import preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class Registrator(preprocessor.VideoPreprocessor):
    """Register the video onto a reference frame.

    This is a "simple" translation registration, where the translation is found by a maximum of correlation
    of the image data on a given reference channel (default to the first channel).

    See ``skimage.registration.phase_cross_correlation`` and ``scipy.ndimage.shift``.

    Note:
        The translation is only estimated on `reference_channel`, but the very same shift is then applied
        to every channel of the frame.

    Warning:
        Phase correlation and shifting are costly operations. If the video is meant to be read several times,
        it may be worth running `preprocess_video` once and caching the result (on disk or in RAM), rather
        than registering it online at each use.

    Attributes:
        reference_channel (int): Channel over which the translation is estimated.
            Default: 0
        reference_frame (np.ndarray | None): The reference frame. If not given, the first frame of the video
            will be used.
            Shape: ([D, ]H, W, C)
        upsample_factor (int): Upsampling factor used to estimate the translation with sub-pixel precision.
            Images will be registered to within ``1 / upsample_factor`` of a pixel.
            See ``skimage.registration.phase_cross_correlation``.
            Default: 10
        normalization (Literal["phase"] | None): Normalization applied to the cross-correlation.
            See ``skimage.registration.phase_cross_correlation``.
            Default: None
        interpolation_order (int): Spline interpolation order used by `scipy.ndimage.shift` to shift the frame.
            0 corresponds to a nearest-neighbor interpolation.
            Default: 3
        padding_mode (str): How pixels outside of the frame boundaries are filled when shifting.
            See `scipy.ndimage.shift` for the accepted values.
            Default: "nearest"

    """

    def __init__(
        self,
        reference_channel: int = 0,
        reference_frame: np.ndarray | None = None,
        *,
        upsample_factor: int = 10,
        normalization: Literal["phase"] | None = None,
        interpolation_order: ndi._interpolation._Order = 3,
        padding_mode: ndi._interpolation._Mode = "nearest",
    ):
        super().__init__()
        self.reference_channel = reference_channel
        self.reference_frame = reference_frame
        self.upsample_factor = upsample_factor
        self.normalization = normalization
        self.interpolation_order = interpolation_order
        self.padding_mode = padding_mode

    @override
    def initialize(self, video: Sequence[np.ndarray] | np.ndarray) -> None:
        """Initialize the preprocessor for the given video.

        This will take the first frame as reference if ``self.reference_frame`` is None, and raise if
        ``self.reference_channel`` is out of bounds for the video.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): The video to preprocess.
                Sequence of T frames (array). Each array is expected to have a shape ([D, ]H, W, C).

        """
        super().initialize(video)

        if self.reference_frame is None:
            self.reference_frame = video[0]

        self.reference_frame[..., self.reference_channel]  # Check that reference_channel is valid

    @override
    def preprocess_frame(self, frame: np.ndarray, frame_id=0) -> np.ndarray:
        if self.reference_frame is None:
            raise ValueError("VideoPreprocessor is not initialized.")

        shift = skimage.registration.phase_cross_correlation(
            self.reference_frame[..., self.reference_channel],
            frame[..., self.reference_channel],
            upsample_factor=self.upsample_factor,
            normalization=self.normalization,
        )[0]

        for channel in range(frame.shape[-1]):
            ndi.shift(
                frame[..., channel].copy() if self.interpolation_order <= 1 else frame[..., channel],
                shift,
                output=frame[..., channel],
                order=self.interpolation_order,
                mode=self.padding_mode,
            )

        return frame
