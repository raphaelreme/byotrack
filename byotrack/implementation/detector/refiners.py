import enum
from typing import cast, Dict, Optional, Tuple
import warnings

import numba  # type: ignore
import numpy as np
import scipy.ndimage as ndi  # type: ignore
import skimage
import torch

import byotrack


@numba.njit(cache=byotrack.NUMBA_CACHE)
def filter_objects_on_size(segmentation: np.ndarray, min_area: float, max_area: float) -> np.ndarray:
    """Filter instances from the segmentation in-place if they do not fit size criteria

    Args:
        segmentation (np.ndarray): Segmentation mask that will be filtered in-place
            Shape ([D, ]H, W), dtype: int
        min_area (float): Minimum number of pixels to be kept in the segmentation.
        max_area (float): Maximum number of pixels to be kept in the segmentation.

    Returns:
        np.ndarray: Deleted instances

    """
    segmentation = segmentation.reshape(-1)
    area = np.zeros(segmentation.max(), np.uint)

    for i in range(segmentation.size):
        instance = segmentation[i] - 1
        if instance != -1:
            area[instance] += 1

    to_delete = (area < min_area) | (area > max_area)

    for i in range(segmentation.size):
        instance = segmentation[i] - 1
        if instance != -1:
            if to_delete[instance]:
                segmentation[i] = 0

    return to_delete


@numba.njit(cache=byotrack.NUMBA_CACHE)
def filter_objects_on_intensity(
    segmentation: np.ndarray, intensity: np.ndarray, mini: float, maxi: float, min_peak: float
) -> np.ndarray:
    """Filter instances from the segmentation in-place if they do not fit intensity criteria

    Args:
        segmentation (np.ndarray): Segmentation mask that will be filtered in-place
            Shape ([D, ]H, W), dtype: int
        intensity (np.ndarray): Intensity map (frame with aggregated channels)
            Shape ([D, ]H, W), dtype: float
        mini (float): Minimum intensity (summed) to be kept in the segmentation.
        maxi (float): Maximum intensity (summed) to be kept in the segmentation.
        min_peak (float): Minimum peak intensity to be kept in the segmentation.

    Returns:
        np.ndarray: Deleted instances

    """
    segmentation = segmentation.reshape(-1)
    intensity = intensity.reshape(-1)
    sum_intensity = np.zeros(segmentation.max(), dtype=intensity.dtype)
    max_intensity = np.zeros(segmentation.max(), dtype=intensity.dtype)

    for i in range(segmentation.size):
        instance = segmentation[i] - 1
        if instance != -1:
            sum_intensity[instance] += intensity[i]
            max_intensity[instance] = max(intensity[i], max_intensity[instance])

    to_delete = (sum_intensity < mini) | (sum_intensity > maxi) | (max_intensity < min_peak)

    for i in range(segmentation.size):
        instance = segmentation[i] - 1
        if instance != -1:
            if to_delete[instance]:
                segmentation[i] = 0

    return to_delete


class FilterDetections(byotrack.DetectionsRefiner):
    """Filter detections based on simple intensity and size criteria

    If images have multiple channels, channels are averaged into a single per-pixel intensity.

    Note: Detections out of the field of view have an area and intensity of 0.

    Attributes:
        min_area (float): Minimum area (pixels) to be kept in the segmentation
        max_area (float): Minimum area (pixels) to be kept in the segmentation
        min_intensity (float): Minimum intensity (summed) to be kept in the segmentation
        max_intensity (float): Maximum intensity (summed) to be kept in the segmentation
        min_peak (float): Minimum peak intensity to be kept in the segmentation

    """

    progress_bar_description = "Detections filtering"

    def __init__(
        self, *, min_area=0.0, max_area=float("inf"), min_intensity=0.0, max_intensity=float("inf"), min_peak=0.0
    ):
        super().__init__()
        self.min_area = min_area
        self.max_area = max_area
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.min_peak = min_peak

    def apply(self, detections, frame=None):
        segmentation = detections.segmentation.numpy().copy()
        deleted = filter_objects_on_size(segmentation, self.min_area, self.max_area)

        if self.min_intensity > 0.0 or self.min_peak > 0.0 or self.max_intensity < float("inf"):
            assert frame is not None, "Cannot filter on intensity without a given frame"
            intensity = frame.mean(axis=-1)  # Sum channels

            deleted |= filter_objects_on_intensity(
                segmentation, intensity, self.min_intensity, self.max_intensity, self.min_peak
            )

        kept = ~torch.tensor(deleted)

        data: Dict[str, torch.Tensor] = {}

        for key, value in detections.data.items():
            if key == "segmentation":
                data[key] = torch.from_numpy(segmentation)
                continue

            if value.shape[0] == len(detections):  # pos, bbox, confidence and others
                data[key] = detections.data[key][kept]
            else:
                data[key] = detections.data[key]

        return byotrack.Detections(
            data,
            frame_id=detections.frame_id,
            use_median_position=detections.use_median_position,
        )


class Watershed:
    """Peak extraction and watershed labeling

    Warning: This part is still experimental and may change in the following versions

    The algorithm works in 2 steps:
    1. Local maxima extraction that will defined the new labels. These are defined on a given image, or from
       the distance transform of binary mask to label. Neighboring local maxima are clustered with `ndi.label`
       to reduce oversegmentation (with a binary dilation to bridge over small gaps).
    2. Watershed algorithm is used to label each pixel in the mask with its corresponding peak

    Attributes:
        maximum_footprint (Optional[np.ndarray]): Binary kernel to use for maximum filtering. Local maxima are defined
            as pixels that are equal or above all their neighbors defined in this kernel.
            Shape: ([d, ]h, w), dtype: np.bool_
            By default, the footprint is a 3x3 square/cube (considering only direct neighbors, including diagonal ones)
        maximum_dilation (Optional[np.ndarray]): Apply binary dilation to the local maxima mask before running
            `ndi.label`. Connected local maxima are merged in a single label before running watershed.
            Shape: ([d, ]h, w), dtype: np.bool_
            By default, dilation is not applied. Only direct maxima neighbors (including diagonal ones) are merged.
        min_peak_intensity (float): Filter local maxima based on their intensity.

    """

    def __init__(
        self,
        maximum_footprint: Optional[np.ndarray] = None,
        maximum_dilation: Optional[np.ndarray] = None,
        min_peak_intensity=0.0,
    ):
        self.maximum_footprint = maximum_footprint
        self.maximum_dilation = maximum_dilation
        self.min_peak_intensity = min_peak_intensity

    def _default_footprint(self, dim: int) -> np.ndarray:
        # Connectivity is defined with a 3x3 square/cube kernel.
        return np.ones([3] * dim, dtype=np.bool_)

    def peak_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract local maxima (peaks) in the image, that are included in mask

        Peaks are returned as a boolean image
        """
        maximum_footprint = self.maximum_footprint
        if maximum_footprint is None:
            maximum_footprint = self._default_footprint(mask.ndim)

        image_max = ndi.maximum_filter(image, footprint=maximum_footprint, mode="nearest")

        peak_mask = image_max == image
        peak_mask &= image > self.min_peak_intensity

        if self.maximum_dilation is not None:
            ndi.binary_dilation(peak_mask, self.maximum_dilation, output=peak_mask)

        peak_mask &= mask

        return peak_mask

    def watershed(self, mask: np.ndarray, source_map: Optional[np.ndarray] = None) -> np.ndarray:
        """Run watershed on the given mask.

        If source_map is not given, the distance map of the mask is used to find local maxima and use watershed.

        Args:
            mask (np.ndarray): Binary segmentation to label into instance segmentation.
                Shape: ([D, ]H, W), dtype: np.uint8 (or bool)
            source_map (Optional[np.ndarray]): "Intensity" value for each pixel of the mask. Local maxima are
                extracted from this map and watershed fills the segmentation based on these intensities.
                Shape: ([D, ], H, W), dtype: np.float

        Returns:
            np.ndarray: Instance segmentation (label) for the given mask
                Shape: ([D, ]H, W), dtype: np.int32

        """
        if source_map is None:
            source_map = cast(np.ndarray, ndi.distance_transform_edt(mask))

        assert (
            source_map.shape == mask.shape
        ), "source_map.shape != mask.shape (channels are not expected in source_map)"

        if mask.dtype != np.bool_:
            mask = mask > 0

        peak_mask = self.peak_mask(source_map, mask)
        markers, _ = ndi.label(peak_mask, self._default_footprint(mask.ndim))  # type: ignore

        return skimage.segmentation.watershed(-source_map, markers=markers, mask=mask)


class WatershedRefiner(byotrack.DetectionsRefiner):
    """Apply Watershed to the given segmentation

    It will convert the instance segmentation of Detections into a binary one, and then apply
    watershed labeling (see `Watershed`).

    Warning: This is still experimental and this may change in future versions

    Args:
        watershed (Watershed): Watershed labeler to use
        mode (WatershedRefiner.Mode): Watershed mode. INTENSITY uses images, EDT distance transforms
            and EDT_LABEL distance transforms by label (avoiding under-segmentation).
        sigma (float): Apply Gaussian smoothing on the frame intensities before running Watershed.
            Only used for INTESITY mode.
            Default: 0.0 (no smoothing)


    """

    progress_bar_description = "Watershed"

    class Mode(enum.Enum):
        """Watershed Mode

        * INTENSITY: Use the averaged intensity in the image to find local maxima and solve watershed
        * EDT: Use the distance transform of the binary segmentation
        * EDT_LABEL: Use the distance transform computed by label in the original segmentation. This ensure
            that at least on maxima is found by label, avoiding merging neighboring labels. (More costly)

        """

        INTENSITY = "intensity"
        EDT = "edt"
        EDT_LABEL = "edt_label"

    def __init__(self, watershed: Watershed, mode: Mode = Mode.EDT, sigma=0.0):
        super().__init__()
        self.watershed = watershed
        self.mode = mode
        self.sigma = sigma

    def apply(self, detections, frame=None):
        if "segmentation" not in detections.data:
            warnings.warn("Watershed can only be applied to segmented Detections. This refiner is skipped")
            return detections

        mask = detections.segmentation.numpy() > 0
        if self.mode == self.Mode.INTENSITY:
            if frame is None:
                warnings.warn("Frame was not provided for mode INTENSITY. Fall back to EDT")
                segmentation = self.watershed.watershed(mask)
            else:
                intensity = frame.mean(axis=-1)  # Average channels
                if self.sigma > 0.0:
                    intensity = ndi.gaussian_filter(intensity, self.sigma)

                segmentation = self.watershed.watershed(mask, intensity)

        elif self.mode == self.Mode.EDT:
            segmentation = self.watershed.watershed(mask)

        else:
            edt = np.zeros(mask.shape)
            labels = detections.segmentation.numpy()
            roi: Tuple[slice, ...]
            for label_idx, roi in enumerate(ndi.find_objects(labels)):
                if roi is None:
                    continue

                # Pad roi with 1 pixel (if possible) to avoid border effect in edt
                pad_roi = tuple(
                    slice(max(slice_.start - 1, 0), min(slice_.stop + 1, mask.shape[ax]))
                    for ax, slice_ in enumerate(roi)
                )

                # Get roi mask
                label_mask = labels[pad_roi] == label_idx + 1
                label_edt = ndi.distance_transform_edt(label_mask)

                edt[pad_roi][label_mask] = label_edt[label_mask]

            segmentation = self.watershed.watershed(mask, edt)

        return byotrack.Detections(
            {"segmentation": torch.from_numpy(segmentation)},
            frame_id=detections.frame_id,
            use_median_position=detections.use_median_position,
        )
