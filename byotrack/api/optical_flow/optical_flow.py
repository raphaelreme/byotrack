from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import scipy.ndimage  # type: ignore


class OpticalFlow(ABC):
    """Optical flow algorithm (Abstract class, see implementations)

    It implements or wraps an existing method.
    In ByoTrack, frames are multidimensional floating arrays, they are
    the inputs of the optical flow algorithm. It outputs an optical flow map which
    is a multidimensional floating array where the first dimension is the flow for
    each spatial dimension.

    For instance, with two dimensional frames (H, W, C), the computed flow map has a shape
    (2, H, W), where the first (resp. second) element of the first dimension is the displacement
    along height (resp. width). The coordinate system of ByoTrack is using the pixel index (i, j) and not
    the pixel position (x, y) (which is often use in OpenCv)

    Additionally, the wrapper allows to downscale (with a blurring) the image to speed up computations.

    Usage:

    .. code-block:: python

        video: Video
        optflow: OpticalFlow

        # Preprocess the frames (reference and moving)
        ref = optflow.preprocess(video[0])
        mov = optflow.preprocess(video[1])

        # Compute the flow map from ref to mov
        flow_map = optflow.compute(ref, mov)

        # flow_map[:, i, j] gives the displacement (di, dj) of the pixel (i, j) of the reference image

        # You can warp mov into ref with warp
        mov_warp = optflow.warp(flow_map, video[1])

        # Or use the flow to move points


    Attributes:
        downscale (Union[float, np.ndarray]): Downscale factor. Can be provided for each spatial axis.
            Default: 2.0
        blur (Union[None, float, np.ndarray]): Std of the Gaussian blur applied before downscaling the images
            Can be provided for each spatial axis. If not provided, a default one is choosen given the downscale.

    """

    def __init__(self, downscale: Union[float, np.ndarray] = 2.0, blur: Union[None, float, np.ndarray] = None):
        self.downscale = downscale
        self.blur = blur

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Downscale a frame before running the optical flow algorithm

        It blurs then downscales the frame.

        Args:
            frame (np.ndarray): The frame to pre-preprocess
                Shape: (H, W, C), dtype: float64

        Returns:
            np.ndarray: The blurred and downscaled frame
                Shape: (H', W', C), dtype: float32

        """
        downscale = np.ones(frame.ndim, np.float32)
        downscale[:-1] = self.downscale  # Do not downscale channels

        blur = np.zeros(frame.ndim, np.float32)
        if self.blur is None:
            blur[:-1] = 2 * downscale[:-1] / 6
        else:
            blur[:-1] = self.blur

        # Blur
        frame = scipy.ndimage.gaussian_filter(frame.astype(np.float32), blur, mode="reflect")

        # Downscale (Linear interpolation)
        return scipy.ndimage.zoom(frame, 1 / downscale, order=1, mode="mirror", grid_mode=True)

    @abstractmethod
    def compute(self, reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
        """Compute the optical flow map from reference to the moving frame

        It computes the displacement of each pixel from the reference frame to be in the moving one.

        Args:
            reference (np.ndarray): Reference frame
                Shape: (H', W', C), dtype: float
            moving (np.ndarray): Moving frame
                Shape: (H', W', C), dtype: float

        Returns:
            np.ndarray: Optical flow map from reference to moving
                The pixel coordinates are stored as (i, j) (and not (x, y)), so is
                their displacement.
                Shape: (2, H', W'), dtype: float32
        """

    def flow_at(self, flow_map: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Extract the flow/displacement at the given location

        The output displacement and the given location are expected to be at the frame scale.

        Args:
            flow_map (np.ndarray): Optical flow map (Displacement in i, j for each pixel)
                Shape: (2, H', W'), dtype: float32
            points (np.ndarray): Points where to extract the displacement (i, j) (Not downscaled)
                Shape: (N, 2), dtype: float

        Returns:
            np.ndarray: The displacement at points (i, j) (Not downscaled)
                Shape: (N, 2), dtype: float
        """
        if isinstance(self.downscale, (int, float)):
            downscale = np.array([self.downscale] * flow_map.shape[0])
        else:
            downscale = self.downscale

        flow = np.empty_like(points)

        for axis in range(flow_map.shape[0]):
            flow[:, axis] = scipy.ndimage.map_coordinates(flow_map[axis], points.T / downscale[axis])

        flow *= downscale

        return flow

    def transform(self, flow_map: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Apply the flow to the given points

        The points are expected to be at the frame scale (not downscaled)

        Args:
            flow_map (np.ndarray): Optical flow map (Displacement in i, j for each pixel)
                Shape: (2, H', W'), dtype: float32
            points (np.ndarray): Points to transform (i, j) (Not downscaled)
                Shape: (N, 2), dtype: float

        Returns:
            np.ndarray: New positions of the points after the flow
                Shape: (N, 2), dtype: float
        """
        return points + self.flow_at(flow_map, points)

    def warp(self, flow_map: np.ndarray, moving: np.ndarray) -> np.ndarray:
        """Warps the moving image onto the reference using the flow map

        Note:
            We only implemented backward warping (It only allows to warp the moving image onto the reference one)
            Forward warping could be implemented in a future version if needed.

        Args:
            flow_map (np.ndarray): Optical flow map. Displacement (di, dj) for each pixel (i, j) of the
                reference frame.
                Shape: (2, H, W), dtype: float32
            moving (np.ndarray): The moving image to warp.
                Shape: (H, W, C]), dtype: float

        Returns:
            np.ndarray: Warped image
                Shape: (H, W, C), dtype: float

        """
        *dims, channels = moving.shape
        points = np.indices(dims, np.float32).reshape(2, -1).transpose()
        points = self.transform(flow_map, points).transpose().reshape(2, *dims)

        return np.concatenate(
            [
                scipy.ndimage.map_coordinates(moving[..., channel], points, mode="nearest", order=1)[..., None]
                for channel in range(channels)
            ],
            axis=-1,
        )


class DummyOpticalFlow(OpticalFlow):
    """Dummy optical flow which predict no displacement for each pixel"""

    def compute(self, reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
        return np.zeros((reference.ndim - 1, *reference.shape[:-1]))
