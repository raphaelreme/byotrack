from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import scipy.ndimage  # type: ignore


class OpticalFlow(ABC):
    """Optical flow algorithm (Abstract class, see implementations)

    It implements or wraps an existing optical flow method.

    In ByoTrack, frames are multidimensional arrays of shape ([D, ]H, W, C).
    Two frames are given as input to the algorithm. It outputs an optical flow map which
    is a multidimensional floating array (dim, [D, ]H, W) where the first dimension is the
    flow (displacement) for each spatial dimension.

    For instance, with two dimensional frames (H, W, C), the computed flow map has a shape
    (2, H, W), where the first (resp. second) element of the first dimension is the displacement
    along height (resp. width). The coordinate system of ByoTrack is using the pixel index (i, j) and not
    the pixel position (x, y) (which is often use in OpenCV). Note that in 3D, the pixel order is (k, i, j)
    where k stand for depth.

    Additionally, the wrapper allows to downscale (with a blurring) the image to speed up computations.

    Usage:

    .. code-block:: python

        video: Video
        optflow: OpticalFlow
        detector: BatchDetector

        # Preprocess the frames (reference and moving)
        ref = optflow.preprocess(video[0])
        mov = optflow.preprocess(video[1])

        # Compute the flow map from ref to mov
        flow_map = optflow.compute(ref, mov)

        # flow_map[:, i, j] gives the displacement (di, dj) of the pixel (i, j) of the reference image (scaled)

        # You can warp mov into ref with warp
        mov_warp = optflow.warp(flow_map, video[1])

        # Or use the flow to move points, for instance positions of detected particles
        detections = detector.detect(video[0][None])[0]  # Detect with a BatchDetector on frame 0
        futur_positions = optflow.transform(detections.position.numpy())  # Expected positions on frame 1


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

        It blurs then downscales the frame. It also converts to float32.

        Args:
            frame (np.ndarray): The frame to pre-preprocess
                It does not normalize the frame. Integer values are simply converted to floating ones.
                Shape: ([D, ]H, W, C)

        Returns:
            np.ndarray: The blurred and downscaled frame (converted to float32)
                Shape: ([D', ]H', W', C), dtype: float32

        """
        downscale = np.ones(frame.ndim, np.float32)
        downscale[:-1] = self.downscale  # Do not downscale channels

        blur = np.zeros(frame.ndim, np.float32)  # Do not blur along channel dimension
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
                Shape: ([D', ]H', W', C), dtype: float
            moving (np.ndarray): Moving frame
                Shape: ([D', ]H', W', C), dtype: float

        Returns:
            np.ndarray: Optical flow map from reference to moving
                The flow field is stored in pixel coordinates ([dk', ]di', dj') (!= xyz)
                Shape: (dim, [D', ]H', W'), dtype: float32
        """

    def flow_at(self, flow_map: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Extract the flow/displacement at the given location

        The flow_map is expected to be downscaled, but the input and outputs points are not.

        Args:
            flow_map (np.ndarray): Optical flow map. Displacement ([dk', ]di', dj') for each pixel ([k', ]i', j') in
                the downscaled coordinates.
                Shape: (dim, [D', ]H', W'), dtype: float32
            points (np.ndarray): Points ([k, ]i ,j) in the original (non-downscale) coordinates
                Shape: (N, dim), dtype: float

        Returns:
            np.ndarray: The displacements ([dk, ], di, dj) in the original coordinates (non-downscale)
                at the given points.
                Shape: (N, dim), dtype: float
        """
        if isinstance(self.downscale, (int, float)):
            downscale = np.array([self.downscale] * flow_map.shape[0])
        else:
            downscale = self.downscale

        flow = np.empty_like(points)

        for axis in range(flow_map.shape[0]):
            flow[:, axis] = scipy.ndimage.map_coordinates(flow_map[axis], points.T / downscale[axis], mode="nearest")

        flow *= downscale

        return flow

    def transform(self, flow_map: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Apply the flow to the given points

        The flow_map is expected to be downscaled, but the input and outputs points are not.

        Args:
            flow_map (np.ndarray): Optical flow map. Displacement ([dk', ]di', dj') for each pixel ([k', ]i', j') in
                the downscaled coordinates.
                Shape: (dim, [D', ]H', W'), dtype: float32
            points (np.ndarray): Points ([k, ]i ,j) in the original (non-downscale) coordinates
                Shape: (N, dim), dtype: float

        Returns:
            np.ndarray: New positions of the poitns ([k, ]i, j) in the original (non-downscale) coordinates
                once the flow is applied.
                Shape: (N, dim), dtype: float
        """
        return points + self.flow_at(flow_map, points)

    def warp(self, flow_map: np.ndarray, moving: np.ndarray) -> np.ndarray:
        """Warps the moving image onto the reference using the flow map

        It warps the non-preprocessed moving image (in the original non-downscaled coordinates).

        Note:
            We only implemented backward warping (It only allows to warp the moving image onto the reference one)
            Forward warping could be implemented in a future version if needed.

        Args:
            flow_map (np.ndarray): Optical flow map. Displacement ([dk', ]di', dj') for each pixel ([k', ]i', j') in
                the downscaled coordinates.
                Shape: (dim, [D', ]H', W'), dtype: float32
            moving (np.ndarray): The moving image to warp (non-preprocessed)
                Shape: ([D, ]H, W, C), dtype: float

        Returns:
            np.ndarray: Warped image
                Shape: ([D, ]H, W, C), dtype: float

        """
        *dims, channels = moving.shape
        points = np.indices(dims, np.float32).reshape(len(dims), -1).transpose()
        points = self.transform(flow_map, points).transpose().reshape(len(dims), *dims)

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
