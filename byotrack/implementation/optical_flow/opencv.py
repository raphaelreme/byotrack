from typing import Union

import cv2
import numpy as np

from byotrack import OpticalFlow


class OpenCVOpticalFlow(OpticalFlow):
    """Wraps Open-CV optical flow implementations

    Currently ByoTrack only supports dense optical flow which computes
    the displacement of every pixel of the reference image. And OpenCV only supports 2D images.

    Usage:

    .. code-block:: python

        import cv2
        from byotrack.implementation.optical_flow.opencv import OpenCVOpticalFlow

        # See the documentation of Open-CV for each algorithm to correctly set the parameters
        parameters = {}

        # Farneback
        optflow = OpenCVOpticalFlow(cv2.FarnebackOpticalFlow.create(**parameters))

        # TVL1 => It requires to install opencv-contrib-python (and not just opencv-python)
        optflow = OpenCVOpticalFlow(cv2.optflow.DualTVL1OpticalFlow.create(**parameters))

        # Other methods from opencv-contrib could probably be used (RLOF, DIS, ...) but have not been tested yet

    Attributes:
        optical_flow (cv2.DenseOpticalFlow): Any implementation of the cv2 DenseOpticalFlow class

    """

    def __init__(
        self,
        optical_flow: cv2.DenseOpticalFlow,
        downscale: Union[float, np.ndarray] = 2,
        blur: Union[None, float, np.ndarray] = None,
    ):
        super().__init__(downscale, blur)
        self.optical_flow = optical_flow

    def compute(self, reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
        if reference.max() > 1.0 or moving.max() > 1.0:  # Prevent dummy conversion to uint8
            raise ValueError("OpenCVOptical flow assumes normalized floating input images in [0, 1]")

        # Average channels and go to uint8 (multi channel is not supported)
        reference = np.round(reference.mean(axis=-1) * 255).astype(np.uint8)
        moving = np.round(moving.mean(axis=-1) * 255).astype(np.uint8)
        flow_map: np.ndarray = self.optical_flow.calc(reference, moving, None)  # type: ignore

        # Convert the flow map into (i, j) coordinates and (2, H', W')
        # In open-cv flow maps are (x, y) and of size (H', W', 2)
        flow_map = flow_map.transpose(2, 0, 1)
        flow_map = flow_map[::-1]
        return np.ascontiguousarray(flow_map)
