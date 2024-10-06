from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from byotrack import OpticalFlow


class SkimageOpticalFlow(OpticalFlow):
    """Wraps Scikit-Image optical flow implementations.

    It supports 3D optical flow.

    Usage:

    .. code-block:: python

        import skimage.registration  # Requires to install scikit image
        from byotrack.implementation.optical_flow.skimage import SkimageOpticalFlow

        # See the documentation of Scikit-Image for each algorithm to correctly set the parameters
        parameters = {}

        # ILK
        optflow = SkimageOpticalFlow(skimage.registration.optical_flow_ilk, parameters=parameters)

        # TVL1
        optflow = SkimageOpticalFlow(skimage.registration.optical_flow_tvl1, parameters=parameters)


    Attributes:
        method (Callable[Any, np.ndarray]): The optical flow function from skimage. In skimage, only two currently
            exists: skimage.registration.optical_flow_ilk or skimage.registration.optical_flow_tvl1.
        parameters (Dict[str, Any]): Parameters that are given to `method` as known arguments.

    """

    def __init__(
        self,
        method: Callable[..., np.ndarray],
        downscale: Union[float, np.ndarray] = 2,
        blur: Union[None, float, np.ndarray] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(downscale, blur)
        self.method = method
        self.parameters = {} if parameters is None else parameters

    def compute(self, reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
        # Average channels (multi channel is not supported)
        reference = reference.mean(axis=-1)
        moving = moving.mean(axis=-1)

        # Flow map is already at the right format (dim, [D, ]H', W') with coordinates ([k, ]i, j)
        return self.method(reference, moving, **self.parameters).astype(np.float32)
