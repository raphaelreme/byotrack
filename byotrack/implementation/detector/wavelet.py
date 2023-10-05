from typing import List, Optional, Union

import cv2  # type: ignore
import numpy as np
import torch

import byotrack
from byotrack.api.detector.detections import relabel_consecutive
from byotrack.api.parameters import ParameterBound, ParameterEnum


# Set to true to follow closely the ICY implementation (slower and less precise)
FOLLOW_ICY = False


class B3SplineUWT(torch.nn.Module):
    """Undecimated Wavelet Transform with B3Spline for 2D images

    Also called A trous Wavelet Transform.

    Let J be the level of the UWD.
    Let c_0 be the original image. It computes successively dilated convolution of it.

    c_j = h_j * c_{j-1}, 1 \\le j \\le J
    w_j = c_{j-1} - c_{j}, 1 \\le j \\le J

    Where h_j is the filter at scale j which is the original filter h_1
    dilated with 2^{j - 1} 0 between each coefficient and h_1 = [1/16, 1/4, 3/8, 1/4, 1/16].

    The parameters returned are [w_1, w_2, ..., w_J, c_J].
    The original image is easily reconstructed from the parameters with c_0 = c_J + \\sum_j w_j

    Boundaries issues are handled using mirror padding.

    """

    weights = torch.tensor([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])

    def __init__(self, level: int = 3):
        """Constructor

        Args:
            level (int): Level of the Wavelet Transform. (J)

        """
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(1, 1, 5, dilation=2**i, padding=2 ** (i + 1), bias=False, padding_mode="reflect")
                for i in range(level)
            ]
        )

        weights_2d = self.weights[None, :] * self.weights[:, None]  # 2D equivalent for the B3Spline

        for layer in self.layers:
            layer.weight.requires_grad_(False)
            layer.weight[0, 0] = weights_2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the parameters of the UWD

        Args:
            x (torch.Tensor): Input images without channel dimension
                Shape: (B, H, W)

        Returns:
            torch.Tensor: Parameters [w_1, ..., w_J, c_J]
                Shape: (B, J + 1, H, W)

        """
        assert len(x.shape) == 3

        x = x[:, None, :, :]  # Shape: B x 1 x H x W

        outputs = torch.zeros((x.shape[0], len(self.layers) + 1, x.shape[-2], x.shape[-1]), device=x.device)

        for j, layer in enumerate(self.layers):
            y = layer(x)
            outputs[:, j] = (x - y)[:, 0]
            x = y

        outputs[:, -1] = x[:, 0]

        return outputs


class B3SplineUWTApprox(torch.nn.Module):
    """Approximation of Undecimated Wavelet Transform with B3Spline for 2D images

    Split the 2D convolution in two 1D convolution first alongside the rows then the columns.

    After some analysis this implementation is slower than
    the first one... even though it should be faster.

    """

    weights = torch.tensor([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])

    def __init__(self, level=3):
        """Constructor

        Args:
            level (int): Level of the Wavelet Transform. (J)

        """
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(1, 1, 5, dilation=2**i, padding=2 ** (i + 1), bias=False, padding_mode="reflect")
                for i in range(level)
            ]
        )

        for layer in self.layers:
            layer.weight.requires_grad_(False)
            layer.weight[0, 0] = self.weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the parameters of the UWD

        XXX: Batchsize != 1 is not supported yet

        Args:
            x (torch.Tensor): Input images without channel dimension
                Shape: B x H x W

        Returns:
            torch.Tensor: Parameters [w_1, ..., w_J, c_J]
                Shape: B x J + 1 x H x W

        """
        assert len(x.shape) == 3
        assert x.shape[0] == 1, "Batch size > 1 is not supported"

        x = x[0, :, None, :]  # H x 1 x W
        outputs = torch.zeros((len(self.layers) + 1, x.shape[-3], x.shape[-1]), device=x.device)

        for j, layer in enumerate(self.layers):
            y = layer(x).permute((2, 1, 0))  # W x 1 x H
            y = layer(y).permute((2, 1, 0))
            outputs[j] = (x - y)[:, 0]
            x = y

        outputs[-1] = x[:, 0]

        return outputs[None, ...]


class WaveletDetector(byotrack.BatchDetector):
    """Detection of bright spots using B3SplineUWT

    Following paper from Olivo-Marin, J.C. Extraction of spots in biological images using
    multiscale products. Pattern Recognit. 35, 1989-1996

    The algorithm is in 4 steps:

    1. UWT decomposition
    2. Scale selection
    3. Noise filtering
    4. Connected components extraction

    The multi scales behavior (choosing multiple scales) was implemented but we decided to drop it.
    It adds complexity without real advantages from our experience.

    The same algorithm is implemented in Icy Software (SpotDetector). The main differences are:

    * 2d wavelets (rather than 2 times one dimensional wavelets). It was designed to improve computations,
      but with torch no gain in time is observed. (Can be switch with FOLLOW_ICY)
    * Thresholding -> We follow the original paper.

    Attributes:
        scale (int): Scale of the wavelet coefficients used. With small scales, the detector focus on
            smaller objects.
        k (float): Noise threshold. Following the paper, the wavelet coefficients
            are filtered if coef \\le k \\sigma. (The higher the less spots you retrieve)
        min_area (float): Filter resulting spots that are too small (less than min_area pixels)
        device (torch.device): Device on which run the B3SplineUWT
            Default to cpu
        b3swt (B3SplineUWT): Undecimated wavelet transform
        **kwargs: Additional arguments for `BatchDetector` (batch_size, add_true_frames)

    Warning: The connected components used (opencv) yields segfault with too many components...

    """

    progress_bar_description = "Detections (Wavelet)"

    parameters = {
        "scale": ParameterEnum({0, 1, 2, 3, 4}),
        "k": ParameterBound(1.0, 10.0),
        "min_area": ParameterBound(0.0, 30.0),
    }

    def __init__(self, scale=2, k=3.0, min_area=10.0, device: Optional[torch.device] = None, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.k = k
        self.min_area = min_area
        self.device = device if device else torch.device("cpu")
        self.b3swt: Union[B3SplineUWT, B3SplineUWTApprox]
        if FOLLOW_ICY:
            self.b3swt = B3SplineUWTApprox(scale + 1).to(self.device)
        else:
            self.b3swt = B3SplineUWT(scale + 1).to(self.device)

    def detect(self, batch: np.ndarray) -> List[byotrack.Detections]:
        assert (
            batch.shape[-1] == 1
        ), "This detector does not support multi channel images. Please aggregate the channels first"

        # Extract wavelet coefficient at the given scale
        inputs = torch.tensor(batch, dtype=torch.float32)[..., 0].to(self.device)
        coefficients: torch.Tensor = self.b3swt(inputs)[:, self.scale]

        # Thresholding. Shape: (B, H, W)
        coefficients[coefficients < self.compute_threshold(coefficients)] = 0
        coefficients[coefficients > 0] = 255

        # Instance segmentation by connected components
        # (Not found a quick and efficient torch implem with 4-way connectivity)
        detections_list = []
        mask: np.ndarray
        for i, mask in enumerate(coefficients.cpu().to(torch.uint8).numpy()):  # Unefficient but fine.
            _, segmentation, stats, _ = cv2.connectedComponentsWithStats(
                mask, np.zeros(mask.shape, dtype=np.uint16), connectivity=4, ltype=cv2.CV_16U  # type: ignore
            )

            # Delete too small detections
            area = stats[:, 4]
            segmentation[(area < self.min_area)[segmentation]] = 0

            detections_list.append(
                byotrack.Detections(
                    {
                        "segmentation": relabel_consecutive(torch.tensor(segmentation.astype(np.int32))),
                        # "confidence": put areas instead of deleting them ?
                    },
                    frame_id=i,
                )
            )

        return detections_list

    def compute_threshold(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Compute threshold for the UWT coefficients

        Note: One could use MAD approx of \\sigma rather than std (or Icy implem) but it's almost equivalent
        (And k is a parameter to tune so it truly is)

        Args:
            coefficients (torch.Tensor): Coefficients of the UWT
                Shape: (..., H, W)

        Returns:
            torch.Tensor: Threshold for each scale
                Shape: (..., 1, 1)

        """
        return coefficients.std(dim=(-1, -2), keepdim=True) * self.k
