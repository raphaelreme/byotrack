import time
from typing import List, Optional, Union
import warnings

import numba  # type: ignore
import numpy as np
import scipy.ndimage as ndi  # type: ignore
import torch

import byotrack


class B3SplineUWT(torch.nn.Module):
    """Undecimated Wavelet Transform with B3Spline for nD images.

    Also called A trous Wavelet Transform.

    Let J be the level of the UWD.
    Let c_0 be the original image. It computes successively dilated convolution of it.

    c_j = h_j * c_{j-1}, 1 \\le j \\le J
    w_j = c_{j-1} - c_{j}, 1 \\le j \\le J

    Where h_j is the filter at scale j which is the original filter h_1
    dilated with 2^{j - 1} 0 between each coefficient and h_1 = [1/16, 1/4, 3/8, 1/4, 1/16].

    The parameters returned are [w_1, w_2, ..., w_J, c_J], or only [w_J] if `return_all` is False
    The original image is easily reconstructed from the parameters with c_0 = c_J + \\sum_j w_j

    Boundaries issues are handled using mirror padding.

    """

    weights = torch.tensor([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])

    def __init__(self, level=3, dim=2, return_all=True):
        """Constructor

        Args:
            level (int): Level of the Wavelet Transform. (J)
            dim (int): Dimension of the UWT.
                Default: 2
            return_all (bool): Return all parameters and the remaining image (w_1, ... w_J, c_J)
                If False it only returns the parameters at the final level (w_J)
                Default: True

        """
        super().__init__()
        self.return_all = return_all
        self.dim = dim

        conv_builder: type
        if dim == 1:
            conv_builder = torch.nn.Conv1d
        elif dim == 2:
            conv_builder = torch.nn.Conv2d
        elif dim == 3:
            conv_builder = torch.nn.Conv3d
        else:
            raise NotImplementedError("B3SplineUWT are implemented only for 1d, 2d and 3d")

        self.layers = torch.nn.ModuleList(
            [
                conv_builder(1, 1, 5, dilation=2**i, padding=2 ** (i + 1), bias=False, padding_mode="reflect")
                for i in range(level)
            ]
        )

        # Construct multi dim weights
        weights = self.weights
        for _ in range(dim - 1):
            weights = weights[..., None] * self.weights

        for layer in self.layers:
            layer.weight.requires_grad_(False)
            layer.weight[0, 0] = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the parameters of the UWD

        Args:
            x (torch.Tensor): Input multidimensional images without channel dimension
                Shape: (B, [D, ]H, W)

        Returns:
            torch.Tensor: Parameters [w_1, ..., w_J, c_J] or only w_J
                Shape: (B, J + 1, [D, ]H, W) or (B, [D, ]H, W) if `return_all` is False

        """
        batch, *spatial = x.shape
        assert len(spatial) == self.dim, f"The module was constructed for {self.dim}D images. Input is {len(spatial)}D."

        x = x[:, None]  # Add a channel dim: (B, 1, [D, ]H, W)
        y = x

        if self.return_all:
            outputs = torch.zeros((batch, len(self.layers) + 1, *spatial), device=x.device)

        for j, layer in enumerate(self.layers):
            x = y
            y = layer(y)

            if self.return_all:
                outputs[:, j] = (x - y)[:, 0]

        if self.return_all:
            outputs[:, -1] = y[:, 0]

            return outputs

        return (x - y)[:, 0]


class B3SplineUWTApprox1(torch.nn.Module):
    """Approximation of Undecimated Wavelet Transform with B3Spline for nD images.

    Split the nD convolution in n 1D convolution along each axis (as done in the original paper).
    Though it reduces the FLOPs it can lead to slower runtime (because of some torch optims).

    Depending on the gpu, the cudnn/cuda kernels, the pytorch version (and so on), it can sometimes
    be much faster than the nD convolution counterpart (especially in 3D).

    In this implementation, we rely on torch.nn.Conv1D, and is able to handle any number of dimension n.
    """

    weights = torch.tensor([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])

    def __init__(self, level=3, dim=2, return_all=True):
        """Constructor

        Args:
            level (int): Level of the Wavelet Transform. (J)
            dim (int): Dimension of the UWT.
                Default: 2
            return_all (bool): Return all parameters and the remaining image (w_1, ... w_J, c_J)
                If False it only returns the parameters at the final level (w_J)
                Default: True

        """
        super().__init__()
        self.return_all = return_all
        self.dim = dim

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

        Args:
            x (torch.Tensor): Input images without channel dimension
                Shape: (B, [D, ]H, W)

        Returns:
            torch.Tensor: Parameters [w_1, ..., w_J, c_J] or only [w_J]
                Shape: (B, J + 1, [D, ]H, W) or (B, [D, ]H, W) if `return_all` is False

        """
        batch, *spatial = x.shape
        assert len(spatial) == self.dim, f"The module was constructed for {self.dim}D images. Input is {len(spatial)}D."

        x = x[:, None]  # Add a channel dim: (B, 1, [D, ]H, W)
        y = x

        if self.return_all:
            outputs = torch.zeros((batch, len(self.layers) + 1, *spatial), device=x.device)

        for j, layer in enumerate(self.layers):
            x = y
            for axis in range(self.dim):
                # Transpose in ((B * S\{axis}, 1, S[axis])
                y = y.permute((0, *range(2, 2 + axis), *range(3 + axis, 2 + self.dim), 1, 2 + axis))
                y = layer(y.reshape(-1, 1, spatial[axis])).reshape(y.shape)
                # Transpose back
                y = y.permute((0, self.dim, *range(1, 1 + axis), 1 + self.dim, *range(1 + axis, self.dim)))

            if self.return_all:
                outputs[:, j] = (x - y)[:, 0]

        if self.return_all:
            outputs[:, -1] = y[:, 0]

            return outputs

        return (x - y)[:, 0]


class B3SplineUWTApprox2(torch.nn.Module):
    """Approximation of Undecimated Wavelet Transform with B3Spline for nD images.

    Split the nD convolution in n 1D convolution along each axis (as done in the original paper).
    Though it reduces the FLOPs it can lead to slower runtime (because of some torch optims).

    Depending on the gpu, the cudnn/cuda kernels, the pytorch version (and so on), it can sometimes
    be much faster than the nD convolution counterpart (especially in 3D).

    In this implementation, we rely on torch.nn.ConvnD with a reduced kernel.
    """

    weights = torch.tensor([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])

    def __init__(self, level=3, dim=2, return_all=True):
        """Constructor

        Args:
            level (int): Level of the Wavelet Transform. (J)
            dim (int): Dimension of the UWT.
                Default: 2
            return_all (bool): Return all parameters and the remaining image (w_1, ... w_J, c_J)
                If False it only returns the parameters at the final level (w_J)
                Default: True

        """
        super().__init__()
        self.return_all = return_all
        self.dim = dim

        conv_builder: type
        if dim == 1:
            conv_builder = torch.nn.Conv1d
        elif dim == 2:
            conv_builder = torch.nn.Conv2d
        elif dim == 3:
            conv_builder = torch.nn.Conv3d
        else:
            raise NotImplementedError("B3SplineUWT are implemented only for 1d, 2d and 3d")

        layers = []
        for i in range(level):
            sub_layers = []
            for j in range(dim):
                conv = conv_builder(
                    1,
                    1,
                    tuple(5 if k == j else 1 for k in range(dim)),  # type: ignore
                    dilation=tuple(2**i if k == j else 1 for k in range(dim)),  # type: ignore
                    padding=tuple(2 ** (i + 1) if k == j else 0 for k in range(dim)),  # type: ignore
                    bias=False,
                    padding_mode="reflect",
                )
                conv.weight.requires_grad_(False)
                conv.weight.view(5)[:] = self.weights
                sub_layers.append(conv)

            layers.append(torch.nn.Sequential(*sub_layers))

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the parameters of the UWD

        Args:
            x (torch.Tensor): Input images without channel dimension
                Shape: (B, [D, ]H, W)

        Returns:
            torch.Tensor: Parameters [w_1, ..., w_J, c_J] or only [w_J]
                Shape: (B, J + 1, [D, ]H, W) or (B, [D, ]H, W) if `return_all` is False

        """
        batch, *spatial = x.shape
        assert len(spatial) == self.dim, f"The module was constructed for {self.dim}D images. Input is {len(spatial)}D."

        x = x[:, None]  # Add a channel dim: (B, 1, [D, ]H, W)
        y = x

        if self.return_all:
            outputs = torch.zeros((batch, len(self.layers) + 1, *spatial), device=x.device)

        for j, layer in enumerate(self.layers):
            x = y
            y = layer(y)

            if self.return_all:
                outputs[:, j] = (x - y)[:, 0]

        if self.return_all:
            outputs[:, -1] = y[:, 0]

            return outputs

        return (x - y)[:, 0]


@numba.njit(cache=True)
def filter_small_objects(segmentation: np.ndarray, min_area: float) -> None:
    """Filter small instances from the segmentation in place

    Args:
        segmentation (np.ndarray): Segmentation mask to filtered inplace
            Shape ([D, ]H, W), dtype: integer
        min_area (float): Minimum number of pixels to be kept in the segmentation.

    """
    segmentation = segmentation.reshape(-1)
    area = np.zeros(segmentation.max(), np.uint)

    for i in range(segmentation.size):
        instance = segmentation[i] - 1
        if instance != -1:
            area[instance] += 1

    to_delete = area < min_area

    for i in range(segmentation.size):
        instance = segmentation[i] - 1
        if instance != -1:
            if to_delete[instance]:
                segmentation[i] = 0


class WaveletDetector(byotrack.BatchDetector):
    """Detection of bright spots using B3SplineUWT

    Following paper from Olivo-Marin, J.C. Extraction of spots in biological images using
    multiscale products. Pattern Recognit. 35, 1989-1996

    It supports 2D and 3D videos.

    The algorithm is in 4 steps:

    1. UWT decomposition
    2. Scale selection
    3. Noise filtering
    4. Connected components extraction

    The multi scales behavior (choosing multiple scales) was implemented but we decided to drop it.
    It adds complexity without real advantages from our experience.

    The same algorithm is implemented in Icy Software (SpotDetector). The main differences are:

    * nd wavelets (rather than n times one dimensional wavelets). It was designed to improve computations,
      but with torch no gain in time is observed in 2D. This can be switched either by calling `optimize`
      that will try to find the fastest option for your case, or manually by modifying the `b3swt` parameter.
    * Thresholding -> We follow the original paper using k times the std

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

    """

    progress_bar_description = "Detections (Wavelet)"

    def __init__(self, scale=2, k=3.0, min_area=10.0, device: Optional[torch.device] = None, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.k = k
        self.min_area = min_area
        self.device = device if device else torch.device("cpu")
        self.b3swt: Union[B3SplineUWT, B3SplineUWTApprox1, B3SplineUWTApprox2]
        self.b3swt = B3SplineUWT(scale + 1, return_all=False).to(self.device)

    def detect(self, batch: np.ndarray) -> List[byotrack.Detections]:
        assert (
            batch.shape[-1] == 1
        ), "This detector does not support multi channel images. Please aggregate the channels first"

        if batch.ndim - 2 != self.b3swt.dim:  # Wrong dim, let's reinit the module
            self.b3swt = type(self.b3swt)(self.scale + 1, dim=batch.ndim - 2, return_all=False).to(self.device)

        # Extract wavelet coefficient at the given scale
        inputs = torch.tensor(batch, dtype=torch.float32)[..., 0].to(self.device)
        coefficients: torch.Tensor = self.b3swt(inputs)

        # Thresholding. Shape: (B, [D, ]H, W)
        masks = (coefficients >= self.compute_threshold(coefficients)).cpu().numpy()

        # Instance segmentation by connected components
        detections_list = []
        for i, mask in enumerate(masks):
            segmentation = np.zeros(mask.shape, dtype=np.int32)
            ndi.label(mask, output=segmentation)  # 1-hop (4-ways in 2D) connected components by default

            if self.min_area:
                filter_small_objects(segmentation, self.min_area)

            detections_list.append(
                byotrack.Detections(
                    {
                        "segmentation": torch.tensor(segmentation),
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
                Shape: (B, [D, ]H, W)

        Returns:
            torch.Tensor: Threshold for each scale
                Shape: (B, 1, ..., 1)

        """
        return coefficients.std(dim=tuple(range(1, coefficients.ndim)), keepdim=True) * self.k

    def optimize(  # pylint: disable=too-many-statements,too-many-branches,too-many-locals
        self, frames: np.ndarray, repeat=5, warm_up=True, splines=(B3SplineUWT, B3SplineUWTApprox1, B3SplineUWTApprox2)
    ) -> "WaveletDetector":
        """Find the fastest configuration for the model on the given frames

        This is mainly designed for 3D videos, where 3D convolutions are heavy and poorly optimized with
        a dilation > 1 (for scales > 0). In particular, we observed cudnn kernels running 10 times faster
        on some gpus when determistic was set to False for 3D conv.

        epending on your hardware, kernels and pytorch version, one solution may be better than the other.
        This allows you to test most of the configuration and use the fastest one.

        Warning:
            With large 3D images, B3SplineUWTApprox1 may not work (it converts the spatial axis into batch axis,
            but conv1D do not support very large batch size. If you are in this case, you may disable it by
            changing the `splines` argument to `(B3SplineUWT, B3SplineUWTApprox2)`.

        Args:
            frames (np.ndarray): Frames of the video on which to test.
                Shape: (B, [D, ]H, W, C), dtype: float
            repeat (int): Number of time to repeat the computation to measure the timings.
                Default: 5
            warm_up (bool): Warm up each model before measuring time.
                Default: True
            splines (tuple): Implementation of B3SplineUWT to test. Reduce this list if one of the implementation
                is too long to run for instance.
                Default: (B3SplineUWT, B3SplineUWTApprox1, B3SplineUWTApprox2)

        Returns:
            WaveletDetector: self, with the best found b3swt. It may also modify pytorch backend for convolution.

        """
        inputs = torch.tensor(frames, dtype=torch.float32)[..., 0].to(self.device)
        models = [module(self.scale + 1, dim=inputs.ndim - 1, return_all=False).to(self.device) for module in splines]

        best_time = float("inf")
        best_model = models[0]

        if self.device.type == "cpu":
            for model in models:
                if warm_up:
                    model(inputs)

                total_time = 0.0
                for _ in range(repeat):
                    t = time.time()
                    model(inputs)
                    t = time.time() - t

                    if t > 5 * best_time:
                        # Stop early if it takes too long before the best one
                        break

                    total_time += t

                if t > 5 * best_time:
                    print(f"CPU: {model.__class__.__name__} computed in {t} => skip")
                    continue

                total_time /= repeat

                print(f"CPU: {model.__class__.__name__} compute in avg of {total_time}")

                if total_time < best_time:
                    best_time = total_time
                    best_model = model

        else:
            old_cudnn, old_deterministic = torch.backends.cudnn.enabled, torch.backends.cudnn.deterministic
            best_cudnn = old_cudnn
            best_deterministic = old_deterministic
            for cudnn, deterministic in [(True, True), (True, False), (False, False)]:
                torch.backends.cudnn.enabled = cudnn
                torch.backends.cudnn.deterministic = deterministic

                for model in models:
                    if warm_up:
                        model(inputs)

                    total_time = 0.0
                    for _ in range(repeat):
                        t = time.time()
                        model(inputs).cpu()
                        t = time.time() - t

                        if t > 5 * best_time:
                            # Stop early if it takes too long before the best one
                            break

                        total_time += t

                    if t > 5 * best_time:
                        print(
                            f"Cuda (cudnn: {cudnn}, deterministic: {deterministic})"
                            + f": {model.__class__.__name__} computed in {t} => skip"
                        )
                        continue

                    total_time /= repeat

                    print(
                        f"Cuda (cudnn: {cudnn}, deterministic: {deterministic})"
                        + f": {model.__class__.__name__} compute in avg of {total_time}"
                    )

                    if total_time < best_time:
                        best_time = total_time
                        best_model = model
                        best_cudnn = cudnn
                        best_deterministic = deterministic

            if best_cudnn and not old_cudnn:
                warnings.warn(f"Enabling cudnn backend (determinisitc={best_deterministic})")
            if not best_cudnn and old_cudnn:
                warnings.warn("Disabling cudnn backend")
            if old_deterministic != best_deterministic:
                warnings.warn(f"Switching cudnn deterministic to {deterministic}")

            torch.backends.cudnn.enabled = best_cudnn
            torch.backends.cudnn.deterministic = best_deterministic

        print(f"Selecting {best_model.__class__.__name__}")
        self.b3swt = best_model  # type: ignore

        return self
