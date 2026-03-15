from __future__ import annotations

import pathlib
import sys
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from stardist.models import StarDist2D, StarDist3D  # type: ignore[import-untyped,import-not-found]

import byotrack

if TYPE_CHECKING:
    import os

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class StarDistDetector(byotrack.BatchDetector):
    """Runs stardist as a detector.

    Wraps the official implementation at https://github.com/stardist/stardist, following the paper:
    Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers. Cell Detection with Star-convex Polygons.
    International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI),
    Granada, Spain, September 2018.

    We do not provide any code to train the stardist model. You can only use a trained or pretrained model.
    We currently only wraps the 2D model of stardist.

    Note:
        This module requires `stardist` lib to be installed (with tensorflow). Please follow the instruction of the
        official implementation to install it.

    Attributes:
        model (StarDist2D): Underlying StarDist model
        prob_threshold (float): Threshold on probability
        nms_threshold (float): Threshold for Non Maximum Suppression

    """

    progress_bar_description = "Detections (StarDist)"

    def __init__(self, model: StarDist2D | StarDist3D, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if self.batch_size != 1:  # TODO: Stardist do not give a straightforward implem for batch size > 1
            warnings.warn(
                "Current implementation do not support batch size greater than 1 and will iterate image by image",
                stacklevel=2,
            )

        self.model = model

        # Prop setter in stardist is changing type thus linting is failing...
        self.prob_threshold: float = self.model.thresholds.prob
        self.nms_threshold: float = self.model.thresholds.nms

    @override
    def detect(self, batch: np.ndarray) -> list[byotrack.Detections]:
        detections_list = []

        for i, image in enumerate(batch):
            segmentation, data = self.model.predict_instances(
                image, prob_thresh=self.prob_threshold, nms_thresh=self.nms_threshold, predict_kwargs={"verbose": 0}
            )

            detections_list.append(
                byotrack.Detections(
                    {
                        "segmentation": torch.from_numpy(segmentation.astype(np.int32)),
                        "confidence": torch.from_numpy(data["prob"].astype(np.float32)),
                        # Could use points data for position (but has been rounded to int, let's be more precise)
                        # "position": torch.tensor(data["points"], dtype=torch.float32),  # noqa: ERA001
                    },
                    frame_id=i,
                )
            )

        return detections_list

    @staticmethod
    def from_pretrained(name: str, dim=2, **kwargs: Any) -> StarDistDetector:
        """Load a pretrained StarDist from the paper.

        Args:
            name (str): A valid identifier (From the official github)
            dim (int): Image dimension (2d or 3d). Will load the model using the correct StarDist class.
                Default: 2 (Use StarDist2D)
            **kwargs: Additional detector arguments. (See `byotrack.BatchDetector`)

        Returns:
            StarDistDetector
        """
        if dim == 2:  # noqa: PLR2004
            return StarDistDetector(StarDist2D.from_pretrained(name), **kwargs)

        return StarDistDetector(StarDist3D.from_pretrained(name), **kwargs)

    @staticmethod
    def from_trained(train_dir: str | os.PathLike, dim=2, **kwargs: Any) -> StarDistDetector:
        """Load a trained StarDist from a local folder.

        Args:
            train_dir (str | os.PathLike): The training folder of the model
            dim (int): Image dimension (2d or 3d). Will load the model using the correct StarDist class.
                Default: 2 (Use StarDist2D)
            **kwargs: Additional detector arguments. (See `byotrack.BatchDetector`)

        Returns:
            StarDistDetector
        """
        path = pathlib.Path(train_dir)

        if dim == 2:  # noqa: PLR2004
            return StarDistDetector(StarDist2D(None, path.name, str(path.parent)), **kwargs)

        return StarDistDetector(StarDist3D(None, path.name, str(path.parent)), **kwargs)
