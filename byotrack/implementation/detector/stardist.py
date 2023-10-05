import os
import pathlib
from typing import List, Union
import warnings

import numpy as np
from stardist.models import StarDist2D  # type: ignore
import torch

import byotrack
from byotrack.api.detector.detections import relabel_consecutive
from byotrack.api.parameters import ParameterBound


class StarDistDetector(byotrack.BatchDetector):
    """Runs stardist as a detector.

    Wraps the official implementation at https://github.com/stardist/stardist, following the paper:
    Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers. Cell Detection with Star-convex Polygons.
    International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI),
    Granada, Spain, September 2018.

    We do not provide any code to train the stardist model. We expect you to already have trained the model.

    Note:
        This module requires `stardist` lib to be installed (with tensorflow). Please follow the instruction of the
        official implementation to install it.
    """

    progress_bar_description = "Detections (StarDist)"

    parameters = {
        "prob_threshold": ParameterBound(0.0, 1.0),
        "nms_threshold": ParameterBound(0.0, 1.0),
    }

    def __init__(self, model_dir: Union[str, os.PathLike], **kwargs) -> None:
        super().__init__(**kwargs)

        if self.batch_size != 1:  # TODO: Stardist do not give a straightforward implem for batch size > 1
            warnings.warn(
                "Current implementation do not support batch size greater than 1 and will iterate image by image"
            )

        path = pathlib.Path(model_dir)
        self._model = StarDist2D(None, path.name, str(path.parent))

        # Prop setter in stardist is changing type thus linting is failing...
        self.prob_threshold: float = self._model.thresholds.prob  # pylint: disable=no-member
        self.nms_threshold: float = self._model.thresholds.nms  # pylint: disable=no-member

    def detect(self, batch: np.ndarray) -> List[byotrack.Detections]:
        detections_list = []

        for i, image in enumerate(batch):
            segmentation, data = self._model.predict_instances(
                image, prob_thresh=self.prob_threshold, nms_thresh=self.nms_threshold, predict_kwargs={"verbose": 0}
            )

            detections_list.append(
                byotrack.Detections(
                    {
                        "segmentation": relabel_consecutive(torch.tensor(segmentation, dtype=torch.int32)),
                        "confidence": torch.tensor(data["prob"], dtype=torch.float32),
                        # Could use points data for position (but has been rounded to int, let's be more precise)
                        # "position": torch.tensor(data["points"], dtype=torch.float32),
                    },
                    frame_id=i,
                )
            )

        return detections_list
