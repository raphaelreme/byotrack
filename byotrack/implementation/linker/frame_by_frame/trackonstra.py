import dataclasses
from typing import List, Tuple, Dict, Union, Sequence, Any, Optional
import logging
import warnings

import numpy as np
import torch
import tqdm.auto as tqdm

from trackastra.model import Trackastra  # type: ignore
from trackastra.model.predict import predict_windows  # type: ignore
from trackastra.utils import normalize  # type: ignore
from trackastra.data import (  # type: ignore
    get_features,
    build_windows,
)
from trackastra.model import TrackingTransformer

import byotrack
from byotrack.implementation.linker.frame_by_frame.base import (
    AssociationMethod,
    FrameByFrameLinkerParameters,
    FrameByFrameLinker,
)
from byotrack.api.optical_flow.optical_flow import OpticalFlow
from byotrack.api.features_extractor import FeaturesExtractor


def dict_builder(
    nodes: List[Dict[str, Any]], weights: Tuple[Tuple[Tuple[int, int], float]]
) -> Dict[Tuple[int, int, int, int], float]:
    """
    Build the dictionnary of possible links from the Trackastra's predictions.

    Args :
        nodes : The list of the graph's nodes (detections),
            shape of node {'id' ,'coords' ,'time' ,'label' }
        weights : The list of all the graph's edges with their weight
            shape of weight ((node1_id,node2_id),weight)

    Returns :
        Dictionnary of possible links between detections,
            shape of link {(detection1_time,detection1_id_on_frame,detection2_time,detection2_id_on_frame)
    """
    cost_dict = {}
    node_by_id = {node["id"]: node for node in nodes}
    for (id1, id2), weight in weights:
        node1 = node_by_id[id1]
        node2 = node_by_id[id2]
        cost_dict[(node1["time"], node1["label"] - 1, node2["time"], node2["label"] - 1)] = -np.log(weight)
    return cost_dict


class TrackastraFlex(Trackastra):
    """ "
    Subclass of Trackastra to be able to modify delta_t
    """

    def __init__(
        self,
        transformer: TrackingTransformer,
        train_args: Dict[str, Any],
        delta_t: int = 4,
        intra_weight: float = 0,
        device=None,
    ):
        super().__init__(transformer, train_args, device)
        self.delta_t = delta_t
        self.intra_weight = intra_weight
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _predict(
        self, imgs: np.ndarray, masks: np.ndarray, edge_threshold: float = 0.05, n_workers: int = 0, progbar_class=None
    ):
        """Same function as the original _predict but it calls predict_windows with the model delta_t"""

        self.logger.info("Predicting weights for candidate graph")
        imgs = normalize(imgs)
        self.transformer.eval()
        features = get_features(
            detections=masks,
            imgs=imgs,
            ndim=self.transformer.config["coord_dim"],
            n_workers=n_workers,
            progbar_class=lambda iterable, **kwargs: tqdm.tqdm(iterable, **kwargs, dynamic_ncols=True),
        )
        self.logger.info("Building windows")
        windows = build_windows(
            features,
            window_size=self.transformer.config["window"],
            progbar_class=lambda iterable, **kwargs: tqdm.tqdm(iterable, **kwargs, dynamic_ncols=True),
        )
        self.logger.info("Predicting windows")
        predictions = predict_windows(
            windows=windows,
            features=features,
            model=self.transformer,
            intra_window_weight=self.intra_weight,
            edge_threshold=edge_threshold,
            spatial_dim=masks.ndim - 1,
            progbar_class=lambda iterable, **kwargs: tqdm.tqdm(iterable, **kwargs, dynamic_ncols=True),
            delta_t=self.delta_t,
        )

        return predictions


@dataclasses.dataclass
class TrackaOnStraParameters(FrameByFrameLinkerParameters):
    """Parameters of TrackaOnStraLinker

    Attributes:
        association_threshold (float) : This is the minimum likelihood to consider a link.
            The default value is provided by Trackstra.
            Default: 0.05
        positional_cutoff (float): It defines the threshold on the euclidean distance used
            not to link tracks with detections.
            The default value is provided by Trackastra.
            We found that reducing it may improve performance.
            Default: 256
        n_valid (int): Number associated detections required to validate the track after its creation.
            Default: 1
        n_gap (int): Number of consecutive frames without association before the track termination.
            Default: 3
        association_method (AssociationMethod): The frame-by-frame association to use. See `AssociationMethod`.
            It can be provided as a string. (Choice: GREEDY, [SPARSE_]OPT_HARD, [SPARSE_]OPT_SMOOTH)
            Default: OPT_SMOOTH
        anisotropy (Tuple[float, float, float]): Anisotropy of images (Ratio of the pixel sizes
            for each axis, depth first). This will be used to scale distances.
            Default: (1., 1., 1.)
        split_factor (float): Allow splitting of tracks, using a second association step.
            The association threshold in this case is `split_factor * association_threshold`.
            Default: 0.0 (No splits)
        merge_factor (float): Allow merging of tracks, using a second association step.
            The association threshold in this case is `merge_factor * association_threshold`.
            Default: 0.0 (No merges)

    """

    def __init__(
        self,
        association_threshold: float = 0.05,
        positional_cutoff: float = 256,
        n_valid=1,
        n_gap=3,
        association_method: Union[str, AssociationMethod] = AssociationMethod.OPT_SMOOTH,
        anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        split_factor: float = 0.0,
        merge_factor: float = 0.0,
    ):
        super().__init__(  # pylint: disable=duplicate-code
            association_threshold=association_threshold,
            n_valid=n_valid,
            n_gap=n_gap,
            association_method=association_method,
            anisotropy=anisotropy,
            split_factor=1 if split_factor > 0 else 0,
            merge_factor=merge_factor,
        )
        self.positional_cutoff = positional_cutoff


class TrackaOnStraLinker(FrameByFrameLinker):
    """Linker using Trackastra associating costs.
    It is not yet an Online linker since the linker has to be setup with the whole video and
    all the detections before it can run.
    See `TrackaOnStraParamaters` for the other attributes.

    Attributes:
        specs (TrackaOnStraParameters): Parameters specifications of the algorithm.
            See `TrackaOnStraParameters`.
        model (NewTrackastra) : Model of Trackastra used to compute the association costs.
        cost_dict (Dict) : Dictionnary of all the possible link between detections associated with its cost.

    """

    progress_bar_description = "TrackaOnStra linking"

    def __init__(
        self,
        specs: TrackaOnStraParameters,
        model: Optional[TrackastraFlex] = None,
        optflow: Optional[OpticalFlow] = None,
        features_extractor: Optional[FeaturesExtractor] = None,
    ) -> None:
        super().__init__(specs, optflow, features_extractor)
        self.specs: TrackaOnStraParameters
        if model is None:
            self.model = TrackastraFlex.from_pretrained("ctc")
        else:
            self.model = model
        self.model.transformer.config["spatial_pos_cutoff"] = self.specs.positional_cutoff
        self.cost_dict: dict[Tuple, float]
        self.model.delta_t = self.specs.n_gap + 1
        if self.specs.n_gap == 0:  # If n_gap =0 window size of 4 seems to be a bit better
            self.model.transformer.config["window"] = 4
        elif self.specs.n_gap <= 3:
            self.model.transformer.config["window"] = 5
        else:
            warnings.warn("delta_t is too big, results might not be good")
            self.model.transformer.config["window"] = self.specs.n_gap + 2
        if optflow is not None:
            warnings.warn("OpticalFlow is not supported by this linker, it will be ignored")
        if features_extractor is not None:
            warnings.warn("Features Extractor is not supported by this linker, it will be ignored")
        self.active_positions = torch.zeros(0, 2)

    def setup(self, video: Union[Sequence[np.ndarray], np.ndarray], detections_sequence: Sequence[byotrack.Detections]):
        """
        Setup the linker offline by computing the costs with Trackastra's predictions and store it into cost_dict

        Args:
            video (Sequence[np.ndarray] | np.ndarray): Sequence of T frames (array).
                Each array is expected to have a shape ([D, ]H, W, C)
            detections_sequence (Sequence[byotrack.Detections]): Detections for each frame
                Detections is expected for each frame of the video, in the same order.
                (Note that for a given frame, the Detections can be empty)

        """
        # Convert Byotrack datas for Trackastra

        vid = np.asarray(video)[..., 0]
        masks_list = []
        for detection in detections_sequence:
            mask = detection.segmentation.cpu().numpy()
            masks_list.append(mask)
        masks = np.stack(masks_list)

        # Then compute the cost
        predictions = self.model._predict(  # pylint: disable=W0212
            vid, masks, edge_threshold=self.specs.association_threshold
        )

        nodes = predictions["nodes"]
        weights = predictions["weights"]
        self.cost_dict = dict_builder(nodes, weights)

    def motion_model(self) -> None:
        pass

    def reset(self, dim=2) -> None:
        super().reset(dim)
        self.active_positions = torch.zeros(0, dim)

    def post_association(self, _: np.ndarray, detections: byotrack.Detections, active_mask: torch.Tensor):

        self.active_positions[self._links[:, 0]] = detections.position[self._links[:, 1]]

        # Merge still active positions and new ones
        self.active_positions = torch.cat(
            (self.active_positions[active_mask], detections.position[self._unmatched_detections])
        )

        self.all_positions.append(self.active_positions.clone())
        for i, track in enumerate(self.active_tracks):
            if track.detection_ids[-1] == -1:
                self.all_positions[-1][i, :] = torch.nan

    def cost(self, _: np.ndarray, detections: byotrack.Detections) -> Tuple[torch.Tensor, float]:
        """Compute the association cost between active tracks and detections.

        Args:
            detections (byotrack.Detections) : Detections on a given frame.

        Returns:
            torch.Tensor: The cost matrix between active tracks and detections
                Shape: (n_tracks, n_dets), dtype: float
            float: The association threshold to use. Simply returns -ln(association_treshold)

        """
        n = len(self.active_tracks)
        m = detections.length
        cost = torch.full((n, m), torch.inf)
        for index, track in enumerate(self.active_tracks):
            for index_det in range(m):
                for i in range(1, self.specs.n_gap + 2):
                    try:
                        edge = (
                            self.frame_id - i,
                            track.detection_ids[-i],
                            self.frame_id,
                            index_det,
                        )
                        if edge in self.cost_dict:
                            cost[index, index_det] = float(self.cost_dict[edge])
                            break
                    except IndexError:
                        break
        return (cost, -np.log(self.specs.association_threshold))
