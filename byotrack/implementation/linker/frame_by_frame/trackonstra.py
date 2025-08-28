import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import torch
import tqdm.auto as tqdm

# pylint: disable=import-error
from trackastra.data import get_features, build_windows  # type: ignore
from trackastra.model import Trackastra, TrackingTransformer  # type: ignore
from trackastra.model.predict import predict_windows  # type: ignore
from trackastra.utils import normalize  # type: ignore

# pylint: enable=import-error


import byotrack
from .base import AssociationMethod, FrameByFrameLinker, FrameByFrameLinkerParameters


def build_cost_dict(
    nodes: List[Dict[str, Any]], weights: Iterable[Tuple[Tuple[int, int], float]]
) -> Dict[Tuple[int, int, int], Dict[int, float]]:
    """Build the cost dictionnary from Trackastra data format.

    It converts the feasible tracking graph predicted by trackastra into a mapping from edge to cost.
    Where an edge is a link between two detections (node). It converts the probability weight into a
    cost = -log weight.

    Args:
        nodes (List[Dict[str, Any]]): Nodes of the graph (detections), with their features
            Format: { "id": int, "coords": (float, float), "time": int, "label": int}
        weights (Iterable[Tuple[Tuple[int, int], float]]): Considered edges, with their weight
            Format: [((node_id, node_id_2), weight), ...]

    Returns:
        Dict[Tuple[int, int, int],Dict[int,float]]: Cost dictionnary of the feasible tracking graph.
        For every (frame1,node1,frame2) there is a dictionnary with key node2 and the value the cost of the edge.
    """
    cost_dict: Dict[Tuple[int, int, int], Dict[int, float]] = {}
    node_by_id = {node["id"]: node for node in nodes}
    for (id_1, id_2), weight in weights:
        node_1 = node_by_id[id_1]
        node_2 = node_by_id[id_2]
        key = (node_1["time"], node_1["label"] - 1, node_2["time"])
        if key not in cost_dict:
            cost_dict[key] = {}
        cost_dict[key][node_2["label"] - 1] = float(-np.log(weight))
    return cost_dict


class TrackastraFlex(Trackastra):  # pylint: disable=too-few-public-methods
    """Trackastra with ability to modify delta_t"""

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

    def _predict(  # pylint: disable=unused-argument
        self, imgs: np.ndarray, masks: np.ndarray, edge_threshold: float = 0.05, n_workers: int = 0, progbar_class=None
    ):
        """Same function as the original _predict but it calls predict_windows with the model delta_t"""

        print("Predicting weights for candidate graph")  # Use print as ByoTrack do not use logging yet
        imgs = normalize(imgs)
        self.transformer.eval()
        features = get_features(
            detections=masks,
            imgs=imgs,
            ndim=self.transformer.config["coord_dim"],
            n_workers=n_workers,
            progbar_class=lambda iterable, **kwargs: tqdm.tqdm(iterable, **kwargs, dynamic_ncols=True),
        )
        print("Building windows")
        windows = build_windows(
            features,
            window_size=self.transformer.config["window"],
            progbar_class=lambda iterable, **kwargs: tqdm.tqdm(iterable, **kwargs, dynamic_ncols=True),
        )
        print("Predicting windows")
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
class TrackOnStraParameters(FrameByFrameLinkerParameters):
    """Parameters of TrackOnStraLinker

    Attributes:
        association_threshold (float) : Minimum probability to consider a link.
            By default, we use the value from Trackstra.
            Default: 0.05
        positional_cutoff (float): Defines an Euclidean threshold on links.
            We use the default value provided by Trackastra. Tuning it improves performance.
            Default: 256.0
        n_valid (int): Number associated detections required to validate the track after its creation.
            Default: 2
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
        *,
        association_threshold: float = 0.05,
        positional_cutoff: float = 256.0,
        n_valid=2,
        n_gap=3,
        association_method: Union[str, AssociationMethod] = AssociationMethod.OPT_SMOOTH,
        anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        split_factor: float = 0.0,
        merge_factor: float = 0.0,
    ):
        super().__init__(  # pylint: disable=duplicate-code
            association_threshold=association_threshold,
            n_gap=n_gap,
            n_valid=n_valid,
            association_method=association_method,
            anisotropy=anisotropy,
            split_factor=1 if split_factor > 0 else 0,
            merge_factor=merge_factor,
        )
        self.positional_cutoff = positional_cutoff

    positional_cutoff: float = 256.0


class TrackOnStraLinker(FrameByFrameLinker):
    """Online TrackAstra

    It uses a trained TrackAstra model to predict linking costs. But it replaces the graph optimization from TrackAstra
    by our online FrameByFrame linking. This allows a simple support for false negative detections that TrackAstra
    do not support by itself.

    Warning: This implementation is not yet Online. Indeed the linker has to be setup with
             the full video and detections_sequence before being usable.

    Note:
        This implementation requires trackastra. (pip install trackastra)
        trackastra is only available for python >= 3.10

    See `FrameByFrameLinker` for the other attributes.

    Attributes:
        specs (TrackOnStraParameters): Parameters specifications of the algorithm.
            See `TrackOnStraParameters`.
        model (TrackastraFlex): Model of Trackastra used to compute the association costs.
        cost_dict (Dict[Tuple[int, int, int, int], float]): Cost dictionnary of the feasible tracking graph.

    """

    progress_bar_description = "TrackaOnStra linking"

    def __init__(
        self,
        specs: TrackOnStraParameters,
        model: Optional[TrackastraFlex] = None,
        optflow: Optional[byotrack.OpticalFlow] = None,
        features_extractor: Optional[byotrack.FeaturesExtractor] = None,
    ) -> None:
        super().__init__(specs, optflow, features_extractor)
        self.specs: TrackOnStraParameters
        if model is None:
            self.model = TrackastraFlex.from_pretrained("ctc")
        else:
            self.model = model

        self.model.transformer.config["spatial_pos_cutoff"] = self.specs.positional_cutoff
        self.model.delta_t = self.specs.n_gap + 1
        if self.specs.n_gap == 0:  # If n_gap =0 window size of 4 seems to be a bit better
            self.model.transformer.config["window"] = 4
        elif self.specs.n_gap <= 3:
            self.model.transformer.config["window"] = 5
        else:
            warnings.warn("delta_t is too big, results might not be good")
            self.model.transformer.config["window"] = self.specs.n_gap + 2

        # Initialize empty cost_dict
        self.cost_dict: Dict[Tuple[int, int, int], Dict[int, float]] = {}

        if optflow is not None:
            warnings.warn("OpticalFlow is not supported by this linker, it will be ignored")
        if features_extractor is not None:
            warnings.warn("Features Extractor is not supported by this linker, it will be ignored")

    def setup(self, video: Union[Sequence[np.ndarray], np.ndarray], detections_sequence: Sequence[byotrack.Detections]):
        """Offline setup of the linker by computing all the linking costs with Trackastra.

        Linking costs are stored into `cost_dict`.

        This function needs to be called on each video for the linking to be able to run.

        Args:
            video (Sequence[np.ndarray] | np.ndarray): Sequence of T frames (array).
                Each frame is expected to have a shape ([D, ]H, W, C)
            detections_sequence (Sequence[byotrack.Detections]): Detections for each frame
                Detections is expected for each frame of the video, in the same order.
                (Note that for a given frame, the Detections can be empty)

        """
        # Convert Byotrack format to TrackAstra
        imgs = np.asarray(video)[..., 0]

        masks_list = []
        for detection in detections_sequence:
            mask = detection.segmentation.cpu().numpy()
            masks_list.append(mask)
        masks = np.stack(masks_list)

        # Then compute the cost
        predictions = self.model._predict(  # pylint: disable=protected-access
            imgs, masks, edge_threshold=self.specs.association_threshold
        )

        nodes = predictions["nodes"]
        weights = predictions["weights"]

        # And convert back to a more usable format
        self.cost_dict = build_cost_dict(nodes, weights)

    def motion_model(self) -> None:
        pass  # No motion to model

    def post_association(self, _: np.ndarray, detections: byotrack.Detections, active_mask: torch.Tensor):
        # Simply store the position of the detection
        active_positions = torch.full((len(self.active_tracks), detections.dim), torch.nan)

        for i, track in enumerate(self.active_tracks):
            j = track.detection_ids[-1]
            if j != -1:
                active_positions[i] = detections.position[j]

        self.all_positions.append(active_positions)

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
            # Find most recent valid detection
            for tau in range(1, self.specs.n_gap + 2):
                if len(track.detection_ids) >= tau and track.detection_ids[-tau] != -1:
                    frame_prev = self.frame_id - tau
                    det_prev = track.detection_ids[-tau]

                    # Direct lookup: all possible detections from cost_dict
                    edges = self.cost_dict.get((frame_prev, det_prev, self.frame_id), {})
                    for index_det, edge_cost in edges.items():
                        cost[index, index_det] = edge_cost
                    break

        return (cost, -np.log(self.specs.association_threshold))
