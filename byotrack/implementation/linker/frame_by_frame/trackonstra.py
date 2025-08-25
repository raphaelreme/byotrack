import dataclasses
from typing import List, Tuple, Dict, Union, Sequence
import logging
import warnings

import numpy as np
import torch
import dask.array as da
from tqdm import tqdm
from scipy.sparse import csr_array  # type: ignore

from trackastra.model import Trackastra  # type: ignore
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


def dict_builder(nodes: List[Dict], weights: Tuple[Tuple]):
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
    dico = {}
    node_by_id = {node["id"]: node for node in nodes}
    for (id1, id2), weight in weights:
        node1 = node_by_id[id1]
        node2 = node_by_id[id2]
        dico[(node1["time"], node1["label"] - 1, node2["time"], node2["label"] - 1)] = -np.log(weight)
    return dico


def predict_dist(batch, model: TrackingTransformer) -> np.ndarray:
    """Predict association scores between objects in a batch of windows. Only difference with the original predict function is the maximum
    distance between detections to consider a link (model.transformer.config["spatial_pos_cutoff"]).

    Args:
        batch: dictionary containing:
            - features: Object features array
            - coords: Object coordinates array
            - timepoints: Time points array
        model: TrackingTransformer model to use for prediction.

    Returns:
        Array of association scores between objects.
    """
    feats = torch.from_numpy(batch["features"])
    coords = torch.from_numpy(batch["coords"])
    timepoints = torch.from_numpy(batch["timepoints"]).long()

    device = next(model.parameters()).device
    feats = feats.unsqueeze(0).to(device)
    timepoints = timepoints.unsqueeze(0).to(device)
    coords = coords.unsqueeze(0).to(device)

    coords = torch.cat((timepoints.unsqueeze(2).float(), coords), dim=2)
    with torch.no_grad():
        A = model(coords, features=feats)

        A = model.normalize_output(A, timepoints, coords)
        A = A[0]

        dist = torch.cdist(coords[0, :, 1:], coords[0, :, 1:])
        invalid = dist > model.config["spatial_pos_cutoff"]
        A[invalid] = -torch.inf

        A = A.detach().cpu().numpy()

    return A


def predict_windows_dist(
    windows: list[dict],
    features: list,
    model,
    intra_window_weight: float = 0,
    delta_t: int = 1,
    edge_threshold: float = 0.05,
    spatial_dim: int = 3,
    progbar_class=tqdm,
) -> dict:
    """Predict associations between objects across sliding windows.

    Exactly the same function as predict_windows from Trackastra except it calls predict_dist instead of predict

    Args:
        windows: List of window dictionaries containing:
            - timepoints: Array of time points
            - labels: Array of object labels
            - features: Object features
            - coords: Object coordinates
        features: List of feature objects containing:
            - labels: Object labels
            - timepoints: Time points
            - coords: Object coordinates
        model: TrackingTransformer model to use for prediction.
        intra_window_weight: Weight factor for objects in middle of window. Defaults to 0.
        delta_t: Maximum time difference between objects to consider. Defaults to 1.
        edge_threshold: Minimum association score to consider. Defaults to 0.05.
        spatial_dim: Dimensionality of input masks. May be less than model.coord_dim.
        batch_size: Number of windows to predict on in parallel. Defaults to 1.
        progbar_class: Progress bar class to use. Defaults to tqdm.

    Returns:
        Dictionary containing:
            - nodes: List of node properties (id, coords, time, label)
            - weights: Tuple of ((node_i, node_j), weight) pairs
    """
    # first get all objects/coords
    time_labels_to_id = dict()
    node_properties = list()
    max_id = np.sum([len(f.labels) for f in features])

    all_timepoints = np.concatenate([f.timepoints for f in features])
    all_labels = np.concatenate([f.labels for f in features])
    all_coords = np.concatenate([f.coords for f in features])
    all_coords = all_coords[:, -spatial_dim:]

    for i, (t, la, c) in enumerate(zip(all_timepoints, all_labels, all_coords)):
        time_labels_to_id[(t, la)] = i
        node_properties.append(
            dict(
                id=i,
                coords=tuple(c),
                time=t,
                # index=ix,
                label=la,
            )
        )

    # create assoc matrix between ids
    sp_weights, sp_accum = (
        csr_array((max_id, max_id), dtype=np.float32),
        csr_array((max_id, max_id), dtype=np.float32),
    )

    for t in progbar_class(
        range(0, len(windows)),
        desc="Computing associations",
    ):
        # This assumes that the samples in the dataset are ordered by time and start at 0.
        batch = windows[t]
        timepoints = batch["timepoints"]
        labels = batch["labels"]

        A = predict_dist(batch, model)

        dt = timepoints[None, :] - timepoints[:, None]
        time_mask = np.logical_and(dt <= delta_t, dt > 0)
        A = A[: len(timepoints), : len(timepoints)]
        A[~time_mask] = 0
        ii, jj = np.where(A >= edge_threshold)

        if len(ii) == 0:
            continue

        labels_ii = labels[ii]
        labels_jj = labels[jj]
        ts_ii = timepoints[ii]
        ts_jj = timepoints[jj]
        nodes_ii = np.array(tuple(time_labels_to_id[(t, lab)] for t, lab in zip(ts_ii, labels_ii)))
        nodes_jj = np.array(tuple(time_labels_to_id[(t, lab)] for t, lab in zip(ts_jj, labels_jj)))

        # weight middle parts higher
        t_middle = t + (model.config["window"] - 1) / 2
        ddt = timepoints[:, None] - t_middle * np.ones_like(dt)
        window_weight = np.exp(-intra_window_weight * ddt**2)  # default is 1
        sp_weights[nodes_ii, nodes_jj] += window_weight[ii, jj] * A[ii, jj]
        sp_accum[nodes_ii, nodes_jj] += window_weight[ii, jj]

    sp_weights_coo = sp_weights.tocoo()
    sp_accum_coo = sp_accum.tocoo()
    assert np.allclose(sp_weights_coo.col, sp_accum_coo.col) and np.allclose(sp_weights_coo.row, sp_accum_coo.row)

    # Normalize weights by the number of times they were written from different sliding window positions
    weights = tuple(
        ((i, j), v / a)
        for i, j, v, a in zip(
            sp_weights_coo.row,
            sp_weights_coo.col,
            sp_weights_coo.data,
            sp_accum_coo.data,
        )
    )

    results: dict[str, object] = {}
    results["nodes"] = node_properties
    results["weights"] = weights

    return results


class TrackastraFlex(Trackastra):
    """ "
    Subclass of Trackastra to be able to modify delta_t and a modified _predict
    function to have a maximum distance to link two detections.
    """

    def __init__(self, transformer, train_args, delta_t=4, intra_weight=0, device=None):
        super().__init__(transformer, train_args, device)
        self.delta_t = delta_t
        self.intra_weight = intra_weight
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _predict_dist(
        self,
        imgs: np.ndarray | da.Array,
        masks: np.ndarray | da.Array,
        edge_threshold: float = 0.05,
        n_workers: int = 0,
    ):
        """Same function as the original _predict but it calls predict_windows_dist"""

        self.logger.info("Predicting weights for candidate graph")
        imgs = normalize(imgs)
        self.transformer.eval()
        features = get_features(
            detections=masks,
            imgs=imgs,
            ndim=self.transformer.config["coord_dim"],
            n_workers=n_workers,
            progbar_class=lambda iterable, **kwargs: tqdm(iterable, **kwargs, dynamic_ncols=True),
        )
        self.logger.info("Building windows")
        windows = build_windows(
            features,
            window_size=self.transformer.config["window"],
            progbar_class=lambda iterable, **kwargs: tqdm(iterable, **kwargs, dynamic_ncols=True),
        )
        self.logger.info("Predicting windows")
        predictions = predict_windows_dist(
            windows=windows,
            features=features,
            model=self.transformer,
            intra_window_weight=self.intra_weight,
            edge_threshold=edge_threshold,
            spatial_dim=masks.ndim - 1,
            progbar_class=lambda iterable, **kwargs: tqdm(iterable, **kwargs, dynamic_ncols=True),
            delta_t=self.delta_t,
        )

        return predictions


@dataclasses.dataclass
class TrackaOnStraParameters(FrameByFrameLinkerParameters):
    """Parameters of TrackaOnStraLinker



    Attributes:
        association_threshold (float) : This is the main hyperparameter, it defines the threshold on the distance used
            not to link tracks with detections. It prevents to link with false positive detections.
        edge_threshold (float): Minimum likelihood to consider
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
        association_threshold: float = 5.0,
        edge_threshold: float = 0.05,
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
        self.edge_threshold = edge_threshold


class TrackaOnStraLinker(FrameByFrameLinker):
    """Linker using Trackastra associating costs.
    It is not yet an Online linker since the linker has to be setup with the whole video and all the detections before it can run.

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
        model: TrackastraFlex | None = None,
    ) -> None:
        super().__init__(specs)
        self.specs: TrackaOnStraParameters
        if model is None:
            self.model = TrackastraFlex.from_pretrained("ctc")
        else:
            self.model = model
        self.model.transformer.config["spatial_pos_cutoff"] = self.specs.association_threshold
        self.cost_dict: dict[Tuple, float]
        self.model.delta_t = self.specs.n_gap + 1
        if self.specs.n_gap == 0:  # If n_gap =0 window size of 4 seems to be a bit better
            self.model.transformer.config["window"] = 4
        elif self.specs.n_gap <= 3:
            self.model.transformer.config["window"] = 5
        else:
            warnings.warn("delta_t is too big, results might not be good")
            self.model.transformer.config["window"] = self.specs.n_gap + 2
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

        vid = np.stack([frame.squeeze() for frame in video], axis=0)
        masks_list = []
        for detection in detections_sequence:
            mask = detection.segmentation.cpu().numpy()
            masks_list.append(mask)
        masks = np.stack(masks_list)

        # Then compute the cost
        predictions = self.model._predict_dist(  # pylint: disable=W0212
            vid, masks, edge_threshold=self.specs.edge_threshold
        )

        nodes = predictions["nodes"]
        weights = predictions["weights"]
        self.cost_dict = dict_builder(nodes, weights)

    def motion_model(self) -> None:
        pass

    def post_association(self, _: np.ndarray, detections: byotrack.Detections, active_mask: torch.Tensor):

        self.active_positions[self._links[:, 0]] = detections.position[self._links[:, 1]]

        # Merge still active positions and new ones
        self.active_positions = torch.cat(
            (self.active_positions[active_mask], detections.position[self._unmatched_detections])
        )

        self.all_positions.append(self.active_positions.clone())
        for i, track in enumerate(self.active_tracks):
            if track.detection_ids[-1] == -1:  # Not truly detected
                self.all_positions[-1][i, :] = torch.nan

    def cost(self, _: np.ndarray, detections: byotrack.Detections) -> Tuple[torch.Tensor, float]:
        """Compute the association cost between active tracks and detections.

        Args:
            detections (byotrack.Detections) : Detections on a given frame.

        Returns:
            torch.Tensor: The cost matrix between active tracks and detections
                Shape: (n_tracks, n_dets), dtype: float

            float: The association threshold to use. Not really useful, since there is already an edge threshold in Trackastra.
            Just returning -ln(edge_threshold)

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
        return (cost, -np.log(self.specs.edge_threshold))
