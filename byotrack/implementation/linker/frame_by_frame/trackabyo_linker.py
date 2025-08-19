import dataclasses
from typing import List, Optional, Tuple, Dict, Union, Collection, Sequence
import warnings
import logging

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
from trackastra.model.predict import predict_windows  # type: ignore
from trackastra.model import TrackingTransformer

import byotrack
from byotrack.implementation.linker.frame_by_frame.base import (
    AssociationMethod,
)
from byotrack.implementation.linker.frame_by_frame.nearest_neighbor import (
    NearestNeighborParameters,
    NearestNeighborLinker,
)


def dict_builder(nodes: List[Dict], weights: Tuple[Tuple]):
    """
    Build the dictionnary matching with the Trackastra's graph

    Args :
        nodes : The list of the graph's nodes (detections),
            shape of node {'id': ,'coords': ,'time': ,'label': }
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
    """Predict association scores between objects in a batch of windows.

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

        # TODO stay on device for further computation?
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

    This function processes a sequence of sliding windows to predict associations
    between objects across time frames. It handles:
    - Object tracking across time
    - Weight normalization across windows
    - Edge thresholding
    - Time-based filtering

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
        # window_weight = np.exp(4*A) # smooth max
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


class NewTrackastra(Trackastra):
    """ "
    Subclass of Trackastra just to modify the _predict function to have delta_t=4
    """

    def __init__(self, transformer, train_args, delta_t=4, intra_weight=0, device=None):
        super().__init__(transformer, train_args, device)
        self.transformer.config["window"] = 5
        self.delta_t = delta_t
        self.intra_weight = intra_weight
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _predict(
        self,
        imgs: np.ndarray,
        masks: np.ndarray,
        edge_threshold: float = 0.05,
        n_workers: int = 0,
        progbar_class=tqdm,
    ):
        self.logger.info("Predicting weights for candidate graph")
        imgs = normalize(imgs)
        self.transformer.eval()

        features = get_features(
            detections=masks,
            imgs=imgs,
            ndim=self.transformer.config["coord_dim"],
            n_workers=n_workers,
            progbar_class=progbar_class,
        )
        self.logger.info("Building windows")
        windows = build_windows(
            features,
            window_size=self.transformer.config["window"],
            progbar_class=progbar_class,
        )

        self.logger.info("Predicting windows")
        predictions = predict_windows(
            windows=windows,
            features=features,
            model=self.transformer,
            edge_threshold=edge_threshold,
            intra_window_weight=self.intra_weight,
            spatial_dim=masks.ndim - 1,
            progbar_class=progbar_class,
            delta_t=self.delta_t,
        )

        return predictions

    def _predict_dist(
        self,
        imgs: np.ndarray | da.Array,
        masks: np.ndarray | da.Array,
        edge_threshold: float = 0.05,
        n_workers: int = 0,
        progbar_class=tqdm,
    ):
        self.logger.info("Predicting weights for candidate graph")
        imgs = normalize(imgs)

        self.transformer.eval()

        features = get_features(
            detections=masks,
            imgs=imgs,
            ndim=self.transformer.config["coord_dim"],
            n_workers=n_workers,
            progbar_class=progbar_class,
        )
        self.logger.info("Building windows")
        windows = build_windows(
            features,
            window_size=self.transformer.config["window"],
            progbar_class=progbar_class,
        )

        self.logger.info("Predicting windows")
        predictions = predict_windows_dist(
            windows=windows,
            features=features,
            model=self.transformer,
            intra_window_weight=self.intra_weight,
            edge_threshold=edge_threshold,
            spatial_dim=masks.ndim - 1,
            progbar_class=progbar_class,
            delta_t=self.delta_t,
        )

        return predictions


@dataclasses.dataclass
class TrackaByoParameters(NearestNeighborParameters):
    """Parameters of TrackaByoLinker



    Attributes:
        association_threshold (float): This is the main hyperparameter, it defines the threshold on the distance used
            not to link tracks with detections. It prevents to link with false positive detections.
        n_valid (int): Number associated detections required to validate the track after its creation.
            Default: 3
        n_gap (int): Number of consecutive frames without association before the track termination.
            Default: 3
        association_method (AssociationMethod): The frame-by-frame association to use. See `AssociationMethod`.
            It can be provided as a string. (Choice: GREEDY, [SPARSE_]OPT_HARD, [SPARSE_]OPT_SMOOTH)
            Default: OPT_SMOOTH
        anisotropy (Tuple[float, float, float]): Anisotropy of images (Ratio of the pixel sizes
            for each axis, depth first). This will be used to scale distances.
            Default: (1., 1., 1.)
        fill_gap (bool): Fill the gap of missed detections using a forward optical flow
            propagation (Only when optical flow is provided). We advise to rather use a
            ForwardBackward interpolation using the same optical flow: it will produce
            smoother interpolations.
            Default: False
        ema (float): Optional exponential moving average to reduce detection noise. Detection positions are smoothed
            using this EMA. Should be smaller than 1. It use: x_{t+1} = ema x_{t} + (1 - ema) det(t)
            As motion is not modeled, EMA may introduce lag that will hinder tracking. It is more effective with
            optical flow to compensate motions, in this case, a typical value is 0.5, to average the previous position
            with the current measured one. For more advanced modelisation, see `KalmanLinker`.
            Default: 0.0 (No EMA)
        split_factor (float): Allow splitting of tracks, using a second association step.
            The association threshold in this case is `split_factor * association_threshold`.
            Default: 0.0 (No splits)
        merge_factor (float): Allow merging of tracks, using a second association step.
            The association threshold in this case is `merge_factor * association_threshold`.
            Default: 0.0 (No merges)

    """

    def __init__(
        self,
        max_dist: float = 0.0,
        association_threshold: float = 0.69,
        n_valid=1,
        n_gap=3,
        association_method: Union[str, AssociationMethod] = AssociationMethod.OPT_SMOOTH,
        anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        ema=1.0,
        fill_gap=False,
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
        self.max_dist = max_dist


class TrackaByoLinker(NearestNeighborLinker):
    """Frame by frame linker using Trackastra associating costs.

    See `TrackaByoParamaters` for the other attributes.

    Attributes:
        specs (TrackaByoParameters): Parameters specifications of the algorithm.
            See `TrackaByoParameters`.
        model (NewTrackastra) : Model of Trackastra used to compute the association costs.
        cost_dict (Dict) : Dictionnary of all the possible link between detections associated with its cost.

    """

    progress_bar_description = "TrackaByo linking"

    def __init__(
        self,
        specs: TrackaByoParameters,
        model: NewTrackastra,
        optflow: Optional[byotrack.OpticalFlow] = None,
        features_extractor: Optional[byotrack.FeaturesExtractor] = None,
        save_all=False,
    ) -> None:
        super().__init__(specs, optflow, features_extractor, save_all)
        self.specs: TrackaByoParameters
        self.model = model
        self.cost_dict: dict[Tuple, float]
        if self.specs.n_gap == 0:  # If n_gap =0 window size of 4 seems to be a bit better
            self.model.delta_t = 1
            self.model.transformer.config["window"] = 4

        if self.specs.fill_gap and not self.optflow:
            warnings.warn("Optical flow has not been provided. Gap cannot be filled")

    def run(
        self, video: Union[Sequence[np.ndarray], np.ndarray], detections_sequence: Sequence[byotrack.Detections]
    ) -> Collection[byotrack.Track]:
        """Run the linker on a whole video

        Args:
            video (Sequence[np.ndarray] | np.ndarray): Sequence of T frames (array).
                Each array is expected to have a shape ([D, ]H, W, C)
            detections_sequence (Sequence[byotrack.Detections]): Detections for each frame
                Detections is expected for each frame of the video, in the same order.
                (Note that for a given frame, the Detections can be empty)
            dist (bool) : Boolean that indicates whether or not a maximum distance should be used when doing the linking.

        Returns:
            Collection[byotrack.Track]: Tracks of particles

        """
        if len(video) != len(detections_sequence):
            warnings.warn(
                f"""Expected to have one Detections for each frame of the video.

            There are {len(detections_sequence)} Detections for {len(video)} frames.
            This can lead to unexpected behavior. By default we assume that the first Detections
            is aligned with the first frame and stop when the end of shortest sequence is reached.
            """
            )

        if len(video) == 0:
            return []

        self.reset(video[0].ndim - 1)

        # Convert Byotrack datas for Trackastra

        vid = np.stack([frame.squeeze() for frame in video], axis=0)
        masks_list = []
        for detection in detections_sequence:
            mask = detection.segmentation.cpu().numpy()
            masks_list.append(mask)
        masks = np.stack(masks_list)

        # Then compute the cost
        if self.specs.max_dist > 0.0:
            self.model.transformer.config["spatial_pos_cutoff"] = self.specs.max_dist
            predictions = self.model._predict_dist(vid, masks)  # pylint: disable=W0212
        else:
            predictions = self.model._predict(vid, masks)  # pylint: disable=W0212
        nodes = predictions["nodes"]
        weights = predictions["weights"]
        self.cost_dict = dict_builder(nodes, weights)

        progress_bar = tqdm(
            desc=self.progress_bar_description,
            total=min(len(video), len(detections_sequence)),
        )

        for frame, detections in zip(
            [np.zeros((1, 1, 1, 1)) for _ in range(len(detections_sequence))],
            detections_sequence,
        ):
            self.update(frame, detections)
            progress_bar.update()

        progress_bar.close()

        tracks = self.collect()

        # Check produced tracks
        byotrack.Track.check_tracks(tracks, warn=True)
        return tracks

    def cost(self, _: np.ndarray, detections: byotrack.Detections) -> Tuple[torch.Tensor, float]:
        """Compute the association cost between active tracks and detections

        For likelihood association, you could provide the association threshold as a probability
        and use -log(threshold) as the true threshold. (See `KalmanLinker` and `NearestNeighborLinker`)

        Args:
            frame_id (int): The index of thecurrent frame of the video


        Returns:
            torch.Tensor: The cost matrix between active tracks and detections
                Shape: (n_tracks, n_dets), dtype: float
            float: The association threshold to use.

        """
        nb_lignes = len(self.active_tracks)
        nb_colonnes = detections.length
        cost = torch.full((nb_lignes, nb_colonnes), torch.inf)
        for index, track in enumerate(self.active_tracks):
            for index_det in range(nb_colonnes):
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
        return (cost, self.specs.association_threshold)
