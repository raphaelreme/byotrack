from __future__ import annotations

import dataclasses
import sys
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import tqdm.auto as tqdm
import trackastra  # type: ignore[import-untyped]
from packaging import version
from trackastra.data import build_windows, get_features  # type: ignore[import-untyped]
from trackastra.model import Trackastra  # type: ignore[import-untyped]
from trackastra.model.predict import predict_windows  # type: ignore[import-untyped]
from trackastra.utils import normalize  # type: ignore[import-untyped]

from byotrack.implementation.linker.frame_by_frame.base import (
    AssociationMethod,
    FrameByFrameLinker,
    FrameByFrameLinkerParameters,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from trackastra.model import TrackingTransformer

    import byotrack

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


def build_cost_dict(
    nodes: list[dict[str, Any]], weights: Iterable[tuple[tuple[int, int], float]]
) -> dict[tuple[int, int, int], dict[int, float]]:
    """Build the cost dictionary from Trackastra data format.

    It converts the feasible tracking graph predicted by trackastra into a mapping from edge to cost.
    Where an edge is a link between two detections (node). It converts the probability weight into a
    cost = -log weight.

    Args:
        nodes (list[dict[str, Any]]): Nodes of the graph (detections), with their features
            Format: { "id": int, "coords": (float, float), "time": int, "label": int}
        weights (Iterable[tuple[tuple[int, int], float]]): Considered edges, with their weight
            Format: [((node_id, node_id_2), weight), ...]

    Returns:
        dict[tuple[int, int, int], dict[int, float]]: Cost dictionary of the feasible tracking graph.
        For every (frame1, node1, frame2) there is a dictionary with key node2 and the value the cost of the edge.
    """
    cost_dict: dict[tuple[int, int, int], dict[int, float]] = {}
    node_by_id = {node["id"]: node for node in nodes}
    for (id_1, id_2), weight in weights:
        node_1 = node_by_id[id_1]
        node_2 = node_by_id[id_2]
        key = (node_1["time"], node_1["label"] - 1, node_2["time"])
        if key not in cost_dict:
            cost_dict[key] = {}
        cost_dict[key][node_2["label"] - 1] = float(-np.log(weight))
    return cost_dict


class TrackastraFlex(Trackastra):
    """Trackastra with ability to modify delta_t."""

    def __init__(
        self,
        transformer: TrackingTransformer,
        train_args: dict[str, Any],
        delta_t: int = 4,
        intra_weight: float = 0,
        device=None,
    ):
        super().__init__(transformer, train_args, device)
        self.delta_t = delta_t
        self.intra_weight = intra_weight

    def predict_with_gap(
        self,
        imgs: np.ndarray,
        masks: np.ndarray,
        edge_threshold: float = 0.05,
        n_workers: int = 0,
    ) -> dict:
        """Same function as the original _predict but it calls predict_windows with the model delta_t."""
        # Note: It lags a bit behind the 0.4.0 implem that has added new parameters to _predict

        print("Predicting weights for candidate graph")  # noqa: T201
        imgs = normalize(imgs)
        self.transformer.eval()
        features = get_features(
            detections=masks,
            imgs=imgs,
            ndim=self.transformer.config["coord_dim"],
            n_workers=n_workers,
            progbar_class=lambda iterable, **kwargs: tqdm.tqdm(iterable, **kwargs, dynamic_ncols=True),
        )
        print("Building windows")  # noqa: T201
        kwargs = {"as_torch": True} if version.parse(trackastra.__version__) >= version.parse("0.4.0") else {}
        windows = build_windows(
            features,
            window_size=self.transformer.config["window"],
            progbar_class=lambda iterable, **kwargs: tqdm.tqdm(iterable, **kwargs, dynamic_ncols=True),
            **kwargs,
        )
        print("Predicting windows")  # noqa: T201
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

        return predictions  # noqa: RET504


@dataclasses.dataclass
class TrackOnStraParameters(FrameByFrameLinkerParameters):
    """Parameters of TrackOnStraLinker.

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
            It can be provided as a string. (Choice: GREEDY, OPT_HARD, OPT_SMOOTH, SPARSE_OPT_HARD, SPARSE_OPT_SMOOTH)
            Default: OPT_SMOOTH
        anisotropy (tuple[float, float, float]): Anisotropy of images (Ratio of the pixel sizes
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
        association_method: str | AssociationMethod = AssociationMethod.OPT_SMOOTH,
        anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
        split_factor: float = 0.0,
        merge_factor: float = 0.0,
    ):
        super().__init__(
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
    """Online TrackAstra.

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
        cost_dict (dict[tuple[int, int, int], dict[int, float]): Cost dictionary of the feasible tracking graph.

    """

    progress_bar_description = "TrackaOnStra linking"

    def __init__(
        self,
        specs: TrackOnStraParameters,
        model: TrackastraFlex | None = None,
        optflow: byotrack.OpticalFlow | None = None,
        features_extractor: byotrack.FeaturesExtractor | None = None,
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
        elif self.specs.n_gap <= 3:  # noqa: PLR2004
            self.model.transformer.config["window"] = 5
        else:
            warnings.warn("delta_t is too big, results might not be good", stacklevel=2)
            self.model.transformer.config["window"] = self.specs.n_gap + 2

        # Initialize empty cost_dict
        self.cost_dict: dict[tuple[int, int, int], dict[int, float]] = {}

        if optflow is not None:
            warnings.warn("OpticalFlow is not supported by this linker, it will be ignored", stacklevel=2)
        if features_extractor is not None:
            warnings.warn("Features Extractor is not supported by this linker, it will be ignored", stacklevel=2)

    def setup(
        self, video: Sequence[np.ndarray] | np.ndarray, detections_sequence: Sequence[byotrack.Detections]
    ) -> None:
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
        masks = np.zeros(imgs.shape, dtype=np.uint16)

        for i, detection in enumerate(detections_sequence):
            masks[i] = detection.segmentation.cpu().numpy().astype(np.uint16)

        # Then compute the cost
        predictions = self.model.predict_with_gap(imgs, masks, edge_threshold=self.specs.association_threshold)

        nodes = predictions["nodes"]
        weights = predictions["weights"]

        # And convert back to a more usable format
        self.cost_dict = build_cost_dict(nodes, weights)

    @override
    def motion_model(self) -> None:
        pass  # No motion to model

    @override
    def post_association(self, frame: np.ndarray, detections: byotrack.Detections, active_mask: torch.Tensor) -> None:
        # Simply store the position of the detection
        active_positions = torch.full((len(self.active_tracks), detections.dim), torch.nan)

        for i, track in enumerate(self.active_tracks):
            j = track.detection_ids[-1]
            if j != -1:
                active_positions[i] = detections.position[j]

        self.all_positions.append(active_positions)

    @override
    def cost(self, frame: np.ndarray, detections: byotrack.Detections) -> tuple[torch.Tensor, float]:
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
