from __future__ import annotations

import dataclasses
import sys
from typing import TYPE_CHECKING, Literal

import numpy as np

import byotrack

if TYPE_CHECKING:
    from trackastra.model import Trackastra  # type: ignore[import-untyped]

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


@dataclasses.dataclass
class TrackAstraParameters:
    """Parameters of TrackAstraLinker.

    Attributes:
        edge_threshold (float) : Minimum probability to consider a link. We advise to keep the default value.
            Default: 0.05
        max_distance (int): Drop links based on an Euclidean thresholding. Can be set to more restrictive values.
            Default: 256
        max_neighbors (int): Keep at most `max_neighbors` in the candidate graph.
            Default: 10
        solver (Literal["greedy_nodiv", "greedy", "ilp_nodiv", "ilp"]): ILP solver to use. "ilp" will solve the optimal
            Integer Linear Program, "greedy" iteratively selects the best feasible edge in the graph. For datasets
            without cellular divisions, add the "nodiv" suffix to drop the support for track splitting.
        channel (int): Channel of the video to use to extract features.
            Default: 0
        batch_size (int): Size of the batches to computes the edge costs.
            Default: 0 (estimated by TrackAstra)
    """

    edge_threshold: float = 0.05
    max_distance: int = 256
    max_neighbors: int = 10
    solver: Literal["greedy_nodiv", "greedy", "ilp_nodiv", "ilp"] = "greedy_nodiv"
    channel: int = 0
    batch_size: int = 0


class TrackAstraLinker(byotrack.Linker):
    """Run TrackAstra [13], wrapping the official implementation.

    About TrackAstra:
    It is a graph-based tracking algorithm, that predicts the associations costs via a trained Transformer.
    The Transformer is first trained on ground truth tracking data, for this you can either use a trained model
    provided by the official github, or train on your own data with the official training scripts.
    Then, the linking of detections is framed as a graph-optimization strategy where the costs are provided by
    the trained model. The optimization can be solved optimally through an Integer Linear Program (ILP), or greedily.


    The workflow is:

    1. Convert the video & detections to numpy.ndarray. (RAM intensive)
    2. Compute the handcrafted features of each detections.
    3. Run the TrackAstra model to predict detection-to-detection association costs.
    4. Build the optimization graph.
    5. Solve for the optimal paths in the graph.
    6. Convert back to byotrack.Track format.

    Note:
        This implementation requires trackastra: `pip install trackastra`
        To use the ILP solver, you should install the ilp extra dependencies: `pip install trackastra[ilp]`

    Warning:
        The TrackAstra implementation do not support missed detections (false negative). If this is a common error
        in your detections, consider using our `TrackOnStraLinker` instead, that implements an online LAP-based
        solving around TrackAstra costs.

    Attributes:
        model (trackastra.model.Trackastra): The TrackAstra model to use.
        specs (TrackAstraParameters): Parameters specifications of the algorithm.
            See `TrackAstraParameters`.
    """

    def __init__(self, model: Trackastra, specs: TrackAstraParameters) -> None:
        super().__init__()
        self.model = model
        self.specs = specs

    @override
    def run(self, video, detections_sequence) -> list[byotrack.Track]:
        # Convert to array the inputs
        imgs = np.asarray(video)[..., self.specs.channel]
        masks = np.zeros(imgs.shape, dtype=np.uint16)
        for frame_id, detections in enumerate(detections_sequence):
            masks[frame_id] = detections.segmentation.numpy().astype(np.uint16)

        # Compute features and association scores
        predictions = self.model._predict(  # noqa: SLF001
            imgs,
            masks,
            edge_threshold=self.specs.edge_threshold,
            normalize_imgs=np.issubdtype(imgs.dtype, np.integer),  # Only normalize if not already floating
            batch_size=self.specs.batch_size or None,
        )

        # Build and solve the graph
        solver = self.specs.solver
        if solver == "ilp_nodiv":
            solver = "ilp"

        graph = self.model._track_from_predictions(  # noqa: SLF001
            predictions,
            mode=solver,
            max_distance=self.specs.max_distance,
            max_neighbors=self.specs.max_neighbors,
            allow_divisions="nodiv" not in self.specs.solver,
        )

        # Convert to ByoTrack
        for node in graph:
            coords = graph.nodes[node].pop("coords")
            if len(coords) == 3:  # noqa: PLR2004
                graph.nodes[node]["z"] = float(coords[-3])

            graph.nodes[node]["y"] = float(coords[-2])
            graph.nodes[node]["x"] = float(coords[-1])

        return byotrack.TrackingGraph.from_nx(graph, frame_key="time").to_tracks()
