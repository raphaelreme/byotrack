from __future__ import annotations

from typing import Collection, List, Dict, Tuple, Union
import warnings

import networkx as nx
import torch

import byotrack  # pylint: disable=cyclic-import


class TrackingGraph(nx.DiGraph):
    """Directed tracking graph.

    An alternative view over tracks where each detection is a node and temporal
    continuity/split/merge relations are edges. Provides conversion to/from the
    default `byotrack.Track` format and can bridge to other tracking formats.

    Note: TrackingGraph should only be created through the two provided builders: `from_tracks` and
        `from_nx`, as they will sanitize and check the data format, allowing its usage across ByoTrack.

    Node attributes:

    * t: frame index
    * [z, ]y, x: detection position (z is optional; 3D only)
    * track_id: positive identifier of the track (optional)
    * detection_id: source detection id (optional, -1 if unknown)

    Edge attributes (optional):

    * split: True if the edge represents a split
    * merge: True if the edge represents a merge

    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def from_tracks(tracks: Collection[byotrack.Track]) -> TrackingGraph:  # pylint: disable=too-many-branches
        """Build a graph from `byotrack.Track` objects.

        Each track becomes a linear chain (one node per defined point). Split/merge
        relations are added as edges between the end of a parent and
        the start of the child track.

        Args:
            tracks (Collection[Track]): Tracks to convert.

        Returns:
            TrackingGraph: The constructed graph.
        """
        trk_graph = TrackingGraph()

        # Handle corner the corner case
        if not tracks:
            return trk_graph

        id_to_track = {track.identifier: track for track in tracks}
        id_to_first_node = {}
        id_to_last_node = {}

        # 1/ Handle linear segments (Track)
        node_id = 0
        for track in tracks:
            id_to_first_node[track.identifier] = node_id
            previous_node = None
            dim = track.points.shape[1]
            has_z = dim == 3
            for i in range(len(track)):
                coords = track.points[i]
                if torch.isnan(coords).any():
                    continue

                node_attrs: Dict[str, Union[float, int]] = {
                    "t": track.start + i,
                    "track_id": int(track.identifier),
                    "detection_id": int(track.detection_ids[i].item()),
                }
                if has_z:
                    node_attrs["z"] = float(coords[0].item())
                    node_attrs["y"] = float(coords[1].item())
                    node_attrs["x"] = float(coords[2].item())
                else:
                    node_attrs["y"] = float(coords[0].item())
                    node_attrs["x"] = float(coords[1].item())

                trk_graph.add_node(node_id, **node_attrs)

                # Let's add the edge between previous node and this one
                if previous_node is not None:
                    trk_graph.add_edge(previous_node, node_id)

                previous_node = node_id
                node_id += 1

            if previous_node is None:  # Empty tracks should not occur
                raise ValueError("Empty tracks are not supported")

            id_to_last_node[track.identifier] = previous_node

        # 2/ Add split and merge edges
        for track in tracks:
            if track.parent_id == -1:
                continue

            parent = id_to_track[track.parent_id]
            trk_graph.add_edge(id_to_last_node[parent.identifier], id_to_first_node[track.identifier], split=True)

        # 3/ Add merge edges
        # NOTE: that an edge could theoretically be included in both a split and a merge... Let's support this here
        for track in tracks:
            if track.merge_id == -1:
                continue

            edge = (id_to_last_node[track.identifier], id_to_first_node[id_to_track[track.merge_id].identifier])
            if edge in trk_graph.edges:  # Already added as a split
                assert trk_graph.edges[edge]["split"]
                trk_graph.edges[edge]["merge"] = True
            else:
                trk_graph.add_edge(*edge, merge=True)

        return trk_graph

    @staticmethod
    def from_nx(  # pylint: disable=too-many-locals
        nx_graph: nx.DiGraph,
        *,
        frame_key="t",
        x_key="x",
        y_key="y",
        z_key="z",
        track_key="track_id",
        detection_key="detection_id",
        merge_key="merge",
        split_key="split",
    ) -> TrackingGraph:
        """Sanitize and convert a generic NetworkX DiGraph to a TrackingGraph.

        Copies nodes/edges and remaps attribute keys. Validates that all edges
        are forward in time. Split/merge flags are imported if present and also
        inferred from in/out degrees.

        Args:
            nx_graph (nx.DiGraph): Source graph.
            *_key (str): Node and edge attribute names to read from `nx_graph`.

        Returns:
            TrackingGraph: The sanitized tracking graph.
        """
        trk_graph = TrackingGraph()
        for node_id in nx_graph.nodes:
            node = nx_graph.nodes[node_id]
            node_attrs: Dict[str, Union[float, int]] = {
                "t": node[frame_key],
                "y": node[y_key],
                "x": node[x_key],
            }
            for key, attr in [(z_key, "z"), (track_key, "track_id"), (detection_key, "detection_id")]:  # Optional keys
                if key in node:
                    node_attrs[attr] = node[key]

            trk_graph.add_node(node_id, **node_attrs)

        for edge in nx_graph.edges:
            # assert we always move forward in time. It also ensures there is no cycle in the graph
            assert trk_graph.nodes[edge[1]]["t"] - trk_graph.nodes[edge[0]]["t"] > 0, "Backward edge are not supported"

            trk_graph.add_edge(*edge)

            # Set split and merges from metadata if it exists
            if nx_graph.edges[edge].get(split_key, False):
                trk_graph.edges[edge]["split"] = True

            if nx_graph.edges[edge].get(merge_key, False):
                trk_graph.edges[edge]["merge"] = True

            # Automatically detect split and merges
            # NOTE: If our format support split with a single child (e.g. some gt in the CTC),
            #       this automatic classification do not support it (i.e. it has to be labeled explicitly)
            if len(list(nx_graph.successors(edge[0]))) > 1:
                trk_graph.edges[edge]["split"] = True

            if len(list(nx_graph.predecessors(edge[1]))) > 1:
                trk_graph.edges[edge]["merge"] = True

        return trk_graph

    def to_tracks(self) -> List[byotrack.Track]:  # pylint: disable=too-many-locals
        """Convert the graph into `byotrack.Track` objects.

        Linear segments (paths without split/merge) become tracks. Gaps inside a
        segment are encoded as NaNs. Track ids are reused when consistent, otherwise
        new unique ids are assigned. Split/merge relations set `parent_id`/`merge_id`.

        Returns:
            List[Track]: Tracks reconstructed from the graph.
        """
        # First extract the required data, then build tracks
        segments = self._segments()
        segment_ids = self._identify_segments(segments)
        id_to_parent, id_to_merge = self._get_split_and_merge_ids(segments, segment_ids)

        # Handle empty graph
        if not segments:
            return []

        dim = 2
        if "z" in self.nodes[segments[0][0]]:
            dim = 3

        tracks = []
        for segment, identifier in zip(segments, segment_ids):
            start = self.nodes[segment[0]]["t"]
            end = self.nodes[segment[-1]]["t"] + 1

            points = torch.full((end - start, dim), torch.nan, dtype=torch.float32)
            detection_ids = torch.full((end - start,), -1, dtype=torch.int32)

            for node_id in segment:
                node = self.nodes[node_id]
                offset = node["t"] - start

                if dim == 3:
                    points[offset, 0] = node["z"]
                    points[offset, 1] = node["y"]
                    points[offset, 2] = node["x"]
                else:
                    points[offset, 0] = node["y"]
                    points[offset, 1] = node["x"]

                detection_ids[offset] = node.get("detection_id", -1)

            tracks.append(
                byotrack.Track(
                    start,
                    points,
                    identifier=identifier,
                    detection_ids=detection_ids,
                    merge_id=id_to_merge.get(identifier, -1),
                    parent_id=id_to_parent.get(identifier, -1),
                )
            )

        return tracks

    def _root_nodes(self) -> List:
        """Return nodes with no predecessors (track births)."""
        return [node for node, in_degree in self.in_degree() if in_degree == 0]  # type: ignore

    def _segments(self) -> List[List]:
        """Return linear segments for track reconstruction.

        Depth-first traversal that grows a segment until a split, merge, or end,
        starting from birth nodes. Labeled split/merge edges also terminate segments.
        """
        start_nodes = self._root_nodes()

        seen: set = set()
        segments = []
        while start_nodes:  # DFS
            node_id = start_nodes.pop()
            if node_id in seen:
                continue  # Merge nodes are added several times, let's go through it once

            seen.add(node_id)

            segment = [node_id]
            while True:
                outs = list(self.successors(node_id))

                if len(outs) != 1:  # Split or End
                    break

                child_id = outs[0]

                if self.edges[(node_id, child_id)].get("split", False):  # Labeled split
                    break

                if len(list(self.predecessors(child_id))) != 1:  # Merge
                    break

                if self.edges[(node_id, child_id)].get("merge", False):  # Labeled merge
                    break

                segment.append(child_id)
                node_id = child_id

            # Split and merge define new starting segments
            start_nodes.extend(outs)
            segments.append(segment)

        return segments

    def _identify_segments(self, segments: List[List]) -> List[int]:
        """Assign a unique track id per segment.

        Reuses a segment's `track_id` when consistent; otherwise issues a warning
        and assigns a new id. Ensures all segment ids are unique.
        """
        segment_ids = []
        for segment in segments:
            track_id = -1
            for node in segment:
                id_ = self.nodes[node].get("track_id", -1)
                if id_ != track_id:
                    if track_id == -1:
                        track_id = id_
                    else:
                        warnings.warn(f"Found two different track_ids for the segment of node {node}")

            segment_ids.append(track_id)

        # Ensure ids are different
        next_id = max(segment_ids) + 1
        used: set = set()
        for i, track_id in enumerate(segment_ids):
            if track_id == -1:
                segment_ids[i] = next_id
                next_id += 1
                continue

            if track_id in used:
                warnings.warn(f"Found duplicate track_id {track_id}. Relabeled into {next_id}.")
                segment_ids[i] = next_id
                next_id += 1
                continue

            used.add(track_id)

        return segment_ids

    def _get_split_and_merge_ids(
        self, segments: List[List], segment_ids: List[int]
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Derive `parent_id` and `merge_id` mappings from edges.

        Labeled edges or degree-based inference are used.
        """
        first_to_id = {segment[0]: segment_ids[i] for i, segment in enumerate(segments)}
        last_to_id = {segment[-1]: segment_ids[i] for i, segment in enumerate(segments)}

        id_to_parent = {}
        id_to_merge = {}

        for source, target in self.edges:
            if len(list(self.successors(source))) > 1 or self.edges[(source, target)].get("split", False):  # Split
                id_to_parent[first_to_id[target]] = last_to_id[source]
            if len(list(self.predecessors(target))) > 1 or self.edges[(source, target)].get("merge", False):  # Merge
                id_to_merge[last_to_id[source]] = first_to_id[target]

        return id_to_parent, id_to_merge
