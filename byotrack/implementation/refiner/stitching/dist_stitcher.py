from typing import Callable, Collection, Dict, List, Sequence, Set, Union

import numpy as np
import pylapy
import torch

import byotrack


Dist = Callable[[Union[Sequence[np.ndarray], np.ndarray], Collection[byotrack.Track]], np.ndarray]


class DistStitcher(byotrack.Refiner):
    """Track stitching using distance minimization

    Attributes:
        dist (Dist): Callable that compute the distance between each track
        eta (float): Soft threshold in LAP solving (See pylapy)
            Default: inf (No soft thresholding)
        lap_solver (LapSolver): Solver of the assignment problem

    """

    def __init__(self, dist: Dist, eta=float("inf")) -> None:
        super().__init__()
        self.dist = dist
        self.eta = eta
        self.lap_solver = pylapy.LapSolver()

    def run(
        self, video: Union[Sequence[np.ndarray], np.ndarray], tracks: Collection[byotrack.Track]
    ) -> List[byotrack.Track]:
        if not tracks:
            return []

        dist = self.dist(video, tracks)
        links = self.lap_solver.solve(dist, self.eta)
        return self.merge(tracks, links)

    @staticmethod
    def merge(tracks: Collection[byotrack.Track], links: np.ndarray) -> List[byotrack.Track]:
        """Merge tracks following the given links

        Args:
            tracks (Collection[byotrack.Track]): Tracks to merge
            links (np.ndarray): Links between tracks. Each link is a pair of track indices
                Shape: (L, 2), dtype: int

        Returns:
            List[byotrack.Track]: Merged tracks. Note that in a merge track there is a probably a unknown gap
                between the two original tracks where the position of the merge track is set to nan by default
        """
        connected_components = _extract_connected_components(len(tracks), dict(links.tolist()))
        print(f"Merging {len(tracks)} tracks into {len(connected_components)} resulting tracks")

        track_list = list(tracks)
        merged_tracks = []
        dim = track_list[0].points.shape[1]

        for connected_component in connected_components:
            if len(connected_component) == 1:  # Do not duplicate the data just reuse the existing track
                merged_tracks.append(track_list[connected_component.pop()])
                continue

            # Merge multiple tracks
            start = min(track_list[i].start for i in connected_component)
            end = max(track_list[i].start + len(track_list[i]) for i in connected_component)

            merge_ids = set(track_list[i].merge_id for i in connected_component)
            if len(merge_ids) > 2:  # TODO: Remove the check, it should never occurs
                raise RuntimeError("Stitching conflics with merging. Please open an Issue.")

            points = torch.full((end - start, dim), torch.nan)
            detection_ids = torch.full((end - start,), -1, dtype=torch.int32)
            identifier = -1

            for i in connected_component:
                if track_list[i].start == start:  # Take the identifier of the first track
                    identifier = track_list[i].identifier

                points[track_list[i].start - start : track_list[i].start - start + len(track_list[i])] = track_list[
                    i
                ].points
                detection_ids[track_list[i].start - start : track_list[i].start - start + len(track_list[i])] = (
                    track_list[i].detection_ids
                )

            merged_tracks.append(byotrack.Track(start, points, identifier, detection_ids, max(merge_ids)))

        return merged_tracks

    @staticmethod
    def skip_computation(
        tracks: Collection[byotrack.Track], max_overlap: int, max_dist: float, max_gap: int
    ) -> torch.Tensor:
        """Compute a boolean mask that indicate which distance should be skipped

        Based on simple rules, prevents the computation of most distances. Let i, j two tracks:

        1. i ends at most `max_gap` frames before j starts
        2. i ends at most `max_overlap` frames after j starts
        3. i last position is at most at `max_dist` from j first position
        4. i is not merged to any other tracks.

        Args:
            tracks (Collection[byotrack.Track]): Current set of tracks
            max_overlap (int): Cannot stitch tracks that overlap more than `max_overlap`
            max_dist (float): Cannot stich track i and track j if the last position of i and
                first position of j are farther than `max_dist` (ignored if max_dist <= 0)
            max_gap (int): Cannot stich track i and track j if i ended more
                than `max_gap` frame before j started (ignored if max_gap <= 0)

        Returns:
            torch.Tensor: Boolean tensor that indicates True when the dist computation should be skipped
        """
        starts = torch.tensor([track.start for track in tracks])
        ends = torch.tensor([track.start + len(track) for track in tracks])

        first_pos = torch.cat([track.points[:1] for track in tracks])
        last_pos = torch.cat([track.points[-1:] for track in tracks])

        merge_ids = torch.tensor([track.merge_id for track in tracks])

        skip = starts[:, None] >= starts[None, :]  # Ensure full asymmetry
        skip |= ends[:, None] > starts[None, :] + max_overlap
        skip |= merge_ids[:, None] >= 0

        if max_gap > 0:
            skip |= ends[:, None] + max_gap < starts[None, :]

        if max_dist > 0:
            skip |= (last_pos[:, None, :] - first_pos[None, :, :]).pow(2).sum(dim=-1) > max_dist**2

        return skip

    @staticmethod
    def normalize(dist: np.ndarray) -> np.ndarray:
        """Normalize a distance matrix into [0, 1]

        Args:
            dist (np.ndarray): Distance matrix to normalize
                Shape: (N, N), dtype: float32

        Returns:
            np.ndarray: Normalized distance matrix
                Shape: (N, N), dtype: float32
        """
        return (dist - dist.min()) / (dist[dist != torch.inf].max() - dist.min())


def _extract_connected_components(n: int, edges: Dict[int, int]) -> List[Set[int]]:
    """Extract the connected components in a linear directed graph

    Assume that each node has at most one child.

    Args:
        n (int): Number of nodes (from 0 to n - 1)
        edges (Dict[int, int]): Edges i -> j

    Returns:
        List[Set[int]]: List of connected components (each is a set of id)
    """
    edges = edges.copy()
    seen_nodes = np.full(n, False)
    start_nodes_to_cc_id: Dict[int, int] = {}

    connected_components: List[Set[int]] = []

    for node in range(n):
        if seen_nodes[node]:  # If seen, no need to go through it again
            continue

        seen_nodes[node] = True
        connected_component = {node}

        start_node = node

        while node in edges:  # Pop each edges from the start node a follow them
            node = edges.pop(node)
            seen_nodes[node] = True
            connected_component.add(node)

        # As nodes are not sorted we could find at the end an already handled start_node
        # Let's then just merge the connected components
        if node in start_nodes_to_cc_id:
            cc_id = start_nodes_to_cc_id[node]
            connected_components[cc_id].update(connected_component)
        else:  # New connected components
            cc_id = len(connected_components)
            connected_components.append(connected_component)

        start_nodes_to_cc_id[start_node] = cc_id

    return connected_components
