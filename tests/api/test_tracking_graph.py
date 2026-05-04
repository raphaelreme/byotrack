from __future__ import annotations

import networkx as nx
import pytest
import torch

import byotrack

## Construction from tracks


def test_from_tracks_empty_collection():
    graph = byotrack.TrackingGraph.from_tracks([])
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


def test_from_tracks_single_2d_track():
    points = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    track = byotrack.Track(5, points, identifier=42)
    graph = byotrack.TrackingGraph.from_tracks([track])

    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2

    nodes = sorted(graph.nodes(data=True), key=lambda n: n[1]["t"])

    n0 = nodes[0][1]

    assert n0["t"] == 5
    assert n0["y"] == 1.0
    assert n0["x"] == 2.0
    assert n0["track_id"] == 42
    assert n0["detection_id"] == -1
    assert "z" not in n0


def test_from_tracks_single_3d_track():
    points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    det_ids = torch.tensor([0, 1], dtype=torch.int32)
    track = byotrack.Track(3, points, identifier=666, detection_ids=det_ids)
    graph = byotrack.TrackingGraph.from_tracks([track])

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1

    nodes = sorted(graph.nodes(data=True), key=lambda n: n[1]["t"])

    n1 = nodes[1][1]

    assert n1["t"] == 4
    assert n1["z"] == 4.0
    assert n1["y"] == 5.0
    assert n1["x"] == 6.0
    assert n1["track_id"] == 666
    assert n1["detection_id"] == 1


def test_from_tracks_independent_tracks_no_cross_edges():
    track_1 = byotrack.Track(0, torch.ones(3, 2))
    track_2 = byotrack.Track(0, torch.ones(3, 2) * 2)
    graph = byotrack.TrackingGraph.from_tracks([track_1, track_2])

    assert len(graph.nodes) == 6
    assert len(graph.edges) == 4

    components = list(nx.connected_components(graph.to_undirected()))

    assert len(components) == 2


# Behavior has been changed.
# def test_from_tracks_nan_position_skipped_and_bridge():
#     points = torch.tensor([[1.0, 2.0], [float("nan"), float("nan")], [5.0, 6.0]])
#     track = byotrack.Track(0, points)
#     graph = byotrack.TrackingGraph.from_tracks([track])

#     assert len(graph.nodes) == 2
#     assert len(graph.edges) == 1
#     assert (0, 1) in graph.edges
#     assert graph.nodes[0]["t"] == 0
#     assert graph.nodes[1]["t"] == 2


def test_from_tracks_split_edge_marked() -> None:
    tracks = [
        byotrack.Track(3, torch.rand(5, 3), identifier=5),
        byotrack.Track(8, torch.rand(7, 3), identifier=7, parent_id=5),
        byotrack.Track(8, torch.rand(9, 3), identifier=12, parent_id=5),
    ]

    graph = byotrack.TrackingGraph.from_tracks(tracks)
    split_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if d.get("split", False)]

    assert len(split_edges) == 2  # Parent to child 1 and 2
    assert "merge" not in split_edges[0][2]
    assert len(graph.edges) == 4 + 6 + 8 + 2
    assert len(graph.nodes) == 5 + 7 + 9

    components = list(nx.connected_components(graph.to_undirected()))
    assert len(components) == 1


def test_from_tracks_merge_edge_marked() -> None:
    tracks = [
        byotrack.Track(3, torch.rand(5, 2), identifier=5, merge_id=12),
        byotrack.Track(5, torch.rand(3, 2), identifier=7, merge_id=12),
        byotrack.Track(8, torch.rand(4, 3), identifier=12),
    ]

    graph = byotrack.TrackingGraph.from_tracks(tracks)
    merge_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if d.get("merge", False)]

    assert len(merge_edges) == 2  # Parent 1 & 2 to child
    assert "split" not in merge_edges[0][2]
    assert len(graph.edges) == 4 + 2 + 3 + 2
    assert len(graph.nodes) == 5 + 3 + 4

    components = list(nx.connected_components(graph.to_undirected()))
    assert len(components) == 1


def test_from_tracks_merge_and_split_typical_case() -> None:
    tracks = [
        byotrack.Track(3, torch.rand(5, 3), identifier=5),
        byotrack.Track(8, torch.rand(7, 3), identifier=7, parent_id=5, merge_id=1),
        byotrack.Track(8, torch.rand(7, 3), identifier=12, parent_id=5, merge_id=1),
        byotrack.Track(15, torch.rand(3, 3), identifier=1),
        byotrack.Track(0, torch.rand(15, 3), identifier=0),  # Indep track
    ]

    graph = byotrack.TrackingGraph.from_tracks(tracks)
    split_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if d.get("split", False)]
    merge_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if d.get("merge", False)]

    assert len(split_edges) == 2
    assert len(merge_edges) == 2

    assert graph.nodes[split_edges[0][0]]["track_id"] == 5
    assert graph.nodes[split_edges[0][0]]["t"] == 7
    assert graph.nodes[split_edges[0][1]]["t"] == 8
    assert graph.nodes[split_edges[0][1]]["track_id"] in (7, 12)

    assert graph.nodes[merge_edges[0][0]]["track_id"] in (7, 12)
    assert graph.nodes[merge_edges[0][0]]["t"] == 14
    assert graph.nodes[merge_edges[0][1]]["t"] == 15
    assert graph.nodes[merge_edges[0][1]]["track_id"] == 1

    assert len(graph.edges) == 4 + 6 + 6 + 2 + 14 + 2 + 2
    assert len(graph.nodes) == 5 + 7 + 7 + 3 + 15

    components = list(nx.connected_components(graph.to_undirected()))
    assert len(components) == 2  # Indep tracks + the others


def test_from_tracks_edge_with_merge_and_split_label() -> None:
    tracks = [
        byotrack.Track(3, torch.rand(5, 3), identifier=5, merge_id=7),
        byotrack.Track(2, torch.rand(6, 3), identifier=6, merge_id=7),
        byotrack.Track(8, torch.rand(7, 3), identifier=7, parent_id=5),
        byotrack.Track(8, torch.rand(7, 3), identifier=12, parent_id=5),
    ]

    graph = byotrack.TrackingGraph.from_tracks(tracks)
    split_and_merge_edges = [
        (u, v, d) for u, v, d in graph.edges(data=True) if d.get("split", False) and d.get("merge", False)
    ]

    assert len(split_and_merge_edges) == 1

    components = list(nx.connected_components(graph.to_undirected()))
    assert len(components) == 1


def test_from_tracks_weird_empty_track_raises() -> None:
    track = byotrack.Track(0, torch.rand(1, 2))
    track.points = torch.rand(0, 2)  # Overwrite point with incorrect data

    with pytest.raises(ValueError, match="Empty tracks are not supported"):
        byotrack.TrackingGraph.from_tracks([track])


def test_from_tracks_without_edges():
    tracks = [
        byotrack.Track(0, torch.rand(1, 2)),
        byotrack.Track(3, torch.rand(1, 2)),
    ]
    graph = byotrack.TrackingGraph.from_tracks(tracks)

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 0


## Creation from nx


def test_from_nx_basic_remapping():
    nx_graph: nx.DiGraph = nx.DiGraph()
    nx_graph.add_node(0, frame=0, row=1.0, col=2.0)
    nx_graph.add_node(1, frame=1, row=3.0, col=4.0)
    nx_graph.add_edge(0, 1)
    graph = byotrack.TrackingGraph.from_nx(nx_graph, frame_key="frame", y_key="row", x_key="col")

    assert len(graph.nodes) == 2
    assert graph.nodes[0]["t"] == 0
    assert graph.nodes[0]["y"] == 1.0
    assert graph.nodes[0]["x"] == 2.0


def test_from_nx_backward_edge_raises():
    nx_graph: nx.DiGraph = nx.DiGraph()
    nx_graph.add_node(0, t=5, y=1.0, x=2.0)
    nx_graph.add_node(1, t=3, y=3.0, x=4.0)
    nx_graph.add_edge(0, 1)  # backward: t=5 → t=3

    with pytest.raises(ValueError, match="Backward edge are not supported"):
        byotrack.TrackingGraph.from_nx(nx_graph)


def test_from_nx_auto_split_from_out_degree():
    nx_graph: nx.DiGraph = nx.DiGraph()
    nx_graph.add_node(0, t=0, y=1.0, x=2.0)
    nx_graph.add_node(1, t=1, y=3.0, x=4.0)
    nx_graph.add_node(2, t=1, y=5.0, x=6.0)
    nx_graph.add_edge(0, 1)
    nx_graph.add_edge(0, 2)  # out-degree 2 => auto-split
    graph = byotrack.TrackingGraph.from_nx(nx_graph)

    assert graph.edges[(0, 1)].get("split", False)
    assert graph.edges[(0, 2)].get("split", False)


def test_from_nx_auto_merge_from_in_degree():
    nx_graph: nx.DiGraph = nx.DiGraph()
    nx_graph.add_node(0, t=0, y=1.0, x=2.0)
    nx_graph.add_node(1, t=0, y=3.0, x=4.0)
    nx_graph.add_node(2, t=1, y=5.0, x=6.0)
    nx_graph.add_edge(0, 2)
    nx_graph.add_edge(1, 2)  # in-degree 2 => auto-merge
    graph = byotrack.TrackingGraph.from_nx(nx_graph)

    assert graph.edges[(0, 2)].get("merge", False)
    assert graph.edges[(1, 2)].get("merge", False)


def test_from_nx_explicit_split_flag():
    nx_graph: nx.DiGraph = nx.DiGraph()
    nx_graph.add_node(0, t=0, y=1.0, x=2.0)
    nx_graph.add_node(1, t=1, y=3.0, x=4.0)
    nx_graph.add_edge(0, 1, split=True)
    graph = byotrack.TrackingGraph.from_nx(nx_graph)
    assert graph.edges[(0, 1)].get("split", False)


def test_from_nx_explicit_merge_flag():
    nx_graph: nx.DiGraph = nx.DiGraph()
    nx_graph.add_node(0, t=0, y=1.0, x=2.0)
    nx_graph.add_node(1, t=1, y=3.0, x=4.0)
    nx_graph.add_edge(0, 1, merge=True)
    graph = byotrack.TrackingGraph.from_nx(nx_graph)
    assert graph.edges[(0, 1)].get("merge", False)


def test_from_nx_optional_keys():
    nx_graph: nx.DiGraph = nx.DiGraph()
    nx_graph.add_node(0, t=0, y=1.0, x=2.0, z=3.0, track_id=42, detection_id=40)
    graph = byotrack.TrackingGraph.from_nx(nx_graph)
    assert graph.nodes[0]["z"] == 3.0
    assert graph.nodes[0]["track_id"] == 42
    assert graph.nodes[0]["detection_id"] == 40


def test_from_nx_optional_keys_undefined_by_default():
    nx_graph: nx.DiGraph = nx.DiGraph()
    nx_graph.add_node(0, t=0, y=1.0, x=2.0)
    graph = byotrack.TrackingGraph.from_nx(nx_graph)
    assert "z" not in graph.nodes[0]
    assert "track_id" not in graph.nodes[0]
    assert "detection_id" not in graph.nodes[0]


def test_from_nx_extra_keys_propagated():
    nx_graph: nx.DiGraph = nx.DiGraph()
    nx_graph.add_node(0, t=0, y=1.0, x=2.0, extra="extra")
    nx_graph.add_node(1, t=1, y=1.0, x=2.0)
    nx_graph.add_edge(0, 1, extra="extra")
    graph = byotrack.TrackingGraph.from_nx(nx_graph)
    assert graph.nodes[0]["extra"] == "extra"
    assert graph.edges[0, 1]["extra"] == "extra"


## to_tracks with round trip


def test_to_tracks_empty_graph():
    graph = byotrack.TrackingGraph()

    tracks = graph.to_tracks()

    assert len(tracks) == 0


def test_to_tracks_single_track_2d():
    points = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    track = byotrack.Track(0, points, identifier=100)
    graph = byotrack.TrackingGraph.from_tracks([track])
    reconstructed = graph.to_tracks()

    assert len(reconstructed) == 1

    r = reconstructed[0]

    assert r.start == track.start
    assert r.identifier == track.identifier
    assert len(r) == len(track)
    assert torch.allclose(r.points, points)


def test_to_tracks_single_track_3d():
    points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    track = byotrack.Track(15, points)
    graph = byotrack.TrackingGraph.from_tracks([track])
    reconstructed = graph.to_tracks()

    assert len(reconstructed) == 1

    r = reconstructed[0]

    assert r.start == track.start
    assert r.identifier == track.identifier
    assert len(r) == len(track)
    assert torch.allclose(r.points, points)


def test_to_tracks_two_independent_tracks():
    track_1 = byotrack.Track(3, torch.ones(4, 2))
    track_2 = byotrack.Track(5, torch.ones(2, 2) * 2)
    graph = byotrack.TrackingGraph.from_tracks([track_1, track_2])
    reconstructed = graph.to_tracks()

    assert len(reconstructed) == 2

    # The order is preserved in the round trip
    assert reconstructed[0].start == track_1.start
    assert reconstructed[1].start == track_2.start
    assert torch.allclose(reconstructed[0].points, track_1.points)
    assert torch.allclose(reconstructed[1].points, track_2.points)


def test_to_tracks_single_split():
    parent = byotrack.Track(1, torch.ones(3, 2), identifier=5)
    child = byotrack.Track(4, torch.ones(2, 2), identifier=3, parent_id=5)
    graph = byotrack.TrackingGraph.from_tracks([child, parent])  # Child first

    with pytest.warns(UserWarning, match="splits into 1 ! = 2 tracks"):
        reconstructed = graph.to_tracks()

    assert len(reconstructed) == 2

    assert reconstructed[0].start == child.start
    assert reconstructed[1].start == parent.start
    assert reconstructed[0].parent_id == 5
    assert reconstructed[0].merge_id == -1
    assert reconstructed[1].parent_id == -1
    assert reconstructed[1].merge_id == -1


def test_to_tracks_single_merge():
    parent = byotrack.Track(1, torch.ones(3, 2), identifier=5, merge_id=3)
    child = byotrack.Track(4, torch.ones(2, 2), identifier=3)
    graph = byotrack.TrackingGraph.from_tracks([child, parent])  # Child first

    with pytest.warns(UserWarning, match="results of 1 ! = 2 tracks"):
        reconstructed = graph.to_tracks()

    assert len(reconstructed) == 2

    assert reconstructed[0].start == child.start
    assert reconstructed[1].start == parent.start
    assert reconstructed[0].parent_id == -1
    assert reconstructed[0].merge_id == -1
    assert reconstructed[1].parent_id == -1
    assert reconstructed[1].merge_id == 3


def test_to_tracks_merge_and_split_typical():
    tracks = [
        byotrack.Track(3, torch.rand(5, 3), identifier=5),
        byotrack.Track(8, torch.rand(7, 3), identifier=7, parent_id=5, merge_id=1),
        byotrack.Track(8, torch.rand(7, 3), identifier=12, parent_id=5, merge_id=1),
        byotrack.Track(15, torch.rand(3, 3), identifier=1),
        byotrack.Track(0, torch.rand(15, 3), identifier=0),  # Indep track
    ]

    graph = byotrack.TrackingGraph.from_tracks(tracks)
    reconstructed = graph.to_tracks()

    assert len(reconstructed) == 5

    # Order is preserved as well as data
    for track, r_track in zip(tracks, reconstructed, strict=True):
        assert track.identifier == r_track.identifier
        assert track.start == r_track.start
        assert torch.allclose(track.points, r_track.points)
        assert track.merge_id == r_track.merge_id
        assert track.parent_id == r_track.parent_id


def test_to_tracks_nan_gap_reconstructed():
    points = torch.tensor([[1.0, 2.0], [float("nan"), float("nan")], [float("nan"), float("nan")], [3.0, 4.0]])
    track = byotrack.Track(0, points)
    graph = byotrack.TrackingGraph.from_tracks([track])
    reconstructed = graph.to_tracks()

    assert len(reconstructed) == 1

    r = reconstructed[0]

    assert r.start == track.start
    assert len(r) == len(track)
    assert torch.allclose(r.points, track.points, equal_nan=True)


def test_to_tracks_nan_border_reconstructed():
    points = torch.tensor([[float("nan"), float("nan")], [1.0, 2.0], [float("nan"), float("nan")]])
    track = byotrack.Track(0, points)
    graph = byotrack.TrackingGraph.from_tracks([track])
    reconstructed = graph.to_tracks()

    assert len(reconstructed) == 1

    r = reconstructed[0]

    assert r.start == track.start
    assert len(r) == len(track)
    assert torch.allclose(r.points, track.points, equal_nan=True)


## to_tracks with more general graphs than from_tracks


def test_to_tracks_from_graph_with_gap():
    graph = byotrack.TrackingGraph()
    graph.add_node(0, t=1, y=1.0, x=2.0)
    graph.add_node(1, t=4, y=1.0, x=2.0)
    graph.add_edge(0, 1)

    tracks = graph.to_tracks()

    assert len(tracks) == 1
    assert tracks[0].start == 1
    assert len(tracks[0]) == 4
    assert torch.isnan(tracks[0].points[1:3]).all()


def test_to_tracks_with_unlabelled_split():
    graph = byotrack.TrackingGraph()
    graph.add_node(0, t=1, y=1.0, x=2.0)
    graph.add_node(1, t=2, y=1.0, x=2.0)
    graph.add_node(2, t=2, y=1.0, x=2.0)
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)

    tracks = graph.to_tracks()

    assert len(tracks) == 3
    assert tracks[1].parent_id == tracks[2].parent_id != -1


def test_to_tracks_with_unlabelled_merge():
    graph = byotrack.TrackingGraph()
    graph.add_node(0, t=1, y=1.0, x=2.0)
    graph.add_node(1, t=1, y=1.0, x=2.0)
    graph.add_node(2, t=2, y=1.0, x=2.0)
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)

    tracks = graph.to_tracks()

    assert len(tracks) == 3
    assert tracks[0].merge_id == tracks[1].merge_id != -1


def test_to_tracks_with_multiple_track_id_warns_and_use_first():
    graph = byotrack.TrackingGraph()
    graph.add_node(0, t=1, y=1.0, x=2.0, track_id=5)
    graph.add_node(1, t=2, y=1.0, x=2.0)
    graph.add_node(2, t=3, y=1.0, x=2.0, track_id=7)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    with pytest.warns(UserWarning, match="Found two different track_ids"):
        tracks = graph.to_tracks()

    assert len(tracks) == 1
    assert tracks[0].identifier == 5


def test_to_tracks_with_single_track_id_with_undefined():
    graph = byotrack.TrackingGraph()
    graph.add_node(0, t=1, y=1.0, x=2.0, track_id=5)
    graph.add_node(1, t=2, y=1.0, x=2.0)
    graph.add_node(2, t=3, y=1.0, x=2.0, track_id=5)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    tracks = graph.to_tracks()

    assert len(tracks) == 1
    assert tracks[0].identifier == 5


def test_to_tracks_with_non_unique_track_id_warns_and_relabel():
    graph = byotrack.TrackingGraph()
    graph.add_node(0, t=1, y=1.0, x=2.0, track_id=5)
    graph.add_node(1, t=2, y=1.0, x=2.0, track_id=5)

    with pytest.warns(UserWarning, match="Found duplicate track_id 5\\. Relabeled into"):
        tracks = graph.to_tracks()

    assert len(tracks) == 2
    assert tracks[0].identifier == 5
    assert tracks[1].identifier != 5
