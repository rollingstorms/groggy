"""Classic registry handles share outer staging with builder algorithms."""
import pytest
from groggy import Graph, Pipeline
from groggy.algorithms.base import RustAlgorithmHandle
from groggy.builder import AlgorithmBuilder


CLASSIC = [
    "centrality.pagerank", "centrality.closeness", "centrality.betweenness",
    "community.louvain", "community.leiden", "community.girvan_newman",
    "community.lpa", "community.connected_components", "community.infomap",
    "pathfinding.astar", "pathfinding.bfs", "pathfinding.dfs", "pathfinding.dijkstra",
]


@pytest.mark.parametrize("algorithm_id", CLASSIC)
def test_classic_handles_share_one_acceptance_and_rollback_with_builders(algorithm_id):
    graph = Graph()
    graph.add_nodes([{"start": False}, {}, {"goal": None}, {}])
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    selected = graph.induced_subgraph([0, 1, 2])
    params = {"output_attr": "result"}
    if algorithm_id.startswith("pathfinding."):
        params["start_attr"] = "start"
    if algorithm_id == "pathfinding.astar":
        params["goal_attr"] = "goal"
    algorithm = RustAlgorithmHandle(algorithm_id, params)
    prefix = AlgorithmBuilder("prefix")
    prefix.attach_as("prefix", prefix.init_nodes(default=7))
    suffix = AlgorithmBuilder("copy result")
    suffix.attach_as("copy", suffix.load_attr("result"))
    pipe = Pipeline([prefix.build(atomic=True), algorithm, suffix.build(atomic=True)])
    before = graph.current_state_id
    old = graph.read_view()
    pipe(selected, atomic=True)
    assert graph.current_state_id == before + 1
    assert not graph.has_uncommitted_changes()
    for node in range(3):
        assert graph.get_node_attr(node, "result") is not None
        assert graph.get_node_attr(node, "copy") == graph.get_node_attr(node, "result")
    assert graph.get_node_attr(3, "result") is None
    assert old.node_attr(0, "result") is None

    invalid = AlgorithmBuilder("invalid identity")
    invalid.attach_as("entity_type", invalid.init_nodes(default=3))
    before = graph.current_state_id
    prior = [graph.get_node_attr(n, "result") for n in range(3)]
    with pytest.raises(RuntimeError):
        Pipeline([algorithm, invalid.build(atomic=True)])(selected, atomic=True)
    assert graph.current_state_id == before
    assert [graph.get_node_attr(n, "result") for n in range(3)] == prior
