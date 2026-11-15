"""Property primitives share pending topology without publishing intermediate state."""
import pytest
from groggy import Graph, Pipeline
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


def builder(*steps):
    return step_pipeline("pending properties", list(steps), atomic=True)


def calculations():
    return [
        stage("core.load_node_attr", target="score", attr="score", default=2.0),
        stage("core.init_scalar", target="two", value=2.0),
        stage("core.mul", left="score", right="two", target="scaled"),
        stage("core.attach_node_attr", source="scaled", attr="scaled"),
        stage("core.load_node_attr", target="again", attr="scaled"),
        stage("core.add", left="again", right="two", target="plus"),
        stage("core.attach_node_attr", source="plus", attr="plus"),
        stage("core.init_nodes_with_index", target="index"),
        stage("core.attach_node_attr", source="index", attr="position"),
        stage("core.edge_weight_scale", attr="weight", factor=2.0, target="weights"),
        stage("core.attach_edge_attr", source="weights", attr="scaled_weight"),
    ]


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("outer", [False, True])
def test_properties_follow_additions_removals_revisions_and_selected_ids(directed, outer):
    graph = Graph(directed=directed)
    graph.add_nodes(5)
    graph.remove_node(1)
    graph.add_edges([(0, 2), (0, 2), (2, 2), (3, 4)])
    old = graph.view()
    head = graph.current_state_id
    topology = [
        stage("graph.add_nodes", count=1, output="new", attributes={"score": 1.0000000000000002}),
        stage("graph.add_edges", edges=[[0, {"added": 0}]], output="new_edges", attributes={"weight": 3.5}),
        stage("graph.set_edge_endpoints", revisions=[[0, 0, 0]]),
        stage("graph.remove_nodes", ids=[2]),
        stage("graph.capture_staged", output="before"),
        stage("graph.emit", source="before", name="before"),
    ]
    suffix = calculations() + [
        stage("graph.capture_staged", output="after"),
        stage("graph.emit", source="after", name="after"),
    ]
    selected = graph.induced_subgraph([0, 2])
    if outer:
        result, profile = Pipeline([builder(*topology), builder(*suffix)])(
            selected, atomic=True, return_profile=True)
    else:
        result, profile = selected.apply(builder(*(topology + suffix)), return_profile=True)
    assert graph.current_state_id == head + 1
    assert sorted(result.node_ids) == [0, 5]
    assert sorted(result.edge_ids) == [0, 4]
    assert graph.get_node_attr(0, "scaled") == 4.0
    assert graph.get_node_attr(5, "scaled") == 2.0000000000000004
    assert graph.get_node_attr(5, "plus") == 4.0
    assert graph.get_node_attr(0, "position") == 0
    assert graph.get_node_attr(5, "position") == 1
    assert graph.get_node_attr(3, "scaled") is None
    assert graph.get_edge_attr(0, "scaled_weight") == 2.0
    assert graph.get_edge_attr(4, "scaled_weight") == 7.0
    assert graph.get_edge_attr(3, "scaled_weight") is None
    assert profile["outputs"]["before"].node_attr(5, "scaled") is None
    assert profile["outputs"]["after"].node_attr(5, "scaled") == 2.0000000000000004
    assert old.get_node_attribute(0, "scaled") is None
    assert old.edge_endpoints(0) == (0, 2)


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("failure", ["stale_map", "entity_type", "external_id", "labels", "late_target"])
def test_invalid_property_targets_and_kinds_roll_back_entire_pending_run(outer, failure):
    graph = Graph()
    graph.add_nodes(2)
    head = graph.current_state_id
    steps = [stage("core.init_nodes", target="old", value=7),
             stage("graph.add_nodes", count=1, output="created"),
             stage("graph.remove_nodes", ids=[1]),
             stage("core.init_nodes", target="new", value=7),
             stage("core.attach_node_attr", source="new", attr="valid")]
    if failure == "stale_map":
        steps.append(stage("core.attach_node_attr", source="old", attr="stale"))
    elif failure == "late_target":
        steps.append(stage("graph.remove_edges", ids=[999]))
    else:
        steps.append(stage("core.attach_node_attr", source="new", attr=failure))
    with pytest.raises((ValueError, RuntimeError)):
        if outer:
            Pipeline([builder(*steps)])(graph.view(), atomic=True)
        else:
            graph.apply(builder(*steps))
    assert graph.current_state_id == head
    assert sorted(graph.node_ids) == [0, 1]
    assert graph.get_node_attr(0, "valid") is None
    assert not graph.has_uncommitted_changes()
    assert graph.add_node() == 2


def test_empty_selection_can_initialize_added_entities_and_counts():
    graph = Graph()
    result = graph.apply(builder(
        stage("graph.add_nodes", count=2, output="new"),
        stage("graph.add_edges", edges=[[{"added": 0}, {"added": 1}]], output="edges"),
        stage("core.graph_node_count", target="count"),
        stage("core.init_nodes", target="values", value=3),
        stage("core.add", left="values", right="count", target="total"),
        stage("core.attach_node_attr", source="total", attr="total"),
        stage("core.init_edges", target="weights", value=5),
        stage("core.attach_edge_attr", source="weights", attr="weight"),
    ))
    assert sorted(result.node_ids) == [0, 1]
    assert [graph.get_node_attr(n, "total") for n in [0, 1]] == [5, 5]
    assert graph.get_edge_attr(0, "weight") == 5


def test_pending_scope_refreshes_after_later_topology_changes():
    graph = Graph()
    graph.add_nodes(2)
    graph.apply(builder(
        stage("graph.add_nodes", count=1, output="first"),
        stage("core.init_nodes", target="first_values", value=1),
        stage("graph.add_nodes", count=1, output="second"),
        stage("core.init_nodes_with_index", target="indices"),
        stage("graph.remove_nodes", ids=[1]),
        stage("core.init_nodes", target="current", value=9),
        stage("core.attach_node_attr", source="current", attr="value"),
        stage("core.graph_node_count", target="count"),
        stage("core.add", left="current", right="count", target="counted"),
        stage("core.attach_node_attr", source="counted", attr="counted"),
    ))
    assert sorted(graph.node_ids) == [0, 2, 3]
    assert [graph.get_node_attr(n, "counted") for n in [0, 2, 3]] == [12, 12, 12]
