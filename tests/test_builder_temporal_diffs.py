"""Retained temporal diffs compose with one atomic property publication."""
import json

import pytest
import groggy as gr
from groggy.algorithms.base import AlgorithmHandle


class Steps(AlgorithmHandle):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        return {"id": self.id, "params": {"steps": gr.AttrValue(json.dumps({"atomic": True, "steps": self.steps}))}}


def snapshots(before, after):
    return [{"id": "temporal.snapshot_at", "params": {"state_id": state, "output": name}}
            for state, name in [(before, "before"), (after, "after")]]


def diffs():
    return [{"id": "temporal.diff_" + kind, "params": {"before": "before", "after": "after", "output_prefix": "diff"}}
            for kind in ["nodes", "edges"]]


def attach(source, attr, edges=False):
    return {"id": "core.attach_edge_attr" if edges else "core.attach_node_attr", "params": {"source": source, "attr": attr}}


def fixture():
    graph = gr.Graph()
    graph.add_nodes([{"score": 1} for _ in range(3)])
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)
    before = graph.current_state_id
    graph.add_node(score=1)
    graph.add_edge(2, 3, weight=1.0)
    graph.set_node_attr(0, "score", 2)
    graph.set_edge_attr(0, "weight", 2.0)
    return graph, before, graph.current_state_id


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_retained_diffs_mark_added_removed_and_modified_with_one_publication(persist, reverse):
    graph, before, after = fixture()
    head = graph.current_state_id
    prefix = [{"id": "core.init_nodes", "params": {"target": "staged", "value": 50}}, attach("staged", "score")]
    topology = "removed" if reverse else "added"
    steps = prefix + snapshots(after if reverse else before, before if reverse else after) + diffs() + [
        attach("diff_nodes_" + topology, "membership"),
        attach("diff_nodes_modified", "modified"),
        attach("diff_edges_" + topology, "membership", True),
        attach("diff_edges_modified", "modified", True),
    ]
    _, profile = graph.view().apply(Steps(steps), persist=persist, return_profile=True)
    if persist:
        assert [graph.get_node_attr(n, "membership") for n in range(4)] == [None, None, None, 1]
        assert [graph.get_node_attr(n, "modified") for n in range(4)] == [1, None, None, None]
        assert [graph.get_edge_attr(n, "membership") for n in range(3)] == [None, None, 1]
        assert [graph.get_edge_attr(n, "modified") for n in range(3)] == [1, None, None]
        assert graph.get_node_attr(1, "score") == 50
    else:
        for key, expected in [("node_attributes.membership", [(3, 1)]), ("node_attributes.modified", [(0, 1)]),
                              ("edge_attributes.membership", [(2, 1)]), ("edge_attributes.modified", [(0, 1)])]:
            assert profile["outputs"][key] == expected
        assert graph.get_node_attr(1, "score") == 1
    assert graph.current_state_id == head + int(persist)


def test_unchanged_snapshot_diff_has_no_modified_entities_or_publication():
    graph, _, after = fixture()
    graph.view().apply(Steps(snapshots(after, after) + diffs() + [attach("diff_nodes_modified", "changed"), attach("diff_edges_modified", "changed", True)]), persist=True)
    assert graph.current_state_id == after
    assert graph.get_node_attr(0, "changed") is None


@pytest.mark.parametrize("kind", ["nodes", "edges"])
@pytest.mark.parametrize("missing", ["before", "after"])
def test_missing_diff_snapshot_discards_prior_property_assignments(kind, missing):
    graph, before, after = fixture()
    steps = [{"id": "core.init_nodes", "params": {"target": "values", "value": 77}}, attach("values", "prefix")]
    steps += snapshots(before, after)
    steps += [{"id": "temporal.diff_" + kind, "params": {"before": "missing" if missing == "before" else "before", "after": "missing" if missing == "after" else "after"}}]
    with pytest.raises(RuntimeError, match="missing"):
        graph.view().apply(Steps(steps), persist=True)
    assert graph.current_state_id == after
    assert graph.get_node_attr(0, "prefix") is None


def test_diff_outputs_include_historical_removed_ids_and_reject_invalid_writeback():
    graph, _, before = fixture()
    graph.remove_node(3)
    after = graph.current_state_id
    steps = snapshots(before, after) + diffs() + [attach("diff_nodes_removed", "removed")]
    with pytest.raises(RuntimeError):
        graph.view().apply(Steps(steps), persist=True)
    assert graph.current_state_id == after


def test_empty_invocation_still_compares_full_owner_retained_states():
    graph, before, after = fixture()
    # Retained diffs intentionally cover the full owner, even for an empty view.
    # Explicit attachment addresses those active owner IDs through the existing boundary.
    selected = graph.view().filter_nodes("score > 100")
    selected.apply(Steps(snapshots(before, after) + diffs() + [attach("diff_nodes_added", "new")]), persist=True)
    assert graph.get_node_attr(3, "new") == 1
    assert graph.current_state_id == after + 1
