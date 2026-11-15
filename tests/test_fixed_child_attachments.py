"""Explicit retained-history outputs cannot escape a fixed child write frame."""
import pytest
from groggy import Graph
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


@pytest.mark.parametrize("kind", ["nodes", "edges"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("atomic", [False, True])
def test_history_diff_attachment_rejects_ids_outside_the_exact_child(kind, nested, persist, atomic):
    graph = Graph()
    graph.add_nodes(3)
    graph.add_edges([(0, 1), (0, 1)])
    before = graph.current_state_id
    if kind == "nodes":
        graph.set_node_attr(2, "value", 7)
    else:
        graph.set_edge_attr(1, "value", 7)
    head = graph.current_state_id
    body = [
        step("temporal.snapshot_at", state_id=before, output="before"),
        step("temporal.snapshot_at", state_id=head, output="after"),
        step("temporal.diff_" + kind, before="before", after="after", output_prefix="diff"),
        step("core.attach_node_attr" if kind == "nodes" else "core.attach_edge_attr",
             source="diff_" + kind + "_modified", attr="escaped"),
    ]
    if nested:
        body = [step("iter.loop", iterations=2, body=body)]
    steps = [step("sample.iterate_" + kind, target="seeds"),
             step("sample.for_each", source="seeds", target="mapped", body=body)]
    if atomic:
        steps[:0] = [step("core.init_nodes", target="prefix", value=9),
                     step("core.attach_node_attr", source="prefix", attr="prefix")]
    with pytest.raises(RuntimeError, match="outside the fixed execution frame"):
        graph.apply(step_pipeline("confined writes", steps, atomic=atomic), persist=persist)
    assert graph.current_state_id == head
    assert [graph.get_node_attr(n, "escaped") for n in range(3)] == [None] * 3
    assert [graph.get_edge_attr(e, "escaped") for e in range(2)] == [None] * 2
    assert graph.get_node_attr(0, "prefix") is None
    # Failed child work reserves no accepted IDs.
    assert graph.add_node() == 3
