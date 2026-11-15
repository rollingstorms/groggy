"""Selected retained-history queries follow surviving pending frame membership."""
import pytest
from groggy import Graph, Pipeline
from tests.test_pending_core_compute import attach, program, step


def fixture(directed=False):
    graph = Graph(directed=directed)
    graph.add_nodes([{"score": 1}, {"score": 2}, {"score": 3}])
    first = graph.current_state_id
    graph.add_edges([(0, 1), (1, 2), (2, 2)])
    previous = graph.current_state_id
    graph.set_node_attr(0, "score", 5)
    return graph, first, previous, graph.current_state_id


def pending():
    return [step("graph.set_edge_endpoints", revisions=[[0, 0, 2]]),
            step("graph.remove_nodes", ids=[1]),
            step("core.init_nodes", target="values", value=99), attach("values", "score")]


def run(graph, operations, outer=False, frame="flat"):
    if frame == "loop":
        operations = [step("iter.loop", iterations=2, body=operations)]
    elif frame == "foreach":
        operations = [step("sample.iterate_nodes", target="seeds"),
                      step("sample.for_each", source="seeds", target="mapped", body=operations)]
    if outer:
        return Pipeline([program(*pending()), program(*operations)])(graph.view(), atomic=True)
    return graph.apply(program(*(pending() + operations)))


def query(case, first, previous, head):
    if case in ["count", "sum", "avg", "min", "max", "first", "last"]:
        return step("temporal.window_aggregate", attr="score", function=case, start=first, end=head, output="history")
    if case == "existed":
        return step("temporal.filter", predicate="existed_at", commit=first, output="history")
    if case == "created_after":
        return step("temporal.filter", predicate="created_after", commit=0, output="history")
    if case == "created_before":
        return step("temporal.filter", predicate="created_before", commit=first, output="history")
    if case == "modified":
        return step("temporal.filter", predicate="modified_in_range", start=previous, end=head, output="history")
    if case.startswith("changed_"):
        return step("temporal.mark_changed_nodes", change_type=case[8:], start=first, end=head, output="history")
    return step("temporal.window", start=0 if case == "empty_root" else first,
                end=0 if case == "empty_root" else head, output="history")


CASES = [("count", [2, 1]), ("sum", [6.0, 3.0]), ("avg", [3.0, 3.0]),
         ("min", [1.0, 3.0]), ("max", [5.0, 3.0]), ("first", [1, 3]),
         ("last", [5, 3]), ("existed", [1, 1]), ("created_after", [1, 1]),
         ("created_before", [0, 0]), ("modified", [1, 0]),
         ("changed_created", [1, 1]), ("changed_deleted", [0, 0]),
         ("changed_modified", [1, 1]), ("changed_any", [1, 1]),
         ("window", [1, 1]), ("empty_root", [0, 0])]


@pytest.mark.parametrize("case,expected", CASES)
@pytest.mark.parametrize("directed", [False, True])
def test_temporal_queries_use_surviving_ids_but_accepted_history_values(case, expected, directed):
    graph, first, previous, head = fixture(directed)
    old = graph.read_view()
    run(graph, [query(case, first, previous, head), attach("history", "answer")])
    assert [graph.get_node_attr(n, "answer") for n in [0, 2]] == expected
    assert sorted(graph.node_ids) == [0, 2]
    assert sorted(graph.edge_ids) == [0, 2]
    assert graph.edge_endpoints(0) == (0, 2)
    assert old.edge_endpoints(0) == (0, 1)
    assert [graph.get_node_attr(n, "score") for n in [0, 2]] == [99, 99]
    assert graph.current_state_id == head + 1


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("frame", ["loop", "foreach"])
def test_all_selected_temporal_helpers_compose_in_nested_and_outer_frames(outer, frame):
    graph, first, previous, head = fixture(True)
    operations = []
    for case in ["sum", "existed", "changed_deleted", "window"]:
        operations.extend([query(case, first, previous, head), attach("history", case)])
    run(graph, operations, outer, frame)
    for name, expected in [("sum", [6.0, 3.0]), ("existed", [1, 1]),
                           ("changed_deleted", [0, 0]), ("window", [1, 1])]:
        assert [graph.get_node_attr(n, name) for n in [0, 2]] == expected
    assert graph.current_state_id == head + 1


@pytest.mark.parametrize("case", ["sum", "existed", "changed_any", "window"])
@pytest.mark.parametrize("frame", ["flat", "foreach"])
def test_provisional_nodes_reject_without_inventing_retained_lifecycle(case, frame):
    graph, first, previous, head = fixture()
    operations = [step("graph.add_nodes", count=1, output="new"),
                  step("core.init_nodes", target="prefix", value=9), attach("prefix", "prefix")]
    query_steps = [query(case, first, previous, head), attach("history", "answer")]
    if frame == "foreach":
        query_steps = [step("sample.iterate_nodes", target="seeds"),
                       step("sample.for_each", source="seeds", target="mapped", body=query_steps)]
    with pytest.raises(RuntimeError, match="unknown to the owner/retained lineage"):
        graph.apply(program(*(operations + query_steps)))
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "prefix") is None
    assert graph.get_node_attr(0, "answer") is None
    assert graph.add_node() == 3


@pytest.mark.parametrize("case", ["sum", "window", "changed_any"])
@pytest.mark.parametrize("outer", [False, True])
def test_invalid_retained_bounds_after_topology_roll_back_the_whole_run(case, outer):
    graph, first, previous, head = fixture()
    operation = query(case, first, previous, head + 100)
    with pytest.raises(RuntimeError, match="lineage"):
        run(graph, [operation, attach("history", "answer")], outer)
    assert graph.current_state_id == head
    assert graph.has_node(1)
    assert graph.edge_endpoints(0) == (0, 1)
    assert graph.get_node_attr(0, "score") == 5
    assert graph.get_node_attr(0, "answer") is None
