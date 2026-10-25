"""Empty table collections keep source ownership and declared schema."""
import pytest
from groggy import Graph


def fixture():
    graph = Graph()
    graph.add_nodes([{"k": 2, "x": 10, "text": "a"}, {"k": 1, "x": 11, "text": "b"}, {"k": 2, "x": 12, "text": "c"}])
    table = graph.read_view().nodes_table().base_table()
    return graph, table


@pytest.mark.parametrize("mode", ["take", "skip", "filter"])
def test_empty_collection_selection_retains_source_and_numeric_schema(mode):
    graph, table = fixture()
    groups = table.groupby("k")
    empty = groups.take(0) if mode == "take" else groups.skip(10) if mode == "skip" else groups.filter(lambda _: False)
    assert len(empty) == 0
    flattened = empty.concat()
    assert len(flattened) == 0
    assert len(flattened.source_views()) == 1
    assert flattened.source_entity_ids is None
    assert empty.sum()["x"].to_list() == []
    with pytest.raises(KeyError):
        empty.sum()["text"]
    graph.set_node_attr(0, "x", 99)
    assert flattened.source_views()[0].node_attr(0, "x") == 10


@pytest.mark.parametrize("method", ["group_by", "groupby"])
def test_grouping_empty_sourced_tables_retains_keys_and_source(method):
    _, table = fixture()
    groups = getattr(table.head(0), method)("k")
    assert len(groups) == 0
    result = groups.agg({"x": "sum"})
    assert result["k"].to_list() == []
    assert result["x"].to_list() == []
    assert len(result.source_views()) == 1


def test_empty_projection_rename_and_drop_apply_to_retained_schema():
    _, table = fixture()
    empty = table.groupby("k").take(0).select(["x"]).rename({"x": "score", "k": "kind"})
    assert empty.sum()["kind"].to_list() == []
    assert empty.sum()["score"].to_list() == []
    assert len(empty.concat().source_views()) == 1
    with pytest.raises(ValueError):
        empty.select(["missing"])
    with pytest.raises(ValueError):
        empty.rename({"score": "kind"})
    assert len(empty.drop_columns(["score"]).concat()) == 0


def test_filter_calls_once_per_table_and_propagates_callback_exception():
    _, table = fixture()
    groups = table.groupby("k")
    calls = []

    def select(group):
        calls.append(group)
        return len(calls) == 1

    assert len(groups.filter(select)) == 1
    assert len(calls) == 2

    def fail(_):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        groups.filter(fail)
    assert len(groups) == 2


def test_map_does_not_call_python_on_an_invented_empty_table():
    _, table = fixture()
    empty = table.groupby("k").take(0)

    def fail(_):
        raise AssertionError("must not run")

    assert len(empty.map(fail).concat().source_views()) == 1
    assert len(empty.apply(fail).sum().source_views()) == 1


def test_row_empty_groups_keep_their_keys_without_selecting_text_for_numeric_reduction():
    _, table = fixture()
    result = table.groupby("k").head(0).sum()
    assert result["k"].to_list() == [2, 1]
    assert result["x"].to_list() == [None, None]
    with pytest.raises(KeyError):
        result["text"]


def test_empty_specialized_node_and_edge_group_adapters_keep_source_capture():
    graph, _ = fixture()
    graph.add_edges([(0, 1, {"k": 1, "weight": 2})])
    view = graph.read_view()
    for table, value in [(view.nodes_table(), "x"), (view.edges_table(), "weight")]:
        grouped = table.head(0).group_by("k")
        result = grouped.agg({value: "sum"})
        assert result[value].to_list() == []
        assert len(result.source_views()) == 1
