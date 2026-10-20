import pytest

from groggy import Graph


def test_bulk_edge_validation_reports_every_bad_row_without_partial_write():
    graph = Graph()
    source, target = graph.add_nodes(2)
    before = len(graph.edges)

    with pytest.raises(ValueError) as error:
        graph.add_edges(
            [
                {"source": source, "target": target},
                {"source": source},
                {"source": 999, "target": target},
            ]
        )

    assert "1:" in str(error.value)
    assert "2:" in str(error.value)
    assert len(graph.edges) == before


def test_bulk_edge_attributes_are_applied_after_validation():
    graph = Graph()
    source, target = graph.add_nodes(2)

    with pytest.raises(ValueError):
        graph.add_edges(
            [
                (source, target, {"weight": 1}),
                (source, 999, {"weight": 2}),
            ]
        )
    assert len(graph.edges) == 0

    assert graph.add_edges([(source, target, {"weight": 1})]) == [0]
