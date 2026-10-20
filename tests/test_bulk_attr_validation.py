import pytest

from groggy import Graph


def test_node_bulk_reports_all_invalid_ids_and_preserves_values():
    graph = Graph()
    node = graph.add_node()

    with pytest.raises(ValueError) as error:
        graph.set_node_attrs(
            {
                "score": {
                    "nodes": [node, 100, node, 101],
                    "values": [1, 2, 3, 4],
                }
            }
        )

    message = str(error.value)
    assert "score[1]: node 100" in message
    assert "score[2]: duplicate node" in message
    assert "score[3]: node 101" in message
    assert graph.get_node_attr(node, "score") is None


def test_edge_bulk_reports_all_invalid_ids_and_preserves_values():
    graph = Graph()
    source, target = graph.add_nodes(2)
    edge = graph.add_edge(source, target)

    with pytest.raises(ValueError) as error:
        graph.set_edge_attrs({"weight": {edge: 1.0, 200: 2.0, 201: 3.0}})

    message = str(error.value)
    assert "weight[1]: edge 200" in message
    assert "weight[2]: edge 201" in message
    assert graph.get_edge_attr(edge, "weight") is None


def test_node_bulk_cannot_change_system_entity_type():
    graph = Graph()
    node = graph.add_node()
    with pytest.raises(ValueError, match="entity_type"):
        graph.set_node_attrs({"entity_type": {node: "meta"}})
    assert graph.get_node_attr(node, "entity_type") == "base"
