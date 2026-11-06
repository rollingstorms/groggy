"""Component convenience neighborhoods return real frozen one-hop selections."""
from groggy import Graph


def test_component_neighborhoods_are_actual_first_center_subgraphs():
    graph = Graph(directed=True)
    graph.add_nodes([{"x": node} for node in range(6)])
    graph.add_edges([(0, 1), (1, 2), (2, 3), (4, 5), (0, 1)])
    components = graph.connected_components()
    graph.set_node_attr(0, "x", 99)
    graph.remove_node(1)
    neighborhoods = components.neighborhood()
    rows = sorted((item.node_ids.to_list(), item.edge_ids.to_list(), item["x"].to_list()) for item in neighborhoods)
    assert rows == [([0, 1], [0, 4], [0, 1]), ([4, 5], [3], [4, 5])]
    assert sorted(component.node_count() for component in components) == [2, 4]
    before = graph.current_state_id
    components.neighborhood()
    assert graph.current_state_id == before


def test_empty_and_isolated_components_have_explicit_neighborhoods():
    graph = Graph()
    assert len(graph.connected_components().neighborhood()) == 0
    graph.add_node(x=7)
    neighborhoods = graph.connected_components().neighborhood()
    assert len(neighborhoods) == 1
    assert neighborhoods[0].node_ids.to_list() == [0]
    assert neighborhoods[0]["x"].to_list() == [7]
