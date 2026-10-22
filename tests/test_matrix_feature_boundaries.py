from groggy import Graph


def test_feature_export_orders_columns_and_preserves_node_labels_and_missing_cells():
    graph = Graph()
    a, b = graph.add_node(), graph.add_node()
    graph.set_node_attr(a, 'z', 8)
    graph.set_node_attr(a, 'a', 3)
    graph.set_node_attr(b, 'a', 'not numeric')
    matrix = graph.to_matrix()
    assert matrix.columns == ['a', 'z']
    assert matrix.row_labels == [f'node_{a}', f'node_{b}']
    assert matrix.get(0, 0) == 3 and matrix.get(0, 1) == 8
    assert matrix.get(1, 0) == 0 and matrix.get(1, 1) == 0
