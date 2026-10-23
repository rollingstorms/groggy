"""Numerical exports consume captured values and selected edge instances."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import numpy as np
import pytest


def array(matrix):
    return np.asarray(matrix.to_numpy()).reshape(matrix.shape)


@pytest.mark.parametrize("directed", [False, True])
def test_weighted_matrix_retains_captured_scope_and_labels(directed):
    graph = gr.Graph(directed=directed)
    a, removed, b, isolated = graph.add_nodes(4)
    graph.remove_node(removed)
    first = graph.add_edge(a, b)
    excluded = graph.add_edge(a, b)
    loop = graph.add_edge(a, a)
    graph.set_edge_attr(first, "weight", 2.5)
    graph.set_edge_attr(excluded, "weight", 100)
    graph.set_edge_attr(loop, "weight", 3)
    view = graph.read_view().select_edges([first, loop])
    graph.set_edge_attr(first, "weight", 999)
    graph.remove_node(a)
    for sparse in [False, True]:
        matrix = view.adjacency_matrix(sparse=sparse, weight_attr="weight")
        assert matrix.is_sparse == sparse
        assert np.array_equal(array(matrix), [[3., 2.5], [0. if directed else 2.5, 0.]])
        assert matrix.row_labels == matrix.columns == [f"node_{a}", f"node_{b}"]
    assert isolated not in view.node_ids


def test_feature_exports_keep_frozen_columns_values_and_zero_imputation():
    graph = gr.Graph()
    a, b, excluded = graph.add_nodes(3)
    graph.set_node_attr(a, "a", 7)
    graph.set_node_attr(a, "b", True)
    graph.set_node_attr(b, "a", "imputed")
    graph.set_node_attr(b, "b", None)
    graph.set_node_attr(excluded, "other", 99)
    view = graph.read_view().select_nodes([a, b])
    matrix = view.to_matrix()
    assert matrix.columns == ["a", "b"]
    assert matrix.row_labels == [f"node_{a}", f"node_{b}"]
    assert np.array_equal(array(matrix), [[7., 1.], [0., 0.]])
    graph.set_node_attr(a, "a", 999)
    assert np.array_equal(array(view.to_matrix()), array(matrix))


def test_missing_null_and_text_weights_follow_existing_graph_conversion():
    graph = gr.Graph()
    a, b = graph.add_nodes(2)
    first = graph.add_edge(a, b)
    graph.add_edge(a, b)
    graph.set_edge_attr(first, "weight", None)
    assert array(graph.read_view().adjacency_matrix(weight_attr="weight"))[0, 1] == 1.
    graph.set_edge_attr(first, "weight", "1.75")
    view = graph.read_view()
    assert array(view.adjacency_matrix(weight_attr="weight"))[0, 1] == 2.75
    graph.set_edge_attr(first, "weight", "invalid")
    with pytest.raises(RuntimeError):
        graph.read_view().adjacency_matrix(weight_attr="weight")
    assert array(view.adjacency_matrix(weight_attr="weight"))[0, 1] == 2.75


def test_empty_shapes_and_read_worker_matrix_exports():
    graph = gr.Graph()
    view = graph.read_view()
    assert view.to_matrix().shape == (0, 0)
    assert view.adjacency_matrix().shape == (0, 0)
    node = graph.add_node()
    assert graph.read_view().to_matrix().shape == (1, 0)
    graph.set_node_attr(node, "value", 3)
    view = graph.read_view()
    with ThreadPoolExecutor(max_workers=2) as workers:
        sums = list(workers.map(lambda _: float(array(view.to_matrix()).sum()), range(4)))
    assert sums == [3.] * 4
