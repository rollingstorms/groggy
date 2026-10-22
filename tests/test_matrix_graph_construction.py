import numpy as np
import pytest
from groggy import Graph


@pytest.mark.parametrize('sparse', [False, True])
def test_view_adjacency_counts_parallel_edges_and_self_loop_once(sparse):
    graph = Graph()
    a, b, isolate = [graph.add_node() for _ in range(3)]
    graph.add_edge(a, b)
    graph.add_edge(a, b)
    graph.add_edge(a, a)
    matrix = graph.view().adjacency_matrix(sparse=sparse)
    assert matrix.is_sparse == sparse
    assert matrix.columns == [f'node_{i}' for i in [a, b, isolate]]
    assert matrix.row_labels == matrix.columns
    assert np.array_equal(np.asarray(matrix.to_numpy()).reshape(matrix.shape),
                          [[1., 2., 0.], [2., 0., 0.], [0., 0., 0.]])
    alias = graph.view().adj(sparse=sparse, weight_attr='missing')
    assert alias.is_sparse == sparse
    assert alias.get(0, 1) == 2
    # Numerical output owns values even though the source view reads its parent.
    graph.add_edge(a, b)
    assert matrix.get(0, 1) == 2
    assert graph.view().adj(sparse=sparse).get(0, 1) == 3
