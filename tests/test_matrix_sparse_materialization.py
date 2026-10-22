import pytest
from groggy import Graph


def test_large_sparse_spectral_call_requires_explicit_dense_materialization():
    graph = Graph()
    for _ in range(2048):
        graph.add_node()
    matrix = graph.view().adj(sparse=True)
    assert matrix.is_sparse
    for method in ['svd', 'svd_thin', 'qr_decomposition', 'eigenvalue_decomposition', 'rank', 'condition_number']:
        with pytest.raises(RuntimeError, match='to_dense'):
            getattr(matrix, method)()
