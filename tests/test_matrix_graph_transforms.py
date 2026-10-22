import numpy as np
import pytest
from groggy import GraphMatrix, Graph


def core(a):
    return GraphMatrix.from_data(np.asarray(a, dtype=float).T.tolist())


def array(a):
    return np.asarray(a.to_numpy()).reshape(a.shape)


@pytest.mark.parametrize('sparse', [False, True])
def test_graph_matrix_operators_match_independent_weighted_formulas(sparse):
    a = np.array([[1., 2., 0., 0.], [2., 0., 3., 0.], [0., 3., 0., 0.], [0., 0., 0., 0.]])
    m = core(a)
    if sparse:
        m = m.to_csr()
    d = a.sum(axis=1)
    scale = np.zeros(4)
    scale[d > 0] = d[d > 0]**-.5
    s = scale[:, None] * a * scale[None, :]
    l = np.diag((d > 0).astype(float)) - s
    expected_p = np.divide(a, d[:, None], out=np.zeros_like(a), where=d[:, None] > 0)
    for output, expected in [(m.to_degree_matrix(), np.diag(d)), (m.to_laplacian(), np.diag(d)-a),
                             (m.normalized_adjacency(), s), (m.normalized_laplacian(), l),
                             (m.transition_matrix(), expected_p)]:
        assert output.is_sparse == sparse
        assert np.allclose(array(output), expected, atol=1e-13)
    for k in [0, 1, 2, 3, 5]:
        assert np.allclose(array(m.to_normalized_laplacian(eps=-.5, k=k)), np.linalg.matrix_power(s, k))


def test_graph_transform_errors_and_overflow_safe_transition_normalization():
    a = core([[np.finfo(float).max, np.finfo(float).max], [0., 0.]])
    assert np.array_equal(array(a.transition_matrix()), [[.5, .5], [0., 0.]])
    for value in [-1., np.nan, np.inf]:
        a = core([[value]])
        for op in [a.normalized_laplacian, a.transition_matrix]:
            with pytest.raises(RuntimeError):
                op()
    with pytest.raises(RuntimeError, match='detach'):
        core([[1.]]).requires_grad_(True).normalized_laplacian()


def test_graph_python_routes_preserve_native_node_identity_and_actual_operators():
    g = Graph()
    a, b, isolate = [g.add_node() for _ in range(3)]
    g.add_edge(a, b)
    l = g.laplacian_matrix(normalized=True)
    assert l.row_labels == l.columns
    i, j, k = [l.columns.index(f'node_{node}') for node in [a, b, isolate]]
    assert l.get(i, i) == 1 and l.get(i, j) == -1 and l.get(k, k) == 0
    p = g.transition_matrix()
    assert p.columns == l.columns
    assert p.get(i, j) == 1 and p.get(k, k) == 0
