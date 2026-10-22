import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    if 0 in a.shape:
        return GraphMatrix.zeros(*a.shape)
    return GraphMatrix.from_data(a.T.tolist())


def array(a):
    return np.array([[a.get(r, c) for c in range(a.shape[1])] for r in range(a.shape[0])]).reshape(a.shape)


@pytest.mark.parametrize('shape', [(8, 3), (3, 8), (5, 5), (0, 3), (3, 0)])
@pytest.mark.parametrize('scale', [1e-250, 1., 1e250])
def test_economy_qr_reconstruction_orthogonality_and_triangularity(shape, scale):
    rng = np.random.default_rng(9)
    a = rng.normal(size=shape) * scale
    q, r = core(a).qr_decomposition()
    k = min(shape)
    assert q.shape == (shape[0], k)
    assert r.shape == (k, shape[1])
    q, r = array(q), array(r)
    assert np.allclose(q.T @ q, np.eye(k), atol=2e-13)
    assert np.allclose((q @ r) / scale, a / scale, atol=2e-13)
    assert np.allclose(r, np.triu(r), atol=0.)


def test_rank_deficiency_retains_orthonormal_columns_and_invalid_values_error():
    for a in [np.zeros((5, 3)), np.outer(np.arange(1., 6.), [1., 2., 3.])]:
        q, r = core(a).qr_decomposition()
        assert np.allclose(array(q).T @ array(q), np.eye(3), atol=2e-13)
        assert np.allclose(array(q) @ array(r), a, atol=2e-13)
    for bad in [np.nan, np.inf]:
        with pytest.raises(RuntimeError):
            core(np.array([[bad]])).qr_decomposition()
