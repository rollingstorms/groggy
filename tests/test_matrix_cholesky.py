import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    if not a.size:
        return GraphMatrix.zeros(*a.shape)
    return GraphMatrix.from_data(a.T.tolist())


def array(a):
    return np.array([[a.get(r, c) for c in range(a.shape[1])] for r in range(a.shape[0])]).reshape(a.shape)


@pytest.mark.parametrize('scale', [1e-250, 1., 1e250])
def test_cholesky_matches_reference_and_reconstructs_across_scales(scale):
    rng = np.random.default_rng(25)
    seed = rng.normal(size=(7, 7))
    a = (seed @ seed.T + np.eye(7)) * scale
    lower = array(core(a).cholesky_decomposition())
    assert np.allclose(lower / np.sqrt(scale), np.linalg.cholesky(a / scale), atol=2e-13)
    assert np.allclose((lower @ lower.T) / scale, a / scale, atol=3e-13)
    assert np.array_equal(lower, np.tril(lower))


def test_cholesky_rejects_invalid_domains_and_handles_zero_order():
    for a in [np.array([[1., 2.], [0., 1.]]), np.array([[1., 2.], [2., 1.]]),
              np.zeros((2, 2)), np.array([[np.nan]]), np.array([[np.inf]])]:
        with pytest.raises(RuntimeError):
            core(a).cholesky_decomposition()
    assert GraphMatrix.zeros(0, 0).cholesky_decomposition().shape == (0, 0)
