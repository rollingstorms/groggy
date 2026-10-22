import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    a = np.asarray(a, dtype=float)
    return GraphMatrix.zeros(*a.shape) if not a.size else GraphMatrix.from_data(a.T.tolist())


def array(a):
    return np.asarray(a.to_numpy()).reshape(a.shape)


@pytest.mark.parametrize('shape', [(7, 3), (3, 7), (4, 4), (0, 3), (3, 0)])
@pytest.mark.parametrize('zero', [False, True])
def test_economy_svd_shapes_reconstruction_orthogonality_and_independent_spectra(shape, zero):
    rng = np.random.default_rng(67 + sum(shape))
    a = np.zeros(shape) if zero else rng.normal(size=shape)
    u, s, vt = core(a).svd_thin()
    k = min(shape)
    assert u.shape == (shape[0], k) and vt.shape == (k, shape[1])
    u, vt = array(u), array(vt)
    assert np.allclose(u @ np.diag(s) @ vt, a, atol=1e-12)
    assert np.allclose(u.T @ u, np.eye(k), atol=1e-12)
    assert np.allclose(vt @ vt.T, np.eye(k), atol=1e-12)
    assert np.allclose(s, np.linalg.svd(a, compute_uv=False), atol=1e-12)


@pytest.mark.parametrize('shape', [(10000, 2), (2, 10000)])
def test_large_rectangular_spectrum_and_least_squares_use_thin_workspaces(shape):
    rng = np.random.default_rng(451)
    a = rng.normal(size=shape)
    b = rng.normal(size=(shape[0], 2))
    m = core(a)
    u, s, vt = m.svd_thin()
    assert u.shape == (shape[0], 2) and vt.shape == (2, shape[1])
    assert m.rank() == 2
    assert m.condition_number() == pytest.approx(np.linalg.cond(a), rel=1e-10)
    expected = np.linalg.lstsq(a, b, rcond=None)[0]
    assert np.allclose(array(m.least_squares(core(b))), expected, rtol=1e-9, atol=1e-11)
