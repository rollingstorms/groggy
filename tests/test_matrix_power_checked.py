import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    a = np.asarray(a, dtype=float)
    return GraphMatrix.from_data(a.T.tolist())


def array(a):
    return np.asarray(a.to_numpy()).reshape(a.shape)


@pytest.mark.parametrize('n', [0, 1, 2, 3, 8, 15])
def test_matrix_power_and_noncommuting_gradients_match_independent_references(n):
    a = np.array([[1.1, .3], [-.2, .9]])
    d = np.array([[.1, -.4], [.2, .3]])
    g = np.array([[.2, .7], [-.3, .1]])
    actual = core(a).power(n)
    assert np.allclose(array(actual), np.linalg.matrix_power(a, n), rtol=2e-13, atol=1e-14)
    tangent = array(core(a).power_jvp(n, core(d)))
    gradient = array(core(a).power_vjp(n, core(g)))
    h = 1e-5
    fd = (np.linalg.matrix_power(a+h*d, n)-np.linalg.matrix_power(a-h*d, n))/(2*h)
    assert np.allclose(tangent, fd, rtol=2e-7, atol=1e-9)
    assert np.sum(tangent*g) == pytest.approx(np.sum(d*gradient), rel=1e-12, abs=1e-12)
    tracked = core(a).requires_grad_(True)
    tracked.power(n).backward_with_gradient(core(g))
    assert np.allclose(array(tracked.grad), gradient, rtol=1e-13, atol=1e-14)


def test_sparse_powers_preserve_storage_and_large_exponents_have_bounded_depth():
    a = GraphMatrix.from_csr(10000, 10000, [1.] * 10000, list(range(10000)), list(range(10001)))
    for n in [0, 3, 2**32-1]:
        output = a.power(n)
        assert output.is_sparse and output.get(9999, 9999) == 1.
