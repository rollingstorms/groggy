import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    a = np.asarray(a, dtype=float)
    return GraphMatrix.zeros(*a.shape) if not a.size else GraphMatrix.from_data(a.T.tolist())


def array(a):
    return np.asarray(a.to_numpy()).reshape(a.shape)


@pytest.mark.parametrize('shape', [(3, 2), (2, 3), (3, 3)])
def test_tracked_singular_values_match_independent_weighted_loss_derivative(shape):
    rng = np.random.default_rng(763 + sum(shape))
    a = rng.normal(size=shape)
    direction = rng.normal(size=shape)
    tracked = core(a).requires_grad_(True)
    u, s, vt = tracked.svd_tracked()
    k = min(shape)
    assert (u.shape, s.shape, vt.shape) == ((shape[0], k), (k, 1), (k, shape[1]))
    weights = np.arange(1., k + 1)
    s.backward_with_gradient(core(weights[:, None]))
    gradient = array(tracked.grad)
    h = 1e-5
    expected = (weights @ np.linalg.svd(a+h*direction, compute_uv=False)
                - weights @ np.linalg.svd(a-h*direction, compute_uv=False)) / (2*h)
    assert np.sum(gradient*direction) == pytest.approx(expected, rel=1e-7, abs=1e-9)
    s.backward_with_gradient(core(weights[:, None]))
    assert np.array_equal(array(tracked.grad), gradient)


def test_tracked_eigenvalues_and_vectors_match_explicit_and_independent_derivatives():
    a = np.array([[4., 1.], [1., 2.]])
    direction = np.array([[.2, -.3], [-.3, .7]])
    tracked = core(a).requires_grad_(True)
    values, vectors = tracked.eigen_tracked()
    weights = np.array([2., -1.])
    values.backward_with_gradient(core(weights[:, None]))
    h = 1e-5
    fd = (weights @ np.linalg.eigvalsh(a+h*direction)[::-1]
          - weights @ np.linalg.eigvalsh(a-h*direction)[::-1]) / (2*h)
    assert np.sum(array(tracked.grad)*direction) == pytest.approx(fd, rel=1e-8)
    seed = core([[.3, -.2], [.1, .4]])
    vectors.backward_with_gradient(seed)
    assert np.allclose(array(tracked.grad), array(core(a).eigen_vjp([0., 0.], seed)))


@pytest.mark.parametrize('factorization', ['qr_decomposition', 'lu_decomposition'])
def test_factor_outputs_preserve_saved_inputs_and_accept_seeded_backward(factorization):
    a = np.array([[4., 1.], [2., 3.]])
    tracked = core(a).requires_grad_(True)
    outputs = getattr(tracked, factorization)()
    outputs = outputs if factorization == 'qr_decomposition' else outputs[1:]
    for i, output in enumerate(outputs):
        seed = core([[.1, .3], [-.2, .4]])
        zeros = core(np.zeros((2, 2)))
        output.backward_with_gradient(seed)
        directions = (seed, zeros) if i == 0 else (zeros, seed)
        fn = core(a).qr_vjp if factorization == 'qr_decomposition' else core(a).lu_vjp
        assert np.allclose(array(tracked.grad), array(fn(*directions)))
    tracked.set(0, 0, 99.)
    outputs[0].backward()
    assert tracked.grad is not None
    with pytest.raises(RuntimeError, match='changed since gradient binding'):
        getattr(tracked, factorization)()
    with pytest.raises(RuntimeError):
        outputs[0].backward_with_gradient(core([[1.]]))


def test_derivative_domains_are_rejected_explicitly():
    a = core(np.eye(2)).requires_grad_(True)
    with pytest.raises(RuntimeError):
        a.svd_tracked()[1].backward()
    with pytest.raises(RuntimeError):
        a.eigen_tracked()[0].backward()
    with pytest.raises(RuntimeError, match='thin'):
        core([[1., 2.]]).requires_grad_(True).svd()
