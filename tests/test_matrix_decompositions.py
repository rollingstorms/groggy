import numpy as np
import pytest

from groggy import GraphMatrix


def core(data):
    data = np.asarray(data, dtype=float)
    if 0 in data.shape:
        return GraphMatrix.zeros(*data.shape)
    return GraphMatrix.from_data(data.T.tolist())  # existing column-oriented factory


def array(matrix):
    return np.asarray(matrix.to_numpy()).reshape(matrix.shape)


@pytest.mark.parametrize("shape", [(1, 1), (4, 2), (2, 4), (5, 5), (8, 3), (3, 8), (0, 3), (3, 0), (0, 0)])
def test_full_svd_matches_reference_and_reconstructs_each_shape(shape):
    rng = np.random.default_rng(310)
    a = rng.normal(size=shape)
    u, singular, vt = core(a).svd()
    u, vt = array(u), array(vt)
    k = min(shape)
    assert u.shape == (shape[0], shape[0])
    assert vt.shape == (shape[1], shape[1])
    assert len(singular) == k
    assert np.allclose(singular, np.linalg.svd(a, compute_uv=False), rtol=1e-10, atol=1e-12)
    assert np.allclose(u.T @ u, np.eye(shape[0]), atol=1e-10)
    assert np.allclose(vt @ vt.T, np.eye(shape[1]), atol=1e-10)
    assert np.allclose((u[:, :k] * singular) @ vt[:k, :], a, atol=1e-10)


@pytest.mark.parametrize("a", [np.zeros((4, 3)), np.array([[1., 2.], [2., 4.]]),
                               np.diag([4., 4., 0.]), np.diag([1., 1e-12])])
def test_rank_deficient_repeated_and_small_singular_values(a):
    u, s, vt = core(a).svd()
    u, vt = array(u), array(vt)
    assert np.allclose(s, np.linalg.svd(a, compute_uv=False), rtol=1e-10, atol=1e-15)
    assert np.allclose(u.T @ u, np.eye(a.shape[0]), atol=1e-10)
    assert np.allclose(vt @ vt.T, np.eye(a.shape[1]), atol=1e-10)
    assert np.allclose((u[:, :len(s)] * s) @ vt[:len(s), :], a, atol=1e-10)


@pytest.mark.parametrize("scale", [1., 1e-200, 1e200])
def test_symmetric_eigenpairs_match_reference_under_rescaling(scale):
    a = np.array([[2., 1., -1.], [1., 3., 2.], [-1., 2., -4.]]) * scale
    values, vectors = core(a).eigenvalue_decomposition()
    vectors = array(vectors)
    assert np.allclose(np.array(values) / scale, np.linalg.eigvalsh(a / scale)[::-1], atol=1e-11)
    assert np.allclose(vectors.T @ vectors, np.eye(3), atol=1e-11)
    assert np.allclose((a / scale) @ vectors, vectors * (np.array(values) / scale), atol=1e-11)


def test_zero_and_repeated_eigenvalues_keep_complete_orthogonal_basis():
    for a in [np.zeros((4, 4)), np.eye(3) * 7, np.zeros((0, 0)), np.array([[2., 1.], [1., 2.]])]:
        values, v = core(a).eigenvalue_decomposition()
        v = array(v)
        assert len(values) == len(a)
        assert np.allclose(values, np.linalg.eigvalsh(a)[::-1])
        assert np.allclose(v.T @ v, np.eye(len(a)), atol=1e-12)
        assert np.allclose(a @ v, v * values, atol=1e-12)


def test_invalid_domains_raise_instead_of_returning_approximate_placeholders():
    for a in [np.array([[np.nan]]), np.array([[np.inf]])]:
        with pytest.raises(RuntimeError):
            core(a).svd()
        with pytest.raises(RuntimeError):
            core(a).eigenvalue_decomposition()
    with pytest.raises(RuntimeError, match="symmetric"):
        core([[1., 2.], [3., 4.]]).eigenvalue_decomposition()
    with pytest.raises(RuntimeError, match="square"):
        core([[1., 2., 3.]]).eigenvalue_decomposition()


def test_larger_random_and_ill_conditioned_fixtures():
    rng = np.random.default_rng(681)
    for shape in [(32, 16), (16, 32)]:
        a = rng.normal(size=shape)
        u, s, vt = core(a).svd()
        u, vt = array(u), array(vt)
        assert np.allclose(s, np.linalg.svd(a, compute_uv=False), rtol=1e-10)
        assert np.allclose((u[:, :len(s)] * s) @ vt[:len(s), :], a, atol=1e-10)
        assert np.allclose(u.T @ u, np.eye(shape[0]), atol=1e-10)
        assert np.allclose(vt @ vt.T, np.eye(shape[1]), atol=1e-10)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    z, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    spectrum = np.geomspace(1., 1e-12, 8)
    a = q @ np.diag(spectrum) @ z.T
    u, s, vt = core(a).svd()
    assert np.allclose(s, np.linalg.svd(a, compute_uv=False), rtol=1e-4, atol=1e-15)
    assert np.allclose(array(u) @ np.diag(s) @ array(vt), a, atol=1e-11)
    symmetric = rng.normal(size=(16, 16))
    symmetric = (symmetric + symmetric.T) / 2
    values, v = core(symmetric).eigenvalue_decomposition()
    assert np.allclose(values, np.linalg.eigvalsh(symmetric)[::-1], atol=1e-10)
    assert np.allclose(symmetric @ array(v), array(v) * values, atol=1e-10)
