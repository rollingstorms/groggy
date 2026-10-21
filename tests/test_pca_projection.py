import numpy as np
import pytest
from groggy import GraphMatrix


def core(data):
    data = np.asarray(data, dtype=float)
    if 0 in data.shape:
        return GraphMatrix.zeros(*data.shape)
    return GraphMatrix.from_data(data.T.tolist())


def reference(data, center, standardize):
    mean = data.mean(axis=0)
    deviation = data.std(axis=0, ddof=0)
    deviation[deviation == 0] = 1.0
    prepared = data - mean if center else data.copy()
    if standardize:
        prepared = prepared / deviation
    _, _, vt = np.linalg.svd(prepared, full_matrices=True)
    return prepared @ vt[:2].T


@pytest.mark.parametrize("shape", [(1, 3), (2, 6), (7, 3), (8, 5), (5, 12)])
@pytest.mark.parametrize("center,standardize", [(False, False), (True, False), (False, True), (True, True)])
def test_projection_matches_numpy_svd_geometry(shape, center, standardize):
    data = np.random.default_rng(914).normal(size=shape)
    data[:, -1] = 3.0  # Constant columns use a scale of one.
    matrix = core(data)
    result = matrix.pca_projection(center=center, standardize=standardize)
    got = np.asarray(result.to_numpy()).reshape(shape[0], 2)
    want = reference(data, center, standardize)
    # Signs/bases in tied eigenspaces need not match.
    assert np.allclose(got @ got.T, want @ want.T, rtol=1e-9, atol=1e-10)
    assert np.array_equal(np.asarray(matrix.to_numpy()).reshape(shape), data)
    assert result.row_labels is None


@pytest.mark.parametrize("scale", [1e-200, 1e200])
@pytest.mark.parametrize("standardize", [False, True])
def test_extreme_uniform_scales_do_not_overflow_or_erase_geometry(scale, standardize):
    base = np.array([[3., 3., 0.], [-3., -3., 0.], [1., -1., 0.], [-1., 1., 0.]])
    got = np.asarray(core(base * scale).pca_projection(standardize=standardize).to_numpy()).reshape(4, 2)
    if not standardize:
        got /= scale
    want = reference(base, True, standardize)
    assert np.allclose(got @ got.T, want @ want.T, atol=1e-10)


def test_repeated_spectrum_and_constant_data_return_finite_scores():
    got = np.asarray(core(np.eye(3)).pca_projection(center=False).to_numpy()).reshape(3, 2)
    assert np.isclose(np.sum(got * got), 2.0)
    assert np.allclose(got.T @ got, np.eye(2), atol=1e-10)
    zeros = core(np.ones((3, 4))).pca_projection(standardize=True)
    assert np.array_equal(np.asarray(zeros.to_numpy()).reshape(3, 2), np.zeros((3, 2)))


@pytest.mark.parametrize("data", [np.empty((0, 2)), np.ones((2, 1)),
                                  np.array([[np.nan, 1.]]), np.array([[np.inf, 1.]])])
def test_invalid_input_rejects(data):
    with pytest.raises(ValueError):
        core(data).pca_projection()
