import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    a = np.asarray(a, dtype=float)
    return GraphMatrix.zeros(*a.shape) if not a.size else GraphMatrix.from_data(a.T.tolist())


def array(a):
    return np.asarray(a.to_numpy()).reshape(a.shape)


@pytest.mark.parametrize('shape', [(7, 3), (3, 7), (4, 4)])
@pytest.mark.parametrize('scale', [1e-250, 1., 1e250])
def test_solutions_match_numpy_and_normal_equations(shape, scale):
    rng = np.random.default_rng(754)
    a, b = rng.normal(size=shape), rng.normal(size=(shape[0], 2))
    rcond = np.finfo(float).eps * max(shape)
    expected = np.linalg.lstsq(a, b, rcond=rcond)[0]
    result = array(core(a * scale).least_squares(core(b * scale)))
    assert np.allclose(result, expected, rtol=1e-11, atol=1e-12)
    assert np.allclose(a.T @ (a @ result - b), 0, atol=1e-11)
    if shape[1] > shape[0]:
        _, _, vt = np.linalg.svd(a)
        assert np.allclose(vt[shape[0]:] @ result, 0, atol=1e-11)


@pytest.mark.parametrize('shape', [(6, 3), (3, 6), (4, 4)])
def test_rank_deficient_minimum_norm_and_explicit_cutoff(shape):
    rng = np.random.default_rng(10)
    a = rng.normal(size=shape)
    a[:, -1] = 2 * a[:, 0]
    if shape[1] > shape[0]:
        a[-1] = a[0]
    b = rng.normal(size=(shape[0], 2))
    result = array(core(a).least_squares(core(b)))
    expected = np.linalg.lstsq(a, b, rcond=np.finfo(float).eps * max(shape))[0]
    assert np.allclose(result, expected, rtol=1e-10, atol=1e-11)
    diagonal = core(np.diag([1., 1e-20]))
    rhs = core([[2.], [1e-20]])
    assert np.allclose(array(diagonal.least_squares(rhs)), [[2.], [0.]])
    assert np.allclose(array(diagonal.least_squares(rhs, rcond=0.)), [[2.], [1.]])
    assert np.allclose(array(diagonal.least_squares(rhs, rcond=1.)), 0.)


@pytest.mark.parametrize('shape,rhs', [((0, 3), 2), ((3, 0), 2), ((3, 3), 0), ((0, 0), 0), ((4, 3), 2)])
def test_zero_and_empty_minimum_norm_solutions(shape, rhs):
    result = core(np.zeros(shape)).least_squares(core(np.ones((shape[0], rhs))))
    assert result.shape == (shape[1], rhs)
    assert np.all(array(result) == 0)


def test_invalid_domains_cutoffs_and_normalization_range_fail_explicitly():
    for value in [np.nan, np.inf]:
        with pytest.raises(RuntimeError):
            core([[value]]).least_squares(core([[1.]]))
        with pytest.raises(RuntimeError):
            core([[1.]]).least_squares(core([[value]]))
    for cutoff in [-1., np.nan, np.inf]:
        with pytest.raises(RuntimeError, match='tolerance'):
            core([[1.]]).least_squares(core([[1.]]), rcond=cutoff)
    with pytest.raises(RuntimeError, match='matching'):
        core(np.eye(2)).least_squares(core([[1.]]))
    with pytest.raises(RuntimeError, match='range'):
        core([[1e-300]]).least_squares(core([[1e300]]))
