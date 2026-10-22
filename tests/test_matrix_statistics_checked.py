import math
import numpy as np
import pytest
from groggy import GraphMatrix


def matrix(a):
    return GraphMatrix.from_data(np.asarray(a, dtype=float).T.tolist())


@pytest.mark.parametrize('sparse', [False, True])
def test_entrywise_norms_propagate_nan_and_avoid_intermediate_overflow(sparse):
    a = matrix([[1e308, 1e308]])
    if sparse:
        a = a.to_csr()
    assert math.isclose(a.norm() / 1e308, math.sqrt(2), rel_tol=1e-14)
    assert a.norm_l1() == math.inf
    assert a.norm_inf() == 1e308
    assert a.mean() == 1e308
    a = matrix([[math.inf, math.nan]])
    if sparse:
        a = a.to_csr()
    assert all(math.isnan(f()) for f in [a.norm, a.norm_l1, a.norm_inf])


@pytest.mark.parametrize('shape', [(3, 5), (5, 3)])
def test_sparse_axis_statistics_include_implicit_zero_observations(shape):
    rng = np.random.default_rng(567)
    a = rng.normal(size=shape)
    a[rng.random(shape) < 0.6] = 0
    m = matrix(a).to_csr()
    for axis, numpy_axis in [(0, 1), (1, 0)]:
        for fn, reference in [('mean_axis', a.mean(axis=numpy_axis)),
                              ('std_axis', a.std(axis=numpy_axis, ddof=1)),
                              ('var_axis', a.var(axis=numpy_axis, ddof=1))]:
            actual = list(getattr(m, fn)(axis))
            assert np.allclose(actual, reference, rtol=2e-14, atol=1e-15)


def test_large_sparse_statistics_stay_sparse():
    n = 10000
    a = GraphMatrix.from_csr(n, n, [4.] * n, list(range(n)), list(range(n + 1)))
    assert math.isclose(a.norm(), 400., rel_tol=1e-14)
    assert math.isclose(a.mean(), 4 / n)
    assert np.allclose(list(a.std_axis(1)), 4 / math.sqrt(n))
    assert a.is_sparse
