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
def test_lu_and_log_determinant_match_reconstruction_and_reference(scale):
    rng = np.random.default_rng(19)
    a = rng.normal(size=(8, 8)) * scale
    p, lower, upper = core(a).lu_decomposition()
    p, lower, upper = array(p), array(lower), array(upper)
    assert np.allclose((p @ a) / scale, (lower @ upper) / scale, atol=2e-13)
    assert np.array_equal(lower, np.tril(lower))
    assert np.array_equal(upper, np.triu(upper))
    assert np.array_equal(np.diag(lower), np.ones(8))
    sign, log = core(a).slogdet()
    expected = np.linalg.slogdet(a)
    assert sign == expected[0]
    assert log == pytest.approx(expected[1], rel=1e-13, abs=1e-12)


def test_determinants_handle_singular_empty_extreme_and_invalid_inputs():
    a = core(np.diag([1e300, 1e-300, 2.]))
    assert a.determinant() == pytest.approx(2., rel=1e-12)
    assert core(np.zeros((3, 3))).slogdet() == (0., -np.inf)
    assert GraphMatrix.zeros(0, 0).determinant() == 1.
    for value in [np.nan, np.inf]:
        with pytest.raises(RuntimeError):
            core(np.array([[value]])).slogdet()
