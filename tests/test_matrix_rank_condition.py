import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    if not a.size:
        return GraphMatrix.zeros(*a.shape)
    return GraphMatrix.from_data(a.T.tolist())


@pytest.mark.parametrize('shape', [(7, 4), (4, 7), (5, 5)])
@pytest.mark.parametrize('scale', [1e-250, 1., 1e250])
def test_rank_and_condition_match_independent_spectrum_across_shapes_and_scale(shape, scale):
    rng = np.random.default_rng(81)
    a = rng.normal(size=shape)
    expected = np.linalg.svd(a, compute_uv=False)
    matrix = core(a * scale)
    assert matrix.rank() == min(shape)
    assert matrix.condition_number() == pytest.approx(expected[0] / expected[-1], rel=2e-12)


def test_trailing_columns_explicit_cutoffs_and_invalid_domains():
    assert core(np.array([[0., 1., 0.], [0., 0., 1.]])).rank() == 2
    a = core(np.diag([1., 1e-8]))
    assert a.rank_with_tolerance(1e-7) == 1
    assert a.rank_with_tolerance(1e-9) == 2
    assert GraphMatrix.zeros(3, 3).condition_number() == np.inf
    for cutoff in [-1., np.nan, np.inf]:
        with pytest.raises(ValueError):
            a.rank_with_tolerance(cutoff)
    with pytest.raises(RuntimeError):
        GraphMatrix.zeros(0, 3).condition_number()
