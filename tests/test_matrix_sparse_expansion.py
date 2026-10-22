import numpy as np
import pytest
from groggy import GraphMatrix


def diagonal(n):
    return GraphMatrix.from_csr(n, n, [2.] * n, list(range(n)), list(range(n + 1)))


@pytest.mark.parametrize('operation', [lambda a: a.exp(), lambda a: a.log(),
                                      lambda a: a + 1, lambda a: a * float('inf'),
                                      lambda a: a >= 0])
def test_large_sparse_full_grid_transforms_require_explicit_opt_in(operation):
    a = diagonal(10000)
    with pytest.raises(RuntimeError, match='to_dense'):
        operation(a)
    assert a.is_sparse and a.get(0, 0) == 2.


def test_small_ieee_results_and_explicit_dense_conversion_remain_correct():
    a = diagonal(2)
    assert a.exp().get(0, 1) == 1.
    with np.errstate(invalid='ignore'):
        expected = np.diag([2., 2.]) * np.inf
    result = np.asarray((a * float('inf')).to_numpy()).reshape(2, 2)
    assert np.array_equal(result, expected, equal_nan=True)
    n = 1025
    a = GraphMatrix.from_csr(n, n, [], [], [0] * (n + 1))
    assert (a.to_dense() + 1).get(n - 1, n - 1) == 1.


def test_large_mixed_nonfinite_product_does_not_materialize_sparse_input():
    a = diagonal(10000)
    column = GraphMatrix.from_data([[float('inf')] * 10000])
    with pytest.raises(RuntimeError, match='to_dense'):
        a.multiply(column)
    assert a.sqrt().is_sparse
