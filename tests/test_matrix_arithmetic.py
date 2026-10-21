import math
import pytest
from groggy import GraphMatrix


def matrix(rows):
    return GraphMatrix.from_data([list(col) for col in zip(*rows)])


def values(a):
    return [[a.get(r, c) for c in range(a.shape[1])] for r in range(a.shape[0])]


def test_matrix_operators_are_checked_and_preserve_inputs():
    a = matrix([[1., 2., 3.], [4., 5., 6.]])
    b = matrix([[6., 5., 4.], [3., 2., 1.]])
    assert values(a + b) == [[7.] * 3, [7.] * 3]
    assert values(a - b) == [[-5., -3., -1.], [1., 3., 5.]]
    assert values(a + 2) == [[3., 4., 5.], [6., 7., 8.]]
    assert values(a - 2) == [[-1., 0., 1.], [2., 3., 4.]]
    assert values(-a) == [[-1., -2., -3.], [-4., -5., -6.]]
    assert values(a * 2) == [[2., 4., 6.], [8., 10., 12.]]
    assert values(a) == [[1., 2., 3.], [4., 5., 6.]]
    bad = matrix([[1., 2.], [3., 4.], [5., 6.]])
    with pytest.raises(RuntimeError, match="dimensions must match"):
        a + bad
    with pytest.raises(RuntimeError, match="dimensions must match"):
        a - bad


def test_direct_division_accepts_small_nonzero_values_and_explicit_ieee_behavior():
    tiny = float.fromhex('0x0.0000000000001p-1022')
    a = matrix([[0., tiny, -tiny]])
    assert values(a / tiny) == [[0., 1., -1.]]
    for zero in [0., -0.]:
        with pytest.raises(RuntimeError, match="Division by zero"):
            a / zero
    assert math.isnan((a / math.nan).get(0, 0))
    assert (a / math.inf).get(0, 1) == 0.


@pytest.mark.parametrize('shape', [(0, 0), (3, 0), (0, 3)])
def test_scalar_operations_preserve_empty_shapes(shape):
    a = GraphMatrix.zeros(*shape)
    for b in [a + 2., a - 2., a * 2., a / 2., -a]:
        assert b.shape == shape


def test_matrix_multiplication_reports_incompatible_shapes_and_keeps_empty_inner_products():
    a = matrix([[1., 2., 3.], [4., 5., 6.]])
    b = matrix([[7., 8.], [9., 10.], [11., 12.]])
    assert values(a * b) == [[58., 64.], [139., 154.]]
    with pytest.raises(RuntimeError, match='incompatible'):
        a * GraphMatrix.identity(2)
    product = GraphMatrix.zeros(3, 0) * GraphMatrix.zeros(0, 2)
    assert product.shape == (3, 2)
    assert values(product) == [[0., 0.]] * 3
