from fractions import Fraction
import math
import random

import pytest
from groggy import GraphMatrix


def matrix(rows):
    return GraphMatrix.from_data([list(col) for col in zip(*rows)])


def exact_backward_error(a, b, x):
    a = [[Fraction(v) for v in row] for row in a]
    b = [[Fraction(v) for v in row] for row in b]
    x = [[Fraction(x.get(r, c)) for c in range(x.shape[1])] for r in range(x.shape[0])]
    residual = [[b[r][c] - sum(a[r][k] * x[k][c] for k in range(len(a)))
                 for c in range(len(b[0]))] for r in range(len(a))]
    norm = lambda values: max(sum(abs(v) for v in row) for row in values)
    denominator = norm(a) * norm(x) + norm(b)
    return float(norm(residual) / denominator) if denominator else 0.


@pytest.mark.parametrize('scale', [1e-250, 1., 1e250])
def test_refinement_reports_independent_residual_and_never_worsens(scale):
    rng = random.Random(129)
    improved = False
    for n in [2, 3, 5, 7]:
        a = [[scale * (rng.random() + (3 if r == c else 0)) for c in range(n)] for r in range(n)]
        b = [[scale * rng.random() for _ in range(2)] for r in range(n)]
        coefficients, rhs = matrix(a), matrix(b)
        plain, initial, count = coefficients.solve_refined(rhs, max_refinements=0)
        result, error, count = coefficients.solve_refined(rhs, max_refinements=4)
        assert error <= initial
        assert error == pytest.approx(exact_backward_error(a, b, result), rel=2e-12, abs=1e-30)
        assert initial == pytest.approx(exact_backward_error(a, b, plain), rel=2e-12, abs=1e-30)
        assert 0 <= count <= 4
        improved |= count > 0
    assert improved


def test_exact_empty_and_invalid_refinement_cases():
    result, error, count = matrix([[2., 0.], [0., 4.]]).solve_refined(matrix([[6.], [8.]]))
    assert [result.get(r, 0) for r in range(2)] == [3., 2.]
    assert (error, count) == (0., 0)
    result, error, count = GraphMatrix.zeros(0, 0).solve_refined(GraphMatrix.zeros(0, 3))
    assert result.shape == (0, 3)
    assert (error, count) == (0., 0)
    for a in [[[0.]], [[math.inf]], [[math.nan]], [[1., 2.]]]:
        with pytest.raises(RuntimeError):
            matrix(a).solve_refined(matrix([[1.]]))
    with pytest.raises(RuntimeError, match='16'):
        matrix([[1.]]).solve_refined(matrix([[1.]]), max_refinements=17)


def test_report_handles_overflowing_norm_denominator_without_zeroing_real_error():
    # ||A||inf * ||X||inf + ||B||inf exceeds f64, but the ratio is representable.
    a = [[1e308, 1e308], [1e308, -1e308]]
    b = [[1e308], [1.]]
    result, error, _ = matrix(a).solve_refined(matrix(b), max_refinements=0)
    expected = exact_backward_error(a, b, result)
    assert expected > 0
    assert error == pytest.approx(expected, rel=2e-12, abs=5e-324)
