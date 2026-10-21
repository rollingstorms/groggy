from fractions import Fraction
import math
import random
import pytest
from groggy import GraphMatrix


def matrix(rows):
    return GraphMatrix.from_data([list(col) for col in zip(*rows)])


def exact_solve(a, b):
    n, m = len(a), len(b[0])
    work = [[Fraction(v) for v in a[i] + b[i]] for i in range(n)]
    for col in range(n):
        pivot = next(i for i in range(col, n) if work[i][col])
        work[col], work[pivot] = work[pivot], work[col]
        divisor = work[col][col]
        work[col] = [v / divisor for v in work[col]]
        for row in range(n):
            if row != col:
                factor = work[row][col]
                work[row] = [x - factor * y for x, y in zip(work[row], work[col])]
    return [[float(v) for v in row[n:n+m]] for row in work]


def test_direct_solves_match_independent_rational_elimination_and_scale_invariance():
    rng = random.Random(46)
    for n in [1, 2, 3, 5, 8]:
        a = [[rng.randint(-3, 3) + (n * 8 if r == c else 0) for c in range(n)] for r in range(n)]
        b = [[rng.randint(-9, 9) for _ in range(2)] for r in range(n)]
        expected = exact_solve(a, b)
        for scale in [1e-250, 1., 1e250]:
            result = matrix([[v*scale for v in row] for row in a]).solve(matrix([[v*scale for v in row] for row in b]))
            for r in range(n):
                for c in range(2):
                    assert result.get(r, c) == pytest.approx(expected[r][c], rel=1e-13, abs=1e-14)


def test_inverse_does_not_use_an_absolute_determinant_threshold_and_invalid_domains_error():
    for scale in [1e-300, 1., 1e300]:
        inverse = matrix([[scale]]).inverse()
        assert inverse.get(0, 0) == pytest.approx(1 / scale, rel=1e-14)
    for data in [[[0.]], [[1., 2.], [2., 4.]], [[math.nan]], [[math.inf]]]:
        with pytest.raises(RuntimeError):
            matrix(data).inverse()
    assert GraphMatrix.zeros(0, 0).inverse().shape == (0, 0)
