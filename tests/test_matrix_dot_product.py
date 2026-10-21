from fractions import Fraction
import math
import random
from groggy import GraphMatrix


def product(a, b):
    left = GraphMatrix.from_data([[v] for v in a])
    right = GraphMatrix.from_data([b])
    return (left * right).get(0, 0)


def reference(a, b):
    total = sum((Fraction(x) * Fraction(y) for x, y in zip(a, b)), Fraction())
    try:
        return float(total)
    except OverflowError:
        return math.inf if total > 0 else -math.inf


def test_finite_matrix_dots_match_exact_rational_products_at_extreme_exponents():
    rng = random.Random(73)
    for _ in range(80):
        a = [math.ldexp(rng.uniform(-1, 1), rng.randint(-1000, 1000)) for _ in range(9)]
        b = [math.ldexp(rng.uniform(-1, 1), rng.randint(-1000, 1000)) for _ in range(9)]
        a += [a[0], 1.]
        b += [-b[0], math.ldexp(1., -1000)]
        assert product(a, b) == reference(a, b)
    largest = float.fromhex('0x1.fffffffffffffp1023')
    assert product([largest, largest, 1.], [largest, -largest, 1.]) == 1.
    tiny = float.fromhex('0x0.0000000000001p-1022')
    assert product([tiny, tiny], [.5, .5]) == tiny
