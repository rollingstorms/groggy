"""Attribute statistics retain integer residuals through final f64 storage."""
import math
from fractions import Fraction

import pytest
from groggy import AttrValue, BaseArray


def test_large_and_mixed_integer_pair_statistics():
    maximum = 2**63 - 1
    right = BaseArray([1, 2, 3])
    for values in ([maximum - 2, maximum - 1, maximum],
                   [maximum - 1, maximum, float(2**63)]):
        left = BaseArray(values)
        assert left.cov(right) == 1.0
        for method in ("pearson", "spearman", "kendall"):
            assert left.corr(right, method) == pytest.approx(1.0, abs=1e-14)


def test_quantile_cancellation_and_empty_validation():
    values = [-(2**63), 2**63 - 1]
    expected = float(sum(map(Fraction, values)) / 2)
    for method in ("linear", "midpoint"):
        assert BaseArray(values).quantile(0.5, method) == expected
        assert BaseArray([2**63 - 1, -float(2**63)]).quantile(0.5, method) == -0.5
    assert BaseArray([]).quantile(0.5) is None
    for q, method in ((math.nan, "linear"), (0.5, "invalid")):
        with pytest.raises((ValueError, RuntimeError)):
            BaseArray([]).quantile(q, method)


def test_paired_missing_and_invalid_domains():
    left, right = BaseArray([1, None, 3]), BaseArray([2, 99, 6])
    assert left.cov(right) == 4.0
    assert left.corr(right) == pytest.approx(1.0, abs=1e-14)
    for action in (lambda: left.cov(right, -1),
                   lambda: left.corr(BaseArray([1])),
                   lambda: BaseArray([math.nan, 1]).corr(BaseArray([1, 2]), "spearman"),
                   lambda: BaseArray([AttrValue.float64(1e300), AttrValue.float64(-1e300)]).cov(BaseArray([AttrValue.float64(1e300), AttrValue.float64(-1e300)]))):
        with pytest.raises((ValueError, RuntimeError)):
            action()


def test_ties_keep_average_ranks_and_legacy_kendall_tau_a():
    left, right = BaseArray([1, 1, 2]), BaseArray([1, 2, 3])
    assert left.corr(right, "spearman") == pytest.approx(math.sqrt(3) / 2, rel=1e-7)
    assert left.corr(right, "kendall") == pytest.approx(2 / 3, rel=1e-7)
