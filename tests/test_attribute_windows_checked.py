"""Windows use stored f32 outputs and exact high-offset integer differences."""
import math
import statistics
from fractions import Fraction

import pytest
from groggy import BaseArray


def values(array):
    return list(array)


def test_expanding_and_rolling_population_std_preserve_integer_offsets():
    data = [2**63 - 3, 2**63 - 2, 2**63 - 1]
    expected = [math.sqrt(float(statistics.pvariance(list(map(Fraction, data[:end])))))
                for end in range(1, 4)]
    assert values(BaseArray(data).expanding("std")) == pytest.approx(expected, rel=1e-7)
    assert values(BaseArray(data).rolling(2, "std")) == [None, 0.5, 0.5]
    assert values(BaseArray([2**63 - 1, float(2**63)]).expanding("std")) == [0.0, 0.5]


def test_cumulative_and_expanding_keep_mixed_cancellation():
    data = [None, 2**63 - 1, -float(2**63), 3, None]
    result = values(BaseArray(data).cumsum())
    assert result[0] == 0.0
    assert result[2:] == [-1.0, 2.0, 2.0]
    result = values(BaseArray(data).expanding("mean"))
    assert result[0] is None
    assert result[2] == -0.5
    assert values(BaseArray([None, 3, 1, None]).cummin()) == [None, 3.0, 1.0, 1.0]


@pytest.mark.parametrize("period", [-(2**31), -10, 10, 2**31-1])
def test_shift_beyond_length_returns_fill(period):
    assert values(BaseArray([1, 2]).shift(period)) == [None, None]
    assert values(BaseArray([]).shift(period)) == []


def test_fractional_change_retains_integer_difference():
    result = values(BaseArray([2**63 - 2, 2**63 - 1]).pct_change())
    assert result[0] is None
    assert result[1] == pytest.approx(1 / (2**63 - 2), rel=1e-7)
    assert values(BaseArray([2, 3, None, 0]).pct_change()) == [None, 0.5, None, None]


def test_windows_reject_bad_domains_and_overflow():
    for action in (lambda: BaseArray([]).expanding("bad"),
                   lambda: BaseArray([]).rolling(0, "sum"),
                   lambda: BaseArray([math.nan]).rolling(2, "mean"),
                   lambda: BaseArray([3e38, 3e38]).cumsum()):
        with pytest.raises((ValueError, RuntimeError)):
            action()
