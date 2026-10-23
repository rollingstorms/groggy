"""Numerical outputs retain f64 precision and the representable working range."""
import sys

import pytest
from groggy import AttrValue, BaseArray, BaseTable


def wide(values):
    return BaseArray([AttrValue.float64(value) for value in values])


def test_wide_statistics_and_windows_keep_bits_beyond_f32():
    x, y = 1 + 2**-40, 1 + 3*2**-41
    array = wide([x, y])
    assert array.sum() == x+y
    assert array.quantile(0.5) == (x+y)/2
    assert list(array.cumsum())[-1] == x+y
    assert list(array.rolling(2, "mean")) == [None, (x+y)/2]
    table = BaseTable.from_dict({"x": [AttrValue.float64(x), AttrValue.float64(y)]})
    assert table.aggregate({"x": "mean"}).column("x").to_list() == [(x+y)/2]


def test_mean_divides_before_overflow_and_variance_and_std_have_distinct_ranges():
    array = wide([sys.float_info.max, sys.float_info.max])
    with pytest.raises((ValueError, RuntimeError)):
        array.sum()
    assert array.mean() == sys.float_info.max
    assert list(array.expanding("mean")) == [sys.float_info.max]*2
    array = wide([-1e154, 1e154])
    with pytest.raises((ValueError, RuntimeError)):
        array.var()
    assert array.std() == pytest.approx(2**0.5 * 1e154, rel=1e-14)
    assert array.cov(array, 0) == pytest.approx(1e308, rel=1e-14)
    assert list(array.expanding("std")) == pytest.approx([0, 1e154], rel=1e-14)
