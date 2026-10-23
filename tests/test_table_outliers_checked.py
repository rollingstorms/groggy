"""IQR classification keeps numeric precision and validates before sorting."""
import math

import pytest
from groggy import AttrValue, BaseTable


def flags(values, factor=None):
    return BaseTable.from_dict({"x": values}).check_outliers("x", factor).to_list()


def test_outliers_preserve_high_integers_and_sub_f32_values():
    m = 2**63 - 1
    assert flags([m - 100, m - 4, m - 3, m - 2, m - 1, m, m, m]) == [True] + [False] * 7
    assert flags([AttrValue.float64(1 + i * 2**-40) for i in [0, 1, 2, 3, 4, 5, 6, 100]]) == [False] * 7 + [True]
    assert flags([1, None, "skip", True]) == [False] * 4


def test_outliers_handle_wide_fences_and_reject_nonfinite_inputs():
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    assert flags([AttrValue.float64(maximum)] * 4, maximum) == [False] * 4
    assert flags([AttrValue.float64(-maximum), -1, 0, 1, AttrValue.float64(maximum)]) == [True, False, False, False, True]
    for factor in [-1, math.nan, math.inf]:
        with pytest.raises((ValueError, RuntimeError)):
            flags([1], factor)
    for value in [math.nan, math.inf]:
        with pytest.raises((ValueError, RuntimeError)):
            flags([AttrValue.float64(value)])
