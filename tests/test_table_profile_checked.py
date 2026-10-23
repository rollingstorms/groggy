"""Profiling keeps source identity and uses checked f64 statistics."""
import math

import pytest
from groggy import AttrValue, BaseTable


def cell(table, column):
    return table.column(column).to_list()[0]


def test_profile_retains_f64_precision_and_population_spread():
    x = 1 + 2**-40
    result = BaseTable.from_dict({"x": [AttrValue.float64(x)] * 2}).profile()
    assert cell(result, "dtype") == "float64"
    assert cell(result, "mean") == cell(result, "median") == x
    result = BaseTable.from_dict({"x": [2**63 - 3, 2**63 - 2, 2**63 - 1]}).profile()
    assert cell(result, "min") == 2**63 - 3
    assert cell(result, "max") == 2**63 - 1
    assert cell(result, "std") == pytest.approx(math.sqrt(2 / 3), rel=1e-15)


def test_profile_counts_text_once_and_rejects_nonfinite_observations():
    result = BaseTable.from_dict({"x": ["a", "a", None]}).profile()
    assert cell(result, "unique_count") == 2
    assert cell(result, "cardinality") == 2 / 3
    for value in [math.nan, math.inf, -math.inf]:
        with pytest.raises((ValueError, RuntimeError)):
            BaseTable.from_dict({"x": [AttrValue.float64(value)]}).profile()
