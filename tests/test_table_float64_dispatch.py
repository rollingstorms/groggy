"""Whole-table numeric dispatch includes f64 and propagates checked failures."""
import math

import pytest
from groggy import AttrValue, BaseTable


def column(table, name="x"):
    return table.column(name).to_list()


def test_whole_table_dispatch_keeps_wide_columns_and_rank_precision():
    x = 1 + 2**-40
    source = BaseTable.from_dict({"x": [AttrValue.float64(v) for v in [x, 1, 3]]})
    assert "x" in source.describe().columns
    assert column(source.nlargest(2, "x")) == [3, x]
    assert column(source.nsmallest(2, "x")) == [1, x]
    assert column(source.rolling_all(2, "mean"), "x_rolling_2_mean")[1] == (x + 1) / 2
    assert column(source.expanding_all("sum"), "x_expanding_sum")[0] == x


def test_pairwise_statistics_use_checked_f64_values_and_validate_diagonals():
    source = BaseTable.from_dict({
        "x": [AttrValue.float64(v) for v in [1, 2, 3]],
        "y": [AttrValue.float64(v) for v in [2, 4, 6]],
    })
    assert column(source.cov(1)) == [1, 2]
    assert column(source.corr("pearson")) == pytest.approx([1, 1], abs=1e-15)
    for action in [lambda: source.corr("unsupported"), lambda: source.cov(-1)]:
        with pytest.raises((ValueError, RuntimeError)):
            action()
    invalid = BaseTable.from_dict({"x": [AttrValue.float64(math.nan)]})
    for action in [lambda: invalid.corr("pearson"), lambda: invalid.cov(0)]:
        with pytest.raises((ValueError, RuntimeError)):
            action()
