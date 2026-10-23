"""Table numeric adapters share canonical column arithmetic and exact ordering."""
import math

import pytest
from groggy import AttrValue, BaseTable


def column(table, name="x"):
    return table.column(name).to_list()


def test_aggregate_cancellation_compact_values_and_checked_output():
    maximum = 2**63 - 1
    table = BaseTable.from_dict({"x": [maximum, 1, -maximum, None, "skip"]})
    assert column(table.aggregate({"x": "sum"})) == [1]
    assert column(table.aggregate({"x": "count"})) == [5]
    assert column(table.aggregate({"x": "mean"})) == pytest.approx([1 / 3], rel=1e-7)
    for values in ([maximum, 1], [AttrValue.float64(1e308), AttrValue.float64(1e308)], [math.nan]):
        with pytest.raises((ValueError, RuntimeError)):
            BaseTable.from_dict({"x": values}).aggregate({"x": "sum"})


def test_exact_numeric_sort_and_independent_text_pivot_keys():
    table = BaseTable.from_dict({"x": [float(2**63), 2**63 - 1, 1]})
    assert column(table.sort_values("x")) == [1, 2**63 - 1, float(2**63)]
    table = BaseTable.from_dict({"g": [1, 1], "key": ["a", "b"], "x": [2, 7]})
    pivot = table.pivot_table(["g"], "key", "x", "sum")
    assert column(pivot, "key_a") == [2.0]
    assert column(pivot, "key_b") == [7.0]


def test_pivot_distinct_numeric_keys_and_ambiguous_column_names():
    table = BaseTable.from_dict({"g": [1, 1], "key": [2**63 - 1, float(2**63)], "x": [2, 7]})
    pivot = table.pivot_table(["g"], "key", "x", "sum")
    assert sorted(column(pivot, name)[0] for name in pivot.columns if name.startswith("key_")) == [2.0, 7.0]
    table = BaseTable.from_dict({"g": [1, 1], "key": [1, 1.0], "x": [2, 7]})
    with pytest.raises((ValueError, RuntimeError)):
        table.pivot_table(["g"], "key", "x", "sum")


def test_describe_preserves_population_spread_and_midpoint_cancellation():
    table = BaseTable.from_dict({"x": [2**63 - 3, 2**63 - 2, 2**63 - 1]})
    assert column(table.describe())[2] == pytest.approx(math.sqrt(2 / 3), rel=1e-7)
    table = BaseTable.from_dict({"x": [2**63 - 1, -float(2**63)]})
    result = column(table.describe())
    assert result[0] == 2.0
    assert result[1] == result[5] == -0.5
