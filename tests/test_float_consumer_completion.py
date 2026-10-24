"""Consumer compatibility before ordinary property inference switches to f64."""
import math

import pytest
from groggy import AttrValue, BaseArray, BaseTable


def test_base_array_operators_sort_and_metadata_keep_wide_values():
    a = BaseArray([AttrValue.float64(1), AttrValue.float64(1 + 2**-40)])
    assert (a == AttrValue.float64(1)).to_list() == [True, False]
    assert (a != AttrValue.float64(1)).to_list() == [False, True]
    assert (a > AttrValue.float64(1)).to_list() == [False, True]
    assert (a <= AttrValue.float64(1)).to_list() == [True, False]
    assert a.infer_numeric_type() == "Float64"
    assert a.sort(False).to_list() == [1 + 2**-40, 1]
    a = BaseArray([2**53 + 1, AttrValue.float64(2**53)])
    assert (a > AttrValue.float64(2**53)).to_list() == [True, False]
    assert a.sort().to_list() == [2**53, 2**53 + 1]


def test_explicit_float32_constructors_preserve_width_and_reject_finite_overflow():
    assert AttrValue.float32(0.1).type_name == "float"
    assert AttrValue.float32(0.1).value != 0.1
    assert AttrValue.float_vec([0.1]).type_name == "float_vec"
    assert math.isnan(AttrValue.float32(math.nan).value)
    for constructor, value in [(AttrValue.float32, 1e300), (AttrValue.float_vec, [1e300])]:
        with pytest.raises(ValueError):
            constructor(value)


def test_table_group_aggregation_uses_shared_native_f64_kernels():
    x = 1 + 2**-40
    source = BaseTable.from_dict({"g": ["a", "a"], "x": [AttrValue.float64(x)] * 2})
    grouped = source.group_by(["g"]).agg({"x": "mean"})
    assert grouped.column("x").to_list() == [x]
