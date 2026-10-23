"""Plain interchange retains finite f64 values; grouping retains typed keys."""
import math

import pytest
from groggy import AttrValue, BaseTable


def test_json_round_trip_and_nonfinite_failure_preserve_destination(tmp_path):
    path = tmp_path / "table.json"
    x = 1 + 2**-40
    source = BaseTable.from_dict({
        "x": [AttrValue.float64(x), AttrValue.float64(1e300)],
        "v": [AttrValue.float64_vec([x, 1e300]), AttrValue.float64_vec([])],
    })
    source.to_json(str(path))
    loaded = BaseTable.from_json(str(path))
    assert loaded.column("x").to_list() == [x, 1e300]
    assert loaded.column("v").to_list() == [[x, 1e300], []]
    for value in [AttrValue.float64(math.nan), AttrValue.float64_vec([math.inf])]:
        path.write_text("existing")
        with pytest.raises((ValueError, RuntimeError), match="finite"):
            BaseTable.from_dict({"x": [value]}).to_json(str(path))
        assert path.read_text() == "existing"


def test_csv_round_trip_does_not_double_escape_and_retains_wide_float(tmp_path):
    path = tmp_path / "table.csv"
    text = 'comma, quote" newline\nλ'
    x = 1 + 2**-40
    BaseTable.from_dict({"x": [AttrValue.float64(x)], "text": [text]}).to_csv(str(path))
    loaded = BaseTable.from_csv(str(path))
    assert loaded.column("x").to_list() == [x]
    assert loaded.column("text").to_list() == [text]


def test_grouping_keeps_numeric_text_null_and_vector_keys_distinct():
    x = 1 + 2**-40
    keys = [AttrValue.float64(x), AttrValue.float64(x), 1, "1", None,
            AttrValue.float64_vec([x])]
    grouped = BaseTable.from_dict({"key": keys, "value": [1] * len(keys)}).group_by_agg(
        ["key"], {"value": "sum"}
    )
    rows = list(zip(grouped.column("key").to_list(), grouped.column("value").to_list()))
    assert len(rows) == 5
    assert (x, 2) in rows and (1, 1) in rows and ("1", 1) in rows
    assert (None, 1) in rows and ([x], 1) in rows
