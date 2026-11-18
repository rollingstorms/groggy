"""Historical presentation compatibility and native display boundary regressions."""

import json
from pathlib import Path

import pytest

import groggy
import groggy.display as display
from groggy.display import formatters
from groggy.display.truncation import (
    calculate_column_widths,
    smart_matrix_truncation,
    truncate_columns,
    truncate_rows,
    truncate_string,
)

FIXTURES = json.loads(
    (Path(__file__).parent / "data/display/python-layout-compatibility.json").read_text(
        encoding="utf-8"
    )
)


def fixture_data(value):
    if isinstance(value, dict):
        if "__float__" in value:
            return float(value["__float__"])
        return {key: fixture_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [fixture_data(item) for item in value]
    return value


@pytest.mark.parametrize(
    "case",
    FIXTURES,
    ids=lambda c: f"{c['kind']}-{c['data'].get('shape')}-{c['options']}",
)
def test_historical_dictionary_layouts_match_captured_outputs(case, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    data = fixture_data(case["data"])
    data["shape"] = tuple(data["shape"])
    assert (
        getattr(display, "format_" + case["kind"])(data, **case["options"])
        == case["expected"]
    )


@pytest.mark.parametrize(
    "budget,indices", [(0, []), (1, [0]), (2, [0, 1]), (3, [0, 9]), (4, [0, 1, 9])]
)
def test_array_labels_track_visible_source_positions_and_real_omissions(
    budget, indices
):
    rendered = display.format_array(
        {"data": list(range(10)), "shape": (10,), "name": "value"}, max_rows=budget
    )
    lines = [line for line in rendered.splitlines() if line.startswith("│")][2:]
    actual = [
        int(line.split("│")[1].strip())
        for line in lines
        if line.split("│")[1].strip() != "…"
    ]
    assert actual == indices
    assert sum(line.split("│")[1].strip() == "…" for line in lines) == (budget >= 3)


def test_literal_ellipsis_values_never_become_synthetic_rows():
    rendered = display.format_array({"data": [["…"], "…", "⋯", 7], "shape": (4,)})
    lines = [line for line in rendered.splitlines() if line.startswith("│")][2:]
    assert [line.split("│")[1].strip() for line in lines] == ["0", "1", "2", "3"]
    assert "['…']" in rendered


def test_table_tail_type_headers_follow_selected_columns():
    data = {
        "columns": ["a", "b", "c", "d"],
        "dtypes": {"a": "string", "b": "int64", "c": "float64", "d": "datetime"},
        "data": [["a", 3, 2.5, "2026-11-18"]],
        "shape": (1, 4),
    }
    rendered = display.format_table(data, max_cols=4)
    header, types = [line for line in rendered.splitlines() if line.startswith("│")][:2]
    assert [v.strip() for v in header.split("│")[1:-1]] == ["#", "a", "…", "d"]
    assert [v.strip() for v in types.split("│")[1:-1]] == ["", "str…", "…", "date"]


def test_render_only_formats_visible_cells_and_preserves_python_object_identity():
    class Hidden:
        def __str__(self):
            raise AssertionError("hidden values must not be formatted")

    data = {
        "columns": ["shown", "hidden", "hidden2", "tail"],
        "dtypes": {},
        "data": [
            ["first", Hidden(), Hidden(), "last"],
            [Hidden()] * 4,
            ["end", Hidden(), Hidden(), "tail"],
        ],
        "shape": (3, 4),
    }
    # Index, first column and a marker; no hidden row or hidden column is read.
    result = display.format_table(data, max_rows=1, max_cols=4)
    assert "first" in result and "last" in result
    value = object()
    selected, truncated = truncate_rows([value, object(), value], 1)
    assert truncated and selected[0] is value


def test_shared_truncation_handles_zero_tiny_custom_unicode_and_ragged_inputs():
    assert truncate_string("猫😺abc", 3, "--") == "猫--"
    assert truncate_string("猫😺abc", 1, "--") == "猫"
    assert truncate_string("猫😺abc", 0) == ""
    assert truncate_rows([1, 2, 3], 1) == ([1], True)
    assert truncate_columns(["a", "b", "c", "d"], [[1, 2, 3, 4]], 3) == (
        ["a", "…", "d"],
        [[1, "⋯", 4]],
        True,
    )
    assert smart_matrix_truncation([[1, 2], [3, 4]], 0, 0) == ([], True, True)
    assert calculate_column_widths(["猫"], [["😺"]], 0) == [3]


def test_configuration_is_effective_atomic_and_preserves_explicit_overrides(
    monkeypatch,
):
    monkeypatch.setattr(display._presentation, "_DISPLAY_OVERRIDES", {})
    for name in [
        "DEFAULT_MAX_ROWS",
        "DEFAULT_MAX_COLS",
        "DEFAULT_MAX_WIDTH",
        "DEFAULT_PRECISION",
    ]:
        monkeypatch.setattr(display, name, getattr(display, name))
    old = display.ArrayDisplayFormatter()
    assert (old.max_rows, old.precision, old.max_width) == (10, 4, 60)
    display.configure_display(max_rows=1, precision=1, max_cols=2)
    assert old.max_rows == 10
    assert display.ArrayDisplayFormatter().max_rows == 1
    assert display.MatrixDisplayFormatter().max_cols == 2
    assert display.TableDisplayFormatter(max_cols=4).max_cols == 4
    assert display.ArrayDisplayFormatter(precision=0).precision == 0
    with pytest.raises(ValueError, match="max_width"):
        display.configure_display(max_rows=9, max_width=-1)
    assert display.ArrayDisplayFormatter().max_rows == 1
    with pytest.raises(TypeError):
        display.configure_display(max_rows=1.5)


def test_color_policy_remains_a_python_terminal_adapter(monkeypatch):
    data = {"columns": ["a"], "dtypes": {}, "data": [[True]], "shape": (1, 1)}
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    colored = display.format_table(data)
    assert "\x1b[1m" in colored and "\x1b[2m" in colored
    monkeypatch.setenv("NO_COLOR", "1")
    plain = display.format_table(data)
    assert "\x1b" not in plain
    assert (
        colored.replace("\x1b[1m", "").replace("\x1b[2m", "").replace("\x1b[0m", "")
        == plain
    )


def test_public_imports_detection_and_shape_summary_remain_compatible():
    assert groggy.format_array is display.format_array is formatters.format_array
    assert formatters._detect_data_type({"data": [[1]]}) == "matrix"
    with pytest.raises(ValueError, match="Unknown data type"):
        formatters.format_data_structure({}, "unknown")
    data = {
        "columns": ["a"],
        "dtypes": {},
        "data": [[2**80]],
        "shape": (1000, 1),
        "table_type": "GraphTable",
        "nodes_shape": (1000, 4),
        "edges_shape": (2000, 3),
    }
    output = display.format_table(data)
    assert str(2**80) in output
    assert "rows: 1,000" in output
    assert "nodes: 1,000×4 • edges: 2,000×3" in output


def test_date_clipping_has_one_ellipsis_within_its_character_budget():
    formatter = display.TableDisplayFormatter()
    value = formatter._format_value("2026-11-18T12:30:00", "datetime")
    assert value == "2026-11-1…"
    assert len(value) == 10


def test_native_dictionary_bridge_preserves_boolean_and_nonfinite_categories():
    from groggy._groggy import DisplayConfig, TableFormatter

    config = DisplayConfig(use_color=False)
    # Typed translation must not treat bool as its integer superclass or turn
    # nonfinite floats into an invented zero.
    output = TableFormatter.format_data_structure(
        {"data": [False, True, float("nan"), float("inf")], "shape": (4,)},
        "array",
        config,
    )
    rows = [line for line in output.splitlines() if line.startswith("│")][2:]
    assert "false" in rows[0] and "true" in rows[1]
    assert "NaN" in rows[2] and "NaN" in rows[3]


def test_native_layout_workers_use_only_owned_text_after_marshalling():
    from concurrent.futures import ThreadPoolExecutor
    from groggy._groggy import _display_table

    headers = ["#", "text"]
    types = ["", "str"]
    rows = [[str(index), "猫😺<&"] for index in range(500)]

    def render(_):
        return _display_table(headers, types, rows, False, "summary", 40, False)

    expected = render(None)
    with ThreadPoolExecutor(max_workers=3) as executor:
        assert all(output == expected for output in executor.map(render, range(9)))


def test_untruncated_compatibility_projection_retains_original_containers():
    matrix = [[object()], [object()]]
    selected, row_cut, col_cut = smart_matrix_truncation(matrix, 10, 10)
    assert selected is matrix and not row_cut and not col_cut
    selected, row_cut, col_cut = smart_matrix_truncation(matrix, 1, 10)
    assert selected[0] is matrix[0] and row_cut and not col_cut
    headers = ["a"]
    h, rows, cut = truncate_columns(headers, matrix, 10)
    assert h is headers and rows is matrix and not cut


def test_native_dictionary_bridge_keeps_numeric_protocol_compatibility():
    import numpy as np
    from groggy._groggy import DisplayConfig, TableFormatter

    config = DisplayConfig(use_color=False)
    output = TableFormatter.format_data_structure(
        {
            "data": [np.int64(-(2**63)), np.uint64(2**64 - 1), np.float32(1.25)],
            "shape": (3,),
        },
        "array",
        config,
    )
    assert str(-(2**63)) in output
    assert str(2**64 - 1) in output
    assert "1.25" in output


def test_invalid_unicode_rejects_without_replacing_or_mutating_input():
    value = "\ud800"
    data = {"data": [value], "shape": (1,)}
    with pytest.raises(UnicodeEncodeError):
        display.format_array(data)
    assert data["data"][0] is value


def test_native_visible_object_format_error_propagates_without_a_panic():
    from groggy._groggy import TableFormatter

    class Broken:
        def __str__(self):
            raise ValueError("display callback failed")

    value = Broken()
    data = {"data": [value], "shape": (1,)}
    with pytest.raises(ValueError, match="display callback failed"):
        TableFormatter.format_data_structure(data, "array", None)
    assert data["data"][0] is value
