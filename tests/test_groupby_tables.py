import math

import groggy as gr


def _build_sample_graph():
    g = gr.Graph()
    g.add_node(object_name="Widget", successful_methods=6, is_active=True)
    g.add_node(object_name="Widget", successful_methods=4, is_active=False)
    g.add_node(object_name="Gadget", successful_methods=10, is_active=True)
    g.add_node(object_name="Gadget", successful_methods=2, is_active=True)
    g.add_node(object_name="Gizmo", successful_methods=1, is_active=True)
    return g


def _manual_group_stats(grouped, value_col):
    manual = {}
    for nodes_table in grouped.to_list():
        name = nodes_table["object_name"][0]
        values = nodes_table[value_col].to_list()
        manual[name] = values
    return manual


def _table_to_mapping(table, key_column, value_column):
    keys = table[key_column]
    values = table[value_column]
    row_count = len(table)
    result = {}
    for idx in range(row_count):
        key = keys[idx]
        result[key] = values[idx]
    return result


def test_groupby_mean_matches_array_mean():
    g = _build_sample_graph()
    grouped = g.nodes.table().group_by("object_name")

    table_mean = grouped.mean()
    array_mean_table = grouped["successful_methods"].mean()

    # Ensure renamed key column propagates
    assert "object_name" in table_mean.columns
    assert "object_name" in array_mean_table.columns

    manual = {
        name: sum(values) / len(values)
        for name, values in _manual_group_stats(grouped, "successful_methods").items()
    }
    table_means = _table_to_mapping(table_mean, "object_name", "successful_methods")
    array_means = _table_to_mapping(array_mean_table, "object_name", "mean")

    for key, value in manual.items():
        assert math.isclose(table_means[key], value, rel_tol=1e-6)
        assert math.isclose(array_means[key], value, rel_tol=1e-6)


def test_groupby_max_matches_manual():
    g = _build_sample_graph()
    grouped = g.nodes.table().group_by("object_name")

    table_max = grouped.max()
    manual = {
        name: max(values)
        for name, values in _manual_group_stats(grouped, "successful_methods").items()
    }
    table_max_map = _table_to_mapping(table_max, "object_name", "successful_methods")

    assert manual == table_max_map


def test_groupby_all_booleans():
    g = _build_sample_graph()
    grouped = g.nodes.table().group_by("object_name")

    table_all = grouped.all()
    manual = {
        name: all(values)
        for name, values in _manual_group_stats(grouped, "is_active").items()
    }
    table_all_map = _table_to_mapping(table_all, "object_name", "is_active")

    assert manual == table_all_map
