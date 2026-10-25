"""Checksummed typed bundles preserve inline data and detach source identity."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import math
import shutil
import struct

import pytest
from groggy import BaseTable, EdgesTable, Graph, GraphTable, NodesTable


def column(table, name):
    value = table[name]
    return value.to_list() if hasattr(value, "to_list") else [value]


def bundle(values):
    n = len(values)
    nodes = NodesTable.from_base_table(BaseTable.from_dict({"node_id": list(range(n)), 'odd,"\nname': values}))
    edges = EdgesTable.from_base_table(BaseTable.from_dict({"edge_id": list(range(n)), "source": [0] * n, "target": [0] * n, "x": values}))
    return GraphTable(nodes, edges)


def test_binary_roundtrip_preserves_types_precision_and_quoted_schema(tmp_path):
    values = [None, True, 2**60 + 1, 1.0000000000000002, -0.0, float("inf"),
              float("nan"), 'a,"b\nλ', "", "true", b"\x00\xff", [1.0000000000000002, 2.0],
              date(2024, 2, 29), datetime(2024, 1, 1, tzinfo=timezone.utc), timedelta(days=-1, microseconds=1)]
    bundle(values).save_bundle(str(tmp_path))
    loaded = GraphTable.load_bundle(str(tmp_path))
    actual = column(loaded.nodes.base_table(), 'odd,"\nname')
    for before, after in zip(values, actual):
        if isinstance(before, float) and math.isnan(before):
            assert math.isnan(after)
        else:
            assert before == after
    assert struct.pack("d", actual[4]) == struct.pack("d", -0.0)
    info = GraphTable.get_bundle_info(str(tmp_path))
    assert info["version"] == "3.0"
    assert info["node_count"] == len(values)
    assert info["checksum_algorithm"] == "xxh3_64"
    assert info["schema_info"]["node_columns"] == ["node_id", 'odd,"\nname']
    assert GraphTable.verify_bundle(str(tmp_path))["checksum_verified"] is True


def test_empty_typed_bundle_keeps_declared_columns(tmp_path):
    bundle([]).save_bundle(str(tmp_path))
    loaded = GraphTable.load_bundle(str(tmp_path))
    assert loaded.is_empty()
    assert loaded.nodes.base_table().columns == ["node_id", 'odd,"\nname']


@pytest.mark.parametrize("damage", ["truncate", "flip", "trailing", "header"])
def test_corrupt_bundles_fail_both_verification_and_loading(tmp_path, damage):
    bundle([1]).save_bundle(str(tmp_path))
    path = tmp_path / "bundle.cbor"
    data = bytearray(path.read_bytes())
    if damage == "truncate":
        data = data[:-1]
    elif damage == "flip":
        data[-1] ^= 1
    elif damage == "trailing":
        data += b"extra"
    else:
        data[0] ^= 1
    path.write_bytes(data)
    result = GraphTable.verify_bundle(str(tmp_path))
    assert result["is_valid"] is False
    assert result["checksum_verified"] is False
    assert result["errors"]
    with pytest.raises(RuntimeError):
        GraphTable.load_bundle(str(tmp_path))
    with pytest.raises(RuntimeError):
        GraphTable.get_bundle_info(str(tmp_path))


def test_repeated_save_replaces_complete_bundle_without_temporary_files(tmp_path):
    bundle([1]).save_bundle(str(tmp_path))
    bundle([2]).save_bundle(str(tmp_path))
    assert column(GraphTable.load_bundle(str(tmp_path)).nodes.base_table(), 'odd,"\nname') == [2]
    assert [path.name for path in tmp_path.iterdir()] == ["bundle.cbor"]


def test_legacy_v2_integrity_is_checked_and_can_be_resaved_as_v3(tmp_path):
    source = Path(__file__).parent / "data" / "graph_table_bundle_v2"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    assert GraphTable.verify_bundle(str(tmp_path))["checksum_verified"] is True
    loaded = GraphTable.load_bundle(str(tmp_path))
    assert column(loaded.nodes.base_table(), "node_id") == [8, 3]
    assert GraphTable.get_bundle_info(str(tmp_path))["version"] == "2.0"
    loaded.save_bundle(str(tmp_path))
    (tmp_path / "nodes.csv").write_text("corrupted legacy sidecar")
    assert GraphTable.verify_bundle(str(tmp_path))["format_version"] == "3.0"


def test_legacy_csv_content_changes_fail_instead_of_reporting_file_existence(tmp_path):
    source = Path(__file__).parent / "data" / "graph_table_bundle_v2"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    with (tmp_path / "nodes.csv").open("a") as handle:
        handle.write("\n")
    assert GraphTable.verify_bundle(str(tmp_path))["is_valid"] is False
    with pytest.raises(RuntimeError, match="checksum"):
        GraphTable.load_bundle(str(tmp_path))


def test_restored_tables_have_no_connection_to_the_original_graph(tmp_path):
    graph = Graph()
    graph.add_node(x=1)
    graph.read_view().table().save_bundle(str(tmp_path))
    nodes = GraphTable.load_bundle(str(tmp_path)).nodes.base_table()
    assert nodes.source_views() == []
    assert nodes.source_entity_ids is None
    graph.set_node_attr(0, "x", 99)
    assert column(nodes, "x") == [1]


def test_missing_directory_is_a_failed_verification_result(tmp_path):
    result = GraphTable.verify_bundle(str(tmp_path / "missing"))
    assert result["is_valid"] is False
    assert result["checksum_verified"] is False
