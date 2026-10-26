"""Physical verification reads disk, rejects tails/corruption and never repairs bytes."""
import os

import pytest
from groggy import DurableGraph, Graph

pytestmark = pytest.mark.skipif(os.name != "posix", reason="inode-checked verification requires Unix")


def test_physical_verification_matches_retained_graph_and_preserves_named_bytes(tmp_path):
    source = Graph()
    source.add_node(value=1)
    path = tmp_path / "graph.ggj"
    with DurableGraph.create_from_graph(path, source) as store:
        store.set_schema({}, store.current_state_id, 0)
        before = path.read_bytes()
        report = store.verify_storage()
        assert report["file_bytes"] == len(before)
        assert report["logical_records"] == 2
        assert report["graph"] == store.check_integrity()
        assert path.read_bytes() == before
        store.compact_storage()
        assert store.verify_storage()["logical_records"] == 1
        with pytest.raises(Exception):
            DurableGraph.open(path)


def test_partial_tail_is_not_repaired_by_verification_and_requires_reopen(tmp_path):
    source = Graph()
    source.add_node(value=1)
    path = tmp_path / "graph.ggj"
    with DurableGraph.create_from_graph(path, source) as store:
        valid = path.read_bytes()
        with path.open("ab") as outsider:
            outsider.write(b"GFRAME01")
        incomplete = path.read_bytes()
        with pytest.raises(ValueError, match="incomplete trailing bytes"):
            store.verify_storage()
        assert path.read_bytes() == incomplete
        with pytest.raises(ValueError, match="reopen"):
            store.current_state_id
    with DurableGraph.open(path) as store:
        assert path.read_bytes() == valid
        assert store.recovered_tail_bytes == 8
        assert store.verify_storage()["graph"]["active_nodes"] == 1


def test_complete_corrupt_frame_is_reported_without_truncation(tmp_path):
    source = Graph()
    source.add_node(value=1)
    path = tmp_path / "graph.ggj"
    with DurableGraph.create_from_graph(path, source) as store:
        corrupt = bytearray(path.read_bytes())
        corrupt[-1] ^= 1
        with path.open("r+b") as outsider:
            outsider.seek(len(corrupt) - 1)
            outsider.write(corrupt[-1:])
        with pytest.raises(ValueError, match="checksum"):
            store.verify_storage()
        assert path.read_bytes() == bytes(corrupt)
        with pytest.raises(ValueError, match="reopen"):
            store.current_state_id
    with pytest.raises(ValueError, match="checksum"):
        DurableGraph.open(path)
    assert path.read_bytes() == bytes(corrupt)
