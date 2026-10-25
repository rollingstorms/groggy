# Typed graph-table bundles

`GraphTable.save_bundle(directory)` now writes format **3.0** to one authoritative
`bundle.cbor` file. It contains both ordered columnar tables, logical property
tags, validation policy and metadata. Loading and verification share the native
codec; Python adapters marshal results and release the GIL during native work.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import BaseTable, NodesTable, EdgesTable, GraphTable
with TemporaryDirectory() as directory:
    table = GraphTable(
        NodesTable.from_base_table(BaseTable.from_dict({
            "node_id": [8, 3], "name": ["", "true"], "x": [1.0000000000000002, None]
        })),
        EdgesTable.from_base_table(BaseTable.from_dict({
            "edge_id": [9], "source": [8], "target": [3], "vector": [[1.25, 2.5]]
        })),
    )
    table.save_bundle(directory)
    loaded = GraphTable.load_bundle(directory)
    assert loaded.nodes.base_table()["name"].to_list() == ["", "true"]
    assert loaded.nodes.base_table()["x"].to_list() == [1.0000000000000002, None]
    info = GraphTable.get_bundle_info(directory)
    assert info["version"] == "3.0" and info["node_count"] == 2
    assert info["checksum_algorithm"] == "xxh3_64"
    assert GraphTable.verify_bundle(directory)["checksum_verified"]
    assert (Path(directory) / "bundle.cbor").is_file()
```

## What is preserved

Inline scalar, vector, bytes, JSON and temporal values reuse Groggy's versioned
property codec. f32/f64 remain distinct; NaN payloads, signed zero and infinities
retain their bit patterns. Optimized small integers/compressed text or vectors
are restored as equivalent logical values, without promising the original memory
representation. Empty tables retain declared columns. IDs and endpoint values,
row order, column order and validation policy are preserved.

This is an owned table export. Captured source views, automatic source-row
writeback identity, graph branch/history and pool contents are not exported.
`SubgraphRef` rejects explicitly because its pool is required; it no longer
becomes a placeholder string. Use the separately documented graph/journal
recovery APIs for their supported history guarantees.

```python
from groggy import Graph
with TemporaryDirectory() as directory:
    owner = Graph()
    owner.add_node(x=1)
    owner.read_view().table().save_bundle(directory)
    restored = GraphTable.load_bundle(directory).nodes.base_table()
    assert restored.source_views() == []
    assert restored.source_entity_ids is None
    owner.set_node_attr(0, "x", 99)
    assert restored["x"] == 1
```

## Integrity and replacement

The binary frame has a fixed version header, exact length and xxh3 checksum.
Readers reject corruption, truncation, trailing data, unknown versions/tags,
missing structural columns, duplicate columns and inconsistent row counts.
`verify_bundle` returns `is_valid`, `checksum_verified` and `errors`.
`is_valid` describes the encoded bundle's integrity/schema; graph topology and
policy validation are reported separately in metadata's `validation_summary`.
Checksums detect accidental changes; they are not authentication.

```python
with TemporaryDirectory() as directory:
    table.save_bundle(directory)
    path = Path(directory) / "bundle.cbor"
    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    result = GraphTable.verify_bundle(directory)
    assert not result["is_valid"] and not result["checksum_verified"]
    assert result["errors"]
    try:
        GraphTable.load_bundle(directory)
    except RuntimeError:
        pass
    else:
        raise AssertionError("corrupt bundle was accepted")
```

Encoding and schema preparation finish before publication. Save writes and syncs
a temporary file in the destination directory, atomically replaces `bundle.cbor`,
then syncs the directory. A failed encode leaves the previous file untouched.
A directory-sync error after replacement can leave a complete new file: reopen
and verify before retrying. Concurrent writers require caller coordination;
there is no bundle transaction lock. Encoding/decoding currently materializes
whole tables in memory.

## Existing v2 CSV bundles

Load/verify also accept existing format 2.0 metadata, manifest and CSV files.
Their actual historical checksums, normalized metadata checksum, manifest,
column schema and row counts are now verified. Historical fields named
`*_sha256` used Rust `DefaultHasher`; metadata identifies that legacy algorithm
accurately. There is no new v2 writer and no claimed v1 placeholder support.
CSV already discarded some property type information, so legacy reads cannot
recover it. Saving a loaded v2 bundle writes v3; the new authoritative binary
file takes precedence over old sidecars, which are retained without deletion.

Native/Python regressions include a frozen v2 fixture under
`tests/data/graph_table_bundle_v2`. Current costs are captured in
`notes/benchmarks/graph-table-bundles-2026-10-25.json`; verification currently
performs full decoding and is not advertised as a constant-time metadata read.
