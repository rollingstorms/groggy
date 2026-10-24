# Plain table interchange and typed grouping

CSV floating fields and plain table JSON floating numbers/numeric arrays infer
f64. Finite f64 values are not narrowed to f32, including vectors in JSON. The
JSON parser enables round-trip floating parsing so serialized f64 values retain
their binary value on import. CSV quoting is handled once by the CSV writer,
including text with commas, quotes and embedded newlines.

These are untyped interchange formats. They do not retain every source tag:
CSV imports integer-looking fields as integers, numeric-looking text may be
inferred as numbers, and vector-looking CSV fields remain text. Plain JSON does
not preserve explicit f32 tags; other structured/pooled types retain the existing
text fallback. Use the tagged graph codec for typed graph persistence/history.

Plain JSON cannot represent NaN or infinities. Export rejects these scalar or
floating-vector values instead of silently replacing them with zero. Validation
and serialization happen before opening the destination, preserving an existing
file on value/serialization rejection. This is not atomic recovery from an I/O
failure during the eventual file write.

`group_by_agg` groups by borrowed typed property values and emits owned copies of
the actual keys, rather than parsing their debug strings. Numeric and text keys
remain distinct, f32 and f64 retain their explicit identities, and vector keys
remain vectors. Floating identity follows the property type's existing zero/NaN
rules. Group output order remains unspecified. Grouping does not allocate graph
entities or publish graph changes. Python grouping and BaseTable CSV/JSON calls
release the GIL while Rust runs.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import AttrValue, BaseTable

x = 1 + 2**-40
source = BaseTable.from_dict({"key": [AttrValue.float64(x)] * 2, "count": [1, 1]})
grouped = source.group_by_agg(["key"], {"count": "sum"})
assert grouped.column("key").to_list() == [x]
assert grouped.column("count").to_list() == [2]
with TemporaryDirectory() as directory:
    path = str(Path(directory) / "table.json")
    source.to_json(path)
    assert BaseTable.from_json(path).column("key").to_list() == [x, x]
```

For explicit mixed node/edge property updates of an existing graph, see
[table writeback](table-writeback.md). Interchange and grouping alone do not mutate
the source graph or supply verified graph provenance.
