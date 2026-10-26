# Physical journal verification

`DurableGraph.verify_storage()` reads the current named journal's actual bytes.
It validates framing, sequences, checksums and semantic replay through the same
parser used by recovery, then verifies every retained graph state and the current
schema. It returns `file_bytes`, `logical_records`, and a nested `graph` integrity
report. This is an explicit expensive audit; it is not part of ordinary writes.

```python
import tempfile
from pathlib import Path
from groggy import Graph, DurableGraph

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "graph.ggj"
    source = Graph()
    source.add_node(value=1)
    with DurableGraph.create_from_graph(path, source) as store:
        original = path.read_bytes()
        report = store.verify_storage()
        assert report["file_bytes"] == len(original)
        assert report["logical_records"] == 1
        assert report["graph"] == store.check_integrity()
        assert path.read_bytes() == original
```

This differs from `check_integrity()`, which checks an owned capture of the current
in-memory graph and schema. Physical verification reconstructs its report from the
file, including a full retained baseline and subsequent operations. It checks
captured version/sequence/schema revision against the owner. It does not compare
all existing in-memory values byte for byte against the recovered graph, certify
external tools that ignore file locks, or inspect derived indexes/caches.

Verification never truncates or repairs the source. An incomplete final frame is
an error. The owner then requires close/reopen before further operations; ordinary
recovery may discard an incomplete final frame as documented. A complete corrupt
frame remains an error during reopen and is not truncated. Changed paths/symlinks
are rejected; an unrelated named replacement is never touched.

The existing inode lock remains held while a separate read-only cursor scans the
file, so verification does not disturb append positioning. Detached input/outcome
holds the lock even if the owner is dropped. Owner operations are blocked until
completion, and only a successful result from the original owner reenables it.
Abandoning a plan/outcome requires reopening. Native integrations must finish the
outcome, including errors, through `finish_storage_verification`.

Python releases the GIL during file reading, replay and retained-state checks.
The operation owns a reconstructed graph plus explicit integrity-validation
buffers; time and memory scale with file/history size. Source verification uses
no source writes; large-baseline framing may use temporary scratch storage, as
recovery already does. The inode/path checks currently require Unix. This is not
a substitute for backups or a power-loss certification.
