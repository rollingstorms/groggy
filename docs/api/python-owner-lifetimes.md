# Python owners, borrowing and finalization

Graph access and graph destruction have different requirements. Live graph state
remains confined to the creating thread. Final release of a native owner is safe
on any thread, including Python cyclic GC running on a worker or after the creator
has exited. No creator-thread poll or deferred-drop queue is required.

## Application use

Keep Graph, DurableGraph, writers and connected graph access on their creating
thread. Pass detached immutable GraphView captures to workers:

```python
from concurrent.futures import ThreadPoolExecutor
from groggy import Graph

graph = Graph()
node = graph.add_node(score=1.25)
captured = graph.read_view()
with ThreadPoolExecutor(max_workers=1) as workers:
    assert workers.submit(captured.node_attr, node, "score").result() == 1.25
graph.writer().set_node_attr(node, "score", 2.5)
del graph
assert captured.node_attr(node, "score") == 1.25
```

Use DurableGraph's context manager or explicit close for a deterministic storage
lifecycle. If Python GC releases an unclosed durable owner on another thread, its
native destructor still releases the journal lock immediately. An explicit close
from a foreign thread rejects; safe destruction does not authorize foreign live
operations or concurrent graph writers.

Copied IDs or owned values do not require live graph borrowing, but this change
does not certify every legacy adapter method for worker execution. GraphView is
the supported detached worker interface. Foreign live-state borrowing currently
rejects with PyO3's panic exception, preserving the former thread-guard rejection
style; a broader Python error/API contract remains checklist item 19.2.

## Ownership design

Core connected handles now use `Arc<OwnerCell<T>>`. OwnerCell checks its creating
thread before borrowing and uses nonblocking lock guards: nested reads work and
conflicting mutable borrows fail immediately. Atomic reference counts and the
contents' compiler-checked Send/Sync traits make final release safe. This does not
freeze/copy the graph per call or add per-entity locks.

There are no custom unsafe Send implementations, global GC changes or suppressed
unraisable errors. The thread-ID cache contains only a copied ThreadId, not graph
data. The graph/pool/connected owner destructors need no thread-local registry.
The formerly raw-pointer attribute collection now retains its graph owner. The
unimplemented matrix View storage metadata retains an immutable backing buffer;
this ownership repair does not implement matrix View operations.

## Native Rust migration for 0.6.0

- Use `Graph::into_shared()` for entity/subgraph owner construction. Explicit
  `Rc<RefCell<Graph>>` handles become `Arc<groggy::ownership::OwnerCell<Graph>>`.
- Pool borrowing returns OwnerReadGuard/OwnerWriteGuard instead of std::cell
  Ref/RefMut. Public read-only pool access still cannot mutate the pool.
- Custom TemporalStorageStrategy implementations must satisfy Send + Sync.
- Explicit MatrixView construction supplies a retained `Arc<[T]>` buffer in its
  data field instead of a bare pointer. Operations on this placeholder storage
  variant remain unsupported.

Python method signatures and checked publication/history semantics are unchanged.
See the [qualification receipt](../../notes/planning/python-owner-lifetime-2026-11-14.md)
for lifetime regressions and measured access costs.
