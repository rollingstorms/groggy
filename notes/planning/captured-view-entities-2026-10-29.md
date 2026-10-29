# Captured entity and column access — 26.1 / AR7.5 / 21.2

Completed the explicit captured accessor path. GraphView remains the sole owner
of frozen state/membership; GraphViewEntity, GraphViewEntities and their iterator
are lightweight read adapters with no working graph, history or mutation owner.
Python exports the two public read types and exposes view.nodes/view.edges.

## Contracts

- Entity lookup is by original ID. Positional access is explicit .at(position) in
  sorted ID order, with negative positions supported. Lists/tuples are ID
  selections; slices select sorted positions. Child graph membership reports
  ascending IDs, including reverse slices. Existing connected indexing defaults
  are not changed by this new captured API.
- ID ordering is lazy and shared within a retained accessor/iterator. Constructing
  an accessor, scalar ID lookup, length and membership do not sort all IDs.
- Owned BaseArray columns align to sorted IDs and retain typed values. Missing
  cells project to null, with separate presence bits preserving absent versus
  explicit-null distinctions. Scalar keys/membership/to_dict include present
  properties only. Attribute notation rejects a wholly absent column; explicit
  column/bracket reads return aligned nulls. No implicit graph writeback exists.
- Masks require exact length and boolean-only BaseArrays, with valid empty
  results. Nodes retain the parent's exact internal-edge scope; edge selections
  retain exact instances/endpoints. Node all retains isolates; edge all retains
  endpoints only. Filters use the existing canonical captured executor.
- Entity degree/neighbors/endpoints and meta-source lookup use the captured
  selection, not current owner liveness or owner-wide adjacency. Handles,
  iterators, child views and columns survive source mutation/removal/release;
  entity/accessor reads can run on workers. Wrong-kind topology operations reject.
- FFI handles type translation and GIL release; masks cross the GIL boundary as
  native array clones rather than PyRef borrows. All selection/column/entity
  behavior lives in Rust. New public classes/stubs are curated without touching
  the owner's pending benchmark/review artifacts.

## Verification

Six native cases cover sparse original IDs versus positions, checked slices and
mask shape/type, null/absence/large integer alignment, owned COW column independence,
exact scope, iterator/source lifetime, workers, retained meta revisions and the
parallel ordered projection path above 1000 rows. Eighteen Python cases exercise
public classes/columns/filters, readonly mutation rejection, explicit native
writeback, time coordinates, invalid ID/position types and owner/meta lifetimes.
The executable guide adds four examples: `docs/api/captured-view-entities.md`.

Final full gates appear in the priority-window log. No serialization or persistent
index format change, full graph clone or new mutation/publication path.

## Measured bulk projection and scalar cost

`benches/captured_view_entities.rs`, raw paired quiet release receipt:
`notes/benchmarks/captured-view-entities-2026-10-29.json`. Four sequential processes
alternate serial/parallel implementations, with five samples each and identical
checks. Fixture: 1k/100k accepted isolates with integer/null/absent values. Scalar
reads occur before positional-order materialization; IDs, columns, presence and
masks have separate cost fields. Setup/capture and Python conversion are excluded.

| 100k rows | Serial median | Ordered parallel median |
|---|---:|---:|
| Owned column | 25.006 ms | 5.430 ms |
| Presence bits | 19.638 ms | 6.461 ms |
| Fresh accessor scalar | 110.448 ns | 112.373 ns |
| Reused accessor scalar | 77.981 ns | 81.006 ns |

Larger columns use the existing Rayon projection policy (>1000 rows), preserving
input order; smaller columns stay serial (about 0.050–0.053 ms at 1k). Scalar costs
stay independent of ID sorting in these fixtures. These are native local receipts,
not certification of the 100ns Python FFI budget or a general throughput/memory
bound. Owned columns copy values; live accessors/handles retain their capture.

## Fixed remaining boundary

The existing connected Subgraph/accessors and mutable entity APIs retain their
legacy ownership/indexing. Default connected/filter return migration and the
remaining read/write facades stay in existing 26.1/AR4/AR7.5/21.2. Native legacy
Subgraph column helpers still drop missing rows; that is the next alignment repair,
not a new checklist row. Compound topology/custom effects remain 09.4b.2.
Inventory remains **211 checked / 57 unchecked**.
