# Shared live/frozen query predicates — 2026-10-22

AR7.5e moves the existing filter algebra onto shared read-source adapters and
adds immutable GraphView filtering. AR7.5/AR4/26.1 and broader query/index item 21
remain open; algorithm read migration follows separately.

## Defects repaired

Legacy Subgraph Python filtering called find_nodes/find_edges on the whole
parent graph, then constructed a new selection from all results. It could add
outside nodes/edges. Its node construction also re-induced all graph edges,
reintroducing excluded parallel instances. Native attribute conveniences had the
same parent-edge re-induction issue. Subgraph now executes within its selected
nodes/edge instances and constructs the result in Rust. Python delegates rather
than reconstructing graph membership itself. Stale or inconsistent live selections
fail explicitly; frozen selections remain valid after owner changes/deactivation.

Numeric range predicates previously converted i64 values to f64, losing ordering
between adjacent large integers. AttributeFilter now uses the existing canonical
numeric comparison helper for exact integer/mixed boundaries and IEEE unordered
NaNs. Equality/inequality retain existing AttrValue semantics; Boolean/text/null
values do not become numeric ordering operands.

## Shared implementation and scope

QueryEngine, legacy Subgraph and immutable GraphView use query/filtering's single
node/edge predicate executor. It resolves one requested attribute column at a
time, borrowing payloads and retaining the existing parallel matching threshold.
Live source adapters resolve GraphSpace references against the borrowed pool;
dangling visited references are errors. Frozen adapters read the existing pin.
No attribute payload is cloned per entity for matching. Boolean conjunction
shrinks candidates, while topology predicates retain the original selection's
edge context. A live incident index is built lazily once per query, replacing
per-candidate full-edge scans. It is not cached across property/topology writes.

GraphView.filter_nodes induces within the parent edge scope; filter_edges retains
matching instances and their endpoints. Children share the state pin and never
refresh from the working graph. Python accepts existing typed filters and parser
strings, releases the GIL over frozen evaluation/selection, and returns GraphView.
Python NodeFilter factories now also expose existing native DegreeRange,
HasNeighbor and NodeSet variants. Working Subgraph queries remain current reads;
query filtering does not freeze all ordinary adapters or mutate graph/history.

## Preserved predicate semantics

- HasAttribute includes explicit null. Missing properties do not match direct
  attribute predicates (including NotEquals); Not complements within candidates.
- Empty node And and Or match nothing, as before. Empty edge And matches all
  candidates; empty edge Or matches none. This asymmetry is preserved explicitly.
- DegreeRange counts selected incident edge instances, with loops once and
  parallel instances separately. HasNeighbor keeps the legacy incident-neighbor
  query meaning for directed graphs; either orientation qualifies. A node is its
  own neighbor only through a selected self-loop. This differs from outgoing
  GraphView.neighbors() and is documented rather than silently redefined.
- ConnectsNodes reverses endpoints only for undirected graphs. ConnectsAny tests
  either endpoint. Nested results cannot escape their parent selection.
- Frozen result ID lists are sorted; live QueryEngine preserves its supplied
  candidate order through Boolean combinations. No promise is added for the
  unordered current GraphSpace input order or legacy NumArray return wrappers.

This reuses the existing parser, not a new expression language or planner. It does
not certify placeholder endpoint-attribute convenience filters, merge array/table
expression parsers, add index execution, or freeze inherited algorithm methods.
The Graph Python facade retains its existing attribute-name validation; frozen
predicates follow native missing-value rules. No workbook policy is chosen here.

## Evidence

Native tests cover every current NodeFilter/EdgeFilter variant, directed and
undirected scope, loops/parallel facts, null/absent, Boolean empty cases, stale
selections, mutation/owner drop and exact integer/mixed/NaN boundaries. A 2,001-row
sparse property fixture exercises the parallel column path. Python tests exercise
legacy nested scope repair, typed/string queries, error translation, factories,
threaded frozen queries, precision and captured values. Existing core/subgraph/
accessor/algorithm/history/property tests and all documentation examples run
against the rebuilt in-tree native extension.

Bounded release costs: `notes/benchmarks/frozen-query-2026-10-22.md`. Output creation
is included; no old-engine speedup, production memory/latency budget or FFI
per-call claim is made.

Executed gates: full locked offline native suites plus the final five-fixture
native query suite; strict all-target core Clippy; rebuilt in-tree Python extension
(three existing FFI warnings). Combined core/subgraph/accessor/algorithm/read/
history/property Python gate: **255 passed, 5 existing skips**. All five documented
read-view Python examples execute successfully.
