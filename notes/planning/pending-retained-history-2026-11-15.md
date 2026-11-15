# Retained history inside pending programs — 2026-11-15

## Completed behavior

The existing retained snapshot acquisition and node/edge diff steps now declare
the shared immutable-input capability. Their owner-checked retained sources are
independent of unpublished working topology/properties. This permits local and
outer atomic composition, nested loops and certified child bodies without
inventing accepted creation states for provisional nodes/edges.

State-ID snapshot selection remains global within retained owner history;
recorded timestamp selection remains Unix seconds on the current first-parent
lineage. Neither is a property effective-time query. The Context temporal scope
and explicit snapshot variables retain their prior resolution and provenance
checks. Endpoint revisions appear in edge modified masks as before. No read
creates a delta, moves the head or switches branches.

Pending captures still expose working values, and historical snapshots remain
unchanged. Diff result IDs are never filtered/remapped into current IDs. Invalid
retained sources and invalid selected attachment targets preserve atomic rollback,
outputs, accepted state and allocation coordinates. Source/column authority rules
remain unchanged. The native immutable-input description now explicitly includes
retained inputs. No public method or storage format changes.

## Qualification

Locked/offline native suite: **1,601 passed / 54 ignored / zero failed**, 223 suite
results including doctests. Strict all-target Clippy, Rust formatting and whitespace
checks pass. Rebuilt the in-tree release extension without wheel installation;
three established binding warnings remain. Initial focused Python: **115 passed**;
all **17 new pending retained-history tests** pass including sibling/recovery.

The native execution regression runs accepted-state and recorded-time acquisition
plus diffs after staged creation/property edits, then accepts once and checks the
older snapshot again. Python covers directed/undirected sparse scopes, local/outer
programs, loops, exact original versus revised endpoints, provisional exclusion,
working/historical values, late missing/future/removed/out-of-scope targets,
rollback without ID consumption, explicit sibling reads without branch switching
and retained archive recovery. Existing temporal provenance, property-only and
fixed child suites remain maintained. An initial branch assertion guessed a
nonexistent current_branch property; it now uses supported branches()/is_current.

Full maintained Python: **3,013 passed / one skipped / 14 established warnings**;
all **281 executable documentation examples** pass. Black/isort/pre-commit remain
unavailable. No external packages were installed. Unrelated owner files untouched.

## Existing roadmap and next work

Advances 09.4b.2, 30.1, AR7.8 and temporal/API conformance under 01.2/19.2.
The inventory remains **215 checked / 53 unchecked**, including **44 unchecked
release rows**. No added rows or broad parent closure.

Four selected lifecycle/window helpers retain their pending execution guards.
Their existing retained-state semantics can be applied to surviving accepted IDs;
provisional-node treatment is the outstanding owner question. General topology
loops, classic wrapper adjacency migration, alternate plans/DSL lowering and the
remaining service/scale/release requirements remain separate work.
