# Canonical entity handle ownership — 2026-10-22

Checklist AR7.4 moves entity handle ownership under `entities`. `Node` and `Edge`
remain the canonical ID/graph-reference representations. Legacy EntityNode and
EntityEdge now contain those handles; storage/node and storage/edge are import
facades with no independent state or implementation. Actual internal callers use
`entities::EntityNode`; historical public storage imports remain supported.

## Compatibility that matters

Node::new/Edge::new keep checked active-ID construction. EntityNode::new/
EntityEdge::new keep their infallible unchecked signatures; subsequent operations
still enforce the existing live-state rules. A crate-private unchecked canonical
constructor exists only to implement that compatibility policy.

Adapters preserve historical specialized trait overrides, summaries and related
entity construction. They do not silently erase neighborhood/meta reconstruction
behavior or change the pending metadata-key migration. Moving ownership does not
settle A01/A03 or redesign liveness, topology, property writes or history.

All handles still target the same Rc<RefCell<Graph>>; no graph/value copy or extra
reference allocation is introduced by the wrapper. Mutations use the existing
GraphEntity/graph setters and change tracking. Deactivation blocks live attribute/
endpoint access, while retained pool/history referents remain separate and
inspectable through their existing APIs.

## Exit checks

Native regressions cover legacy/current type identity, shared graph ownership,
checked-versus-unchecked absent IDs, cross-adapter mutations, branch restoration,
inactive reads and retained endpoints. Existing handle/trait/meta/subgraph and
Python live/history/topology suites must pass after rebuilding the extension.
This ownership consolidation does not claim graph views are now state-pinned;
AR7.5/AR4 are separate work.

Executed: full locked offline native suites; strict all-target core Clippy
(clean); rebuilt extension and active/retained/batch/bulk/identity/table Python
gate: **109 passed, 1 existing skip**. Constructor/mutation/restoration fixtures
pass through the historical and new import paths.
