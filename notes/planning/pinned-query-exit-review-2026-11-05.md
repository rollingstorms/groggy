# Pinned query exit review — 2026-11-05

Closes existing checklist item **21.2**. No new rows or query language added.

The remaining exit criterion was the legacy filter migration. Commit `22a05d11`
completed that migration: default filters return immutable source-bound selections;
querying or deriving another selection preserves the captured values, membership,
topology and source clocks after owner changes. Explicit connected compatibility
reads current owner state. Graph.query uses GraphView rather than a placeholder.
Historical effective and optional recorded cutoffs survive selections; explicit
indexes preserve scan ordering, exact scalar semantics and pagination. Query
capture does not publish changes or implicitly build an index.

Verification on the completed migration:

- `python -m pytest tests/test_query_capture.py tests/test_default_immutable_views.py -q`: **26 passed**.
- Existing query tests cover effective/recorded cutoffs, durable close/reopen,
  thread transfer, index parity, validation and owner mutation independence.
- Existing default-view tests cover filter/derivation pins, endpoint/property and
  membership changes, weak capture cache lifetime and explicit connected reads.
- The migration receipt records full native, rebuilt Python and documentation gates.

Limits remain explicit: finite predicates/patterns are the supported language;
unsupported domains fail. Full first/invalidated capture and property ordering
are not constant-cost operations. Production scale certification belongs to 32.

Fixed accounting: **215 checked / 53 open / 268 total**. Compound topology and
pipeline effects remain open in 09.4b.2/AR3/30.1/AR7.8.
