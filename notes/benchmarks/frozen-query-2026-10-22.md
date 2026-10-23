# Shared query timing receipts — 2026-10-22

Locked offline release Criterion runner, 200 ms warmup, 1 second measurement,
10 samples. No competing builds/tests during the retained run. Native ring
fixtures with scalar value IDs: attribute predicate selects value > count/2;
incident degree range 2..2 selects the whole ring. Setup/capture excluded, returned
selection construction and drop included. No previous-engine baseline, FFI,
allocation/RSS or service-budget claim.

| Nodes / edges | Live attribute query | Frozen attribute query | Live degree query | Frozen degree query |
| --- | ---: | ---: | ---: | ---: |
| 1,000 / 1,000 | 0.254 ms | 0.274 ms | 0.394 ms | 0.442 ms |
| 10,000 / 10,000 | 3.939 ms | 5.708 ms | 4.115 ms | 5.770 ms |

Both paths use the same predicate executor but different read/output ownership.
Live Subgraph validates current selection, lazily builds incident topology for
degree predicates and constructs its legacy result/metadata. Frozen GraphView
reads indexed captured incidence and constructs immutable membership sharing the
pin. Attribute queries also include the established parallel matching threshold
and result edge induction. These fixtures do not isolate predicate throughput or
prove a speedup against the previous implementation. Frozen capture cost remains
separate; neither arbitrary filtering nor selection construction is constant time.

Receipt: `benches/frozen_query_2026-10-22.json`; fixture: `benches/read_state.rs`.
Filter: `read_state/(live_attribute_query|frozen_attribute_query|live_degree_query|frozen_degree_query)`.
