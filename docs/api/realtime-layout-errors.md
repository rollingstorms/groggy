# Realtime flat-layout errors

When honeycomb layout requests `flat_embed`, the engine now calls the existing
owned `compute_flat_embedding_matrix` helper. Its graph/row validation, solver
errors, matrix construction errors and ID labels pass through the realtime path.
A preprocessing error returns `Err` instead of projecting the original feature
matrix and reporting success. The engine no longer rebuilds a label-free matrix
from returned position pairs.

## Parameters

`flat_embed` accepts `true`, `false`, `1`, or `0`, ignoring case and surrounding
whitespace. An absent flag disables preprocessing. Any other supplied value errors.
When enabled, all native controls are available as strings:

- `flat_embed.iterations` and `flat_embed.seed`: unsigned integers.
- `flat_embed.learning_rate`, `flat_embed.repulsion_power`,
  `flat_embed.radius_clip`: finite positive numbers.
- `flat_embed.edge_cohesion_weight`, `flat_embed.repulsion_weight`,
  `flat_embed.spread_weight`: finite nonnegative numbers.

Absent controls use `FlatEmbedConfig` defaults. Malformed supplied controls return
errors identifying the parameter and supplied text; the native solver checks
numeric domains. Disabled preprocessing ignores its unused numeric controls.
This preserves the default nonflat path and makes bad requested controls explicit.

Unknown names in `EngineUpdate::LayoutChanged` also return an error from
`RealTimeVizEngine::apply`; they no longer warn and broadcast an applied update.
Existing supported layout aliases remain.

For `ChangeLayout`, the WebSocket bridge now waits for the engine result rather
than sending an immediate success acknowledgement. Both server run paths send the
requesting client `ControlAck { success: false, message: <native error> }` on
failure, including an unavailable engine, and acknowledge success only after the
layout applies. The browser logs rejected acknowledgements as errors. Other
control-message acknowledgement behavior is unchanged.

## State and scope

Layout-change requests now prepare their embedding, projection, derived controls,
finite coordinates, and publication timestamp before updating engine state. A
failed request leaves the prior controller/view and drag policies, configuration,
layout/parameter caches, cached embedding, animation, positions, timing metadata,
pending parameter deltas and synchronization queues unchanged. A freshly computed
embedding is cached only after the whole layout succeeds. Failures emit no engine
layout/view/position success updates; the server still returns the failure ack.

On success, publish the accepted settings/coordinates, reset interpolation, switch
the controller and emit one envelope containing matching layout parameters,
positions and any controller view. Retain preexisting pending parameter deltas in
that envelope. Validate centering output as well as raw coordinates before
publication. Honeycomb derived cell size updates the effective configuration and
accepted parameter cache together. Empty incoming parameters continue to restore
the last accepted cache for that layout.

Only coalesced position deltas are superseded by a successful layout. Attribute
and structural updates remain queued; the previous implementation discarded
coalesced property updates along with motion. Failed requests discard neither.

This guarantee covers recoverable errors in layout-change requests. Other control commands, coordinate identity across all layouts, crash recovery,
concurrent-reader isolation, visual error panels and network delivery/retry
semantics still need their own audit. Embedding/projection configuration updates
are covered by the additional scope below. Transport failure
after successful application does not undo an accepted layout.

## Evidence

Native realtime-engine tests compare the preprocessing matrix to the shared native
solver with every control supplied and with defaults, including sparse/permuted
IDs and ownership. Failure fixtures cover parsing, numeric domains, row-count and
ID errors, and enabled/disabled behavior. Async tests check failed recomputation's
unchanged coordinates/timestamp and lack of a success broadcast, plus public
unknown-layout rejection without a success update. See tests in
`src/viz/realtime/engine.rs` and [flat layout](flat-embedding.md).

Server/bridge unit tests check that a queued layout emits no premature success,
that unavailable engines reject, and that completion acknowledgements reflect
valid layout, unknown-name and flat-preprocessing failures. These are in-process
engine/channel checks; this slice does not certify browser rendering or real
network disconnect/retry behavior.

Layout-publication tests snapshot configuration, controllers/views/policies,
positions, cached embeddings/parameters, animation/timing, pending settings and
queues before cached/uncached preprocessing failures, late projection errors,
embedding failures and nonfinite candidates. Successful retry checks a single
coherent envelope, accepted-cache restoration, discarded motion and retained
property/structural updates. No full control-command or concurrent transaction
certification is implied.


## Embedding and projection configuration updates

Typed embedding and projection controls now prepare candidate configuration,
embedding/coordinates, derived layout settings and timestamp before publication.
Embedding changes compute a fresh embedding; projection changes reuse an existing
one when available. A missing embedding is cached only after projection succeeds.
Existing explicit layout parameters retain precedence over projection defaults.
Recoverable failures leave configuration, caches, animation, controllers/views,
coordinates/timing, pending settings and queues unchanged and emit no success.

Accepted changes publish a coherent envelope containing projection configuration,
layout parameters/positions and, for embedding changes, embedding configuration.
The controller remains selected; fresh embeddings update its dimension-dependent
activation after successful publication. Successful recomputation supersedes only
coalesced motion and preserves property/structural updates.

String embedding requests reject unknown methods rather than substituting spectral.
Legacy `energy`/`force_directed` names and native `_nd` aliases are accepted alongside
`spectral` and `random`. They execute immediately, rather than entering a structural
update queue and being acknowledged before execution. Both server paths return
actual embedding completion/failure acknowledgements. Nonempty string-control
parameter maps now reject explicitly; use typed native configuration for those
controls instead of relying on previously ignored values. The bridge defers
embedding acknowledgements just as it does layout acknowledgements.

`UpdateQualitySettings` now returns `NotImplemented` without changing configuration.
The former quality-only recomputation was a no-op. This repair makes that boundary
explicit; it does not implement or certify the quality evaluator/optimizer. Empty
graphs requiring a new embedding reject via the native engine instead of treating
old display coordinates as the requested embedding.

Native tests compare full prior-state signatures for cached/uncached projection,
embedding and unsupported quality failures, verify coherent successful embedding
and projection updates, and exercise immediate string request validation and
server/bridge acknowledgements. Other realtime controls and broader visualization
correctness remain open.
