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

A flat-preprocessing failure stops projection recomputation before replacing
positions, updating projection timing/last-update metadata or broadcasting its
success envelope. Existing coordinates remain available. This does not establish
transactional rollback for all control commands: layout-change controller/config
bookkeeping occurs before recomputation and needs a separate atomicity audit.
Other layout controls, centering, visual error panels, recovery policies and
display-coordinate ownership are outside this repair.

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
