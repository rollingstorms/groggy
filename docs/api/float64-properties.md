# f64 property representation: foundation

This is the first implementation slice of checklist 07.3a. Native properties now
have distinct `AttrValue::Float64(f64)` and `Float64Vec(Vec<f64>)` variants. The
existing `Float(f32)`, `FloatVec(Vec<f32>)` and compressed f32 variants retain
their types. New variants are appended to the public enum, retaining old variant
indices. Width changes remain visible as typed property changes.

```python
import groggy as gr

precise = 1 + 2**-40
graph = gr.Graph()
node = graph.add_node(
    scalar=gr.AttrValue.float64(precise),
    embedding=gr.AttrValue.float64_vec([precise, 1e300]),
)
assert graph.get_node_attr(node, "scalar") == precise
assert graph.get_node_attr(node, "embedding") == [precise, 1e300]
```

The explicit Python constructors retain the original f64 values. Historical
snapshots report `float64`/`float64_vec` type names. The legacy `float_vec`
constructor continues to request f32 explicitly. Ordinary Python float/list
inference has **not switched yet**: accepted f64 defaults follow completion of
consumer migration. Do not infer complete numerical/legacy algorithm support
from the existence of these tags.

## Lossless value lifecycle

Tagged export and checked journal codecs store f64 scalar/vector bit patterns as
u64 payloads, preserving precision, finite range, signed zero, infinities and NaN
payloads through the tested round trips. Existing f32 tags/payloads and legacy
unversioned decoding are unchanged; previously lost precision cannot be restored.
The versioned export/journal envelopes remain unchanged; old readers that do not
know the new value tags reject them. No silent downgrade is provided.

Property identity groups signed zeros and NaN encodings within each floating
width, matching existing f32 identity rules. f32/f64 tags remain distinct;
f64 vectors are not compressed by the f32-only optimization path. Explicit
Float64-to-Float32 numeric conversion rejects finite range overflow; requesting
the narrower type can still round precision. Existing typed f32 getters remain
f32-only. `as_f64` widens either scalar floating width; `as_float64_vec` borrows
an explicitly stored f64 vector.

Schema families `float`/`float_vec` accept both widths. Explicit `float32`,
`float64`, `float32_vec`, `float64_vec` constraints select storage width. Shared
numeric input inference, checked graph-matrix conversion, native query numeric
comparison, captured weights/heuristics and PageRank personalization recognize
f64. Algorithm parameters preserve their f64 values when converted to properties.

## Remaining default-migration gate

Before changing ordinary Python float defaults, migrate the remaining array/table
predicate and arithmetic paths, output casts, legacy weight accessors and algorithm
persistence, display/type inference, text/CSV conversion and vector consumers.
Verify dtype, mixed precision, old-format reads, schema, history, full numerical
and Python compatibility at the final switch. The parent 07.3a remains open.

The scalar payload is 8 bytes versus 4 for f32; raw f64 vector payload is twice
as wide. This does not imply the whole graph's memory doubles. No universal
throughput/memory SLA is claimed; capture timings are in the benchmark receipt.
