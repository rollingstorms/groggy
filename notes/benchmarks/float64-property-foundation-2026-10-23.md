# f64 property capture receipts — 2026-10-23

Locked offline release Criterion; idle retained run, 200 ms warmup, 1 second
measurement, 10 samples. Each fixture is a one-node graph with one explicitly
tagged f32/f64 vector property (plus normal node metadata). Setup excluded;
current-state capture and result destruction included. No mutation, codec I/O,
GIL/FFI or RSS measurement.

| Vector elements | f32 capture | f64 capture | f32 raw payload | f64 raw payload |
| --- | ---: | ---: | ---: | ---: |
| 1,000 | 0.6741 µs | 0.7187 µs | 4,000 bytes | 8,000 bytes |
| 10,000 | 1.6570 µs | 2.5772 µs | 40,000 bytes | 80,000 bytes |

Values are generated at their explicitly chosen width; this measures capture
cost and payload width, not equal-precision alternatives or a universal speedup.
Capture copies property payloads once; sharing the resulting pin remains separate.
The raw vector payload doubles. Whole-graph memory, allocation overhead, future
consumer cost and production latency are not established by this fixture.
Receipt: `benches/float64_property_foundation_2026-10-23.json`; fixture:
`benches/read_state.rs`.
