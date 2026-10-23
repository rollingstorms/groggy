# Typed property materialization — 2026-10-23

Native locked offline release Criterion; 200 ms warmup, 1 s measurement, 10
samples. Source values 1+i*2^-40. Input setup excluded; result allocation and
destruction included. No concurrent builds/tests. These are current measurements,
not a comparison against a released revision, FFI latency, RSS or scale guarantee.

| Values | Median |
| ---: | ---: |
| 1,000 | 10.467 µs |
| 10,000 | 94.417 µs |

The scalar converter streams directly into the property vector, eliminating
the prior intermediate NumericInput vector. This allocation change follows from
the implementation; no measured RSS or universal speedup claim is made. Matrix
display shares this converter with checked reads and explicit source widths.

Raw receipts: `benches/typed_property_materialization_2026-10-23.json`.
