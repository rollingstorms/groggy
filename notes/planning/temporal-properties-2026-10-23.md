# 07.3b — Native temporal property types

Implements accepted workbook A06.1: separate UTC instant, calendar date and fixed
elapsed duration properties. Existing integer timestamp properties and strings
are not reinterpreted. Zoned/local calendar types remain follow-on work.

## Implementation choices

- Exact microseconds match Python datetime/timedelta without a second precision
  domain or float epoch conversion. Shared instant/date year range: 1..=9999.
- UTC instants use floor Unix seconds and a nonnegative microsecond fraction;
  aware Python datetime normalizes to UTC with its actual offset/fold. Naive
  datetime rejects. Zone identity/local representation is not retained.
- Gregorian dates validate leap years. Durations retain Python's full normalized
  signed-day range with nonnegative seconds/microseconds; no calendar months.
- Native wrappers have private fields and validated constructors/deserializers.
  Temporal payload version 1 is independent of the unchanged export version 2
  and journal envelope. New enum variants append after existing variants.
- Distinct property identity/hash/order, same-domain chronological predicates,
  dtype/profiling, typed schemas, table cells, direct/bulk setters, retained
  snapshots, tagged exports and checked journal recovery share the native types.
- FFI only marshals datetime components, invokes timezone conversion and maps
  native validation errors. Formatting adapters show readable text. Plain table
  CSV/JSON cannot recover these tags; graph storage preserves them.

These are temporal **property values**, not mutation effective-time inputs or a
publication-policy migration. Temporal arithmetic, native temporal vector types,
and local/timezone calendar values are outside this item.

## Verification

Shared fixed fixture: `tests/data/temporal_properties_v1.json`, read by Rust and
Python. Native tests cover canonical constructor limits, invalid calendar/clock
components, property identity, numeric exclusion, chronological predicates,
version rejection, schemas, retained reads, export and journal recovery. Python
coverage includes min/max domains, pre-epoch -1us, negative durations, fractional
offsets, DST folds, explicit constructors, no string inference, bulk setters and
table projections.

- Focused native temporal property suite: 4 passed.
- Full `cargo test --offline --locked --quiet`: 900 passed, 52 ignored across
  unit, integration and documentation suites.
- Rebuilt in-tree Python extension; root API baseline excluding the owner's
  independent algorithm benchmark: 991 passed, 1 existing skip. The old
  `tests/modules/` survey scripts are not included in this baseline.
- 23 documentation examples passed against that extension.
- `cargo clippy --offline --locked --all-targets -- -D warnings`: passed.

No external dependencies were added. Black is unavailable in the existing
Python environment; Rust changes were formatted and whitespace checked. Three
existing unrelated FFI warnings remain. No new performance SLA is claimed.
