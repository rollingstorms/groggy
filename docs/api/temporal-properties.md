# Temporal properties

Groggy stores three distinct temporal property types:

| Property type | Python value | Native Rust type | Meaning |
| --- | --- | --- | --- |
| `utc_instant` | Aware `datetime` | `UtcInstant` | A point on the UTC timeline |
| `date` | `datetime.date` | `CalendarDate` | A Gregorian calendar date without time or timezone |
| `duration` | `datetime.timedelta` | `FixedDuration` | Fixed elapsed time; one day is exactly 86400 seconds |

```python
from datetime import date, datetime, timedelta, timezone
from groggy import AttrValue, Graph

local = datetime(2024, 2, 29, 12, 0, 0, 123456,
                 tzinfo=timezone(timedelta(hours=5, minutes=30)))
graph = Graph()
node = graph.add_node(when=local, day=date(2024, 2, 29),
                      elapsed=timedelta(microseconds=-1))
assert graph.get_node_attr(node, "when") == local.astimezone(timezone.utc)
assert graph.get_node_attr(node, "when").tzinfo is timezone.utc
assert graph.get_node_attr(node, "elapsed") == timedelta(microseconds=-1)
assert AttrValue(local).type_name == "utc_instant"
assert AttrValue("2024-02-29").type_name == "text"
checkpoint = graph.commit("temporal example", "docs")
frozen = graph.snapshot_at_commit(checkpoint)
graph.set_node_attr(node, "day", date(2024, 3, 1))
assert frozen.node_attr(node, "day").value == date(2024, 2, 29)
```

Direct and bulk node/edge setters, checked writers, table cells and snapshots
share this translation. Explicit constructors `AttrValue.utc_instant(datetime)`,
`AttrValue.date(date)` and `AttrValue.duration(timedelta)` check the input kind.
Dates are distinguished from datetime values even though Python datetime is a
date subclass. Naive datetimes reject rather than assuming a local timezone.
Aware inputs normalize to UTC, honoring their offset and DST fold; the source
zone name and local clock representation are not retained.

## Exact domain and comparisons

Resolution is one microsecond throughout Rust, Python and storage. Conversion
uses integer components, without floating-point epoch arithmetic. UTC instant
and calendar date years range from 1 to 9999 inclusive. Leap seconds are not
supported. UTC normalization outside this range rejects. Rust constructors
validate calendar dates and canonical fractions; invalid components never enter
an attribute value, including through deserialization.

Durations use canonical signed days in -999999999..=999999999, nonnegative
seconds below 86400 and nonnegative microseconds below 1000000. Thus -1us is
`days=-1, seconds=86399, microseconds=999999`. Months/years and civil time periods
are separate future concepts. Native constructors accept canonical components;
Python timedelta performs its usual normalization before translation.

Equality, hashing, grouping and sorting retain the temporal kind. Chronological
comparison works within each kind through typed graph filters and array scalar
operators. Temporal values do not act as integers, numeric measurements or text;
array ordering against another domain rejects, while graph ordering filters do
not match it. String query literals remain strings; use typed filter objects for
temporal comparisons. Temporal arithmetic and specialized temporal vector types
are not part of this property-type item.

Schema types `utc_instant`, `date` and `duration` validate the respective kind;
they do not accept strings or numeric epoch guesses. Plain table CSV/JSON and
visualization display produce readable text, without temporal type recovery on
reimport. Use tagged graph storage for lossless round trips.

## Versioned storage

Graph export version 2 and the existing checked journal envelope carry new
`utc_instant`, `date`, and `duration` tags. Each temporal payload has `version: 1`:

- Instant: floor Unix `seconds` plus fractional `microseconds`.
- Date: integer `year`, `month`, and `day`.
- Duration: normalized `days`, `seconds`, and `microseconds`.

Unknown versions, unknown fields, missing fields and invalid components reject.
Existing property tags and their encodings are unchanged. Readers without these
new tags reject them instead of silently reducing them to text.

A temporal property is data. Setting it does not set a mutation's effective time,
create an extra recorded-by clock, or schedule graph reconstruction. Those
publication/history contracts remain separate roadmap items.

## Python build profiles

The extension build script follows PyO3's resolved configuration and supplies
extension-module linker arguments. Normal CPython builds keep direct datetime C
access. Limited-ABI builds (including the configured forward-compatibility mode)
marshal through cached standard-library datetime types and their base field
descriptors. Both normalize aware instants to UTC with integer components, retain
date/duration bounds, reject naive datetimes and ignore subclass field overrides
just as direct C access does. No float epoch conversion is introduced.

Both profiles have been built and exercised on CPython 3.9.6. This fixes the
previous limited-ABI datetime import failure; certification of other Python
versions, PyPy and platforms remains part of the compatibility matrix.
