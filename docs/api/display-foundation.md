# Native display ownership and compatibility

`groggy::display::render` owns native typed display windows/schema, configuration,
themes and Unicode/HTML rendering. Table, matrix, graph and streaming adapters
refer to this owner directly. `groggy::viz::display` (including its submodules)
and the existing `groggy::core` display exports remain compatibility aliases to
the same types. No wrapper engine, copied theme catalog or second native typed
renderer is created by an old import.

The root `groggy::display::DisplayConfig` and JSON-dictionary formatting functions
retain their existing compatibility input contract. Its total-width/color options
and the typed renderer's cell-width/theme options remain distinct APIs. The historical Python formatter classes now use native
`display::render::interactive` layouts. They retain their public imports, class
names and normal presentation defaults. Python marshals scalar spellings and
terminal-color capability; Rust owns selection plans, widths, clipping, alignment,
borders and ANSI layout. Python's arbitrary-size integers, object `str`, `g`/`f`
float formatting, booleans, signed zero and insertion-ordered summaries are kept
at that language presentation boundary. The root native JSON formatter retains
its separate JSON scalar spellings and legacy width policy.

Shared native text clipping counts Unicode scalar characters and never cuts an
invalid UTF-8 boundary. A zero budget returns empty text. It examines only the
visible budget plus one character; it does not first count a whole large property.
This is a character budget, not a grapheme or terminal-glyph width guarantee.
Compact rendering retains its existing tiny-width prefix policy; HTML JSON and
legacy text clipping retain an ellipsis when truncated. Zero row/column budgets
in legacy truncation utilities no longer underflow.

HTML JSON values are escaped after safe clipping; full tooltip values are also
escaped. HTML integer grouping accepts the complete signed 64-bit domain. Display
does not change stored properties or publish a delta.

```python
from groggy import Graph
display_graph = Graph()
display_graph.add_node(integer=-(2**63), message="猫<&")
display_table = display_graph.read_view().nodes_table()
head = display_graph.current_state_id
display_html = display_table._repr_html_()
assert "-9,223,372,036,854,775,808" in display_html
assert "猫&lt;&amp;" in display_html
assert display_graph.current_state_id == head
display_graph.set_node_attr(0, "message", "changed")
assert "猫&lt;&amp;" in display_table._repr_html_()
```

Native ownership, compatibility and measured boundaries are recorded in
`notes/planning/display-foundation-2026-11-11.md`. The ASCII render corpus checks
the preserved layouts; native regressions cover every UTF-8 cutoff in a JSON
fixture, zero/maximum budgets, large text prefixes and minimum signed integers.
The adapter consolidation and its bounded compatibility/performance gate are
complete under AR7.6; whole-engine scale and release qualification remain their
separate checklist items. See `notes/planning/display-closeout-2026-11-18.md`.


## Dictionary display configuration

`groggy.display.configure_display` supplies overrides to subsequently created
`ArrayDisplayFormatter`, `MatrixDisplayFormatter` and `TableDisplayFormatter`
instances, including instances constructed by `format_array/matrix/table`.
Explicit per-instance arguments win. An existing formatter keeps its current
settings. Configuration is validated as nonnegative integer counts before any
new global defaults are published; zero is meaningful. `None` requests the current
default. Calling the configuration function with no arguments changes nothing.
Previously it only changed unused constants; making those overrides effective is
an intentional bug fix.

Without overrides the historical defaults remain array rows=10, precision=4,
width=60; table/matrix rows=10, columns=20, precision=2, width=120. Total width
constrains table column allocation, with the historical three-character minimum;
it is **not a hard line-width limit**. Array/matrix `max_width` remains a retained
compatibility setting without a new width-limiting interpretation. Matrix cells
keep their existing spellings even when wider than the decorative column cap.
These settings do not override the independent typed table/HTML renderer's
configuration or themes.

```python
from groggy.display import ArrayDisplayFormatter, format_array, format_table

array_formatter = ArrayDisplayFormatter(max_rows=3)
text = array_formatter.format({"data": list(range(10)), "shape": (10,)})
assert "shape: (10,)" in text
assert "…" in text
small = format_array({"data": list(range(10)), "shape": (10,)}, max_rows=1)
assert "│ 0 │" in small  # the displayed prefix is source row zero
literal = format_array({"data": [["…"], "…"], "shape": (2,)})
assert "['…']" in literal  # a literal cell is never an omission sentinel
summary = format_table({"columns": ["a"], "dtypes": {}, "data": [[True]],
                        "shape": (1, 1)})
assert "True" in summary
```

## Compatibility corrections and boundaries

- Omission markers have explicit native positions; literal `…`, `⋯` and `['…']`
  values retain their original indices. Prefix budgets one/two also retain the
  correct source indices. Tail column types follow the same selected columns as
  their names. Ragged compatibility rows use original column positions, with
  absent table cells empty; no value is relabeled as another column.
- Historical date text has one ellipsis within its ten-character budget. Native
  dictionary array/matrix string clipping is UTF-8 safe; native matrix rendering
  obeys `max_rows`. Native compatibility selection helpers now show prefixes for
  budgets one/two, matching the shared plan; their omission-slot-free contract
  remains distinct from interactive layouts that display markers.
- Native dictionary translation preserves booleans before integer conversion,
  maps nonfinite floats to the existing null display instead of inventing zero,
  and preserves out-of-range Python integer spelling instead of narrowing through
  f64. Ordinary Python adapters retain Python integer formatting directly.
- Python table formatting reads/formats selected visible cells only. Hidden
  objects' formatting callbacks are not invoked. The lower-level native JSON
  dictionary API still owns/materializes its supplied JSON data; this is not a
  zero-copy or whole-input allocation guarantee.
- Native workers receive owned text, with no Python objects, and release the GIL
  while composing complete layouts. Python scalar conversion and generic-object
  projection run with the GIL. Text must be UTF-8 representable: lone Python
  surrogate code points reject with `UnicodeEncodeError`, without changing input.
  Character budgets count Unicode scalar values, not grapheme/terminal width.
- Existing imports and public class/function names remain; obsolete private
  Python border/row builders are removed. `unicode_chars` keeps presentation
  constants and terminal detection as a thin Python adapter. Existing native
  `core`/`viz::display` type aliases retain one typed renderer owner.
