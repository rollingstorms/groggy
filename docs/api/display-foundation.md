# Native display ownership and compatibility

`groggy::display::render` owns native typed display windows/schema, configuration,
themes and Unicode/HTML rendering. Table, matrix, graph and streaming adapters
refer to this owner directly. `groggy::viz::display` (including its submodules)
and the existing `groggy::core` display exports remain compatibility aliases to
the same types. No wrapper engine, copied theme catalog or second native typed
renderer is created by an old import.

The root `groggy::display::DisplayConfig` and JSON-dictionary formatting functions
retain their existing compatibility input contract. Its total-width/color options
and the typed renderer's cell-width/theme options remain distinct APIs. Standalone
historical Python formatter classes also retain their existing presentation and
defaults. Their remaining migration is tracked under AR7.6; this consolidation
does not silently replace those layouts or retire public names.

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
The broader display compatibility migration and release gates remain open.
