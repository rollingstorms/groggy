# Visualization Issues Analysis

## Current Architecture Overview

The visualization system has several key components:

### Core Components
1. **VizAccessor** (`python-groggy/src/ffi/viz_accessor.rs`)
   - Python `.viz` property accessor
   - Provides `.show()`, `.server()`, `.update()` methods
   - Manages server lifecycle and parameters

2. **VizConfig** (`src/viz/realtime/mod.rs`)
   - Comprehensive styling configuration
   - Supports `VizParameter<T>` enum with variants:
     - `Array(Vec<T>)` - Array of values
     - `Column(String)` - Column name reference
     - `Value(T)` - Single value for all elements
     - `None` - Use defaults

3. **Server System** (`src/viz/realtime/server/`)
   - WebSocket-based real-time visualization
   - Supports parameter updates via control messages
   - Server reuse and lifecycle management

4. **Frontend** (`web/`)
   - Three.js/D3.js visualization renderer
   - Handles layout algorithms (honeycomb, force-directed, circular, grid)

## Issues to Address

### Issue 1: Viz update server parameters intuitively and with .show()

**Current State:**
- `.show()` creates/reuses server and displays visualization
- `.update()` sends control messages to existing server
- Parameters passed as kwargs to both methods

**Problem:**
- Users might expect `.show()` to also update existing server parameters
- Parameter updates are split between initial `.show()` and subsequent `.update()`

**Options:**

**Option A: Make .show() also update (Recommended)**
```python
# Initial visualization
g.viz.show(node_color='red', layout='force')

# Later, change parameters - .show() updates existing server
g.viz.show(node_color='blue')  # Updates server if exists, creates if not
```
**Pros:** Intuitive, single method for all cases
**Cons:** Name ".show()" might imply display only, not update

**Option B: Merge .update() functionality into .show() with flag**
```python
g.viz.show(node_color='blue', update_only=True)  # Don't create new server
```
**Pros:** Explicit control
**Cons:** More complex API

**Option C: Smart detection in .show()**
```python
# Auto-detects if server exists and updates parameters
g.viz.show(node_color='blue')  # Creates OR updates
```
**Pros:** Clean API, smart behavior
**Cons:** Magic behavior might be confusing

**Recommendation:** **Option C** - Make `.show()` intelligent:
- If server exists for this data source → send update
- If no server exists → create new server
- Keep `.update()` for explicit update-only cases
- Keep `.server()` for explicit server-only mode

---

### Issue 2: Viz node color - discrete or continuous

**Current State:**
```rust
pub node_color: VizParameter<String>,  // Only supports string colors
```

**Problem:**
- Numeric column values need automatic color mapping
- No distinction between discrete (categorical) and continuous (numeric) coloring
- Color scale type exists but not integrated with parameter parsing

**Options:**

**Option A: Auto-detect from data type**
```python
# String/categorical → discrete palette
g.viz.show(node_color='category')  # Auto discrete

# Numeric → continuous gradient
g.viz.show(node_color='score')  # Auto continuous
```
**Pros:** Automatic, no user configuration needed
**Cons:** User might want different behavior

**Option B: Explicit color scale parameter**
```python
g.viz.show(
    node_color='score',
    color_scale='continuous',  # or 'discrete'
    color_palette=['red', 'yellow', 'green']
)
```
**Pros:** Explicit control
**Cons:** More verbose

**Option C: Smart defaults with override**
```python
# Auto-detect from column dtype
g.viz.show(node_color='score')  # Numeric → continuous

# Override with scale parameter
g.viz.show(node_color='score', color_scale='discrete')
```
**Pros:** Best of both worlds
**Cons:** Need robust dtype detection

**Option D: Separate parameters**
```python
# Discrete coloring
g.viz.show(node_color_discrete='category')

# Continuous coloring
g.viz.show(node_color_continuous='score', 
           color_range=['#ff0000', '#00ff00'])
```
**Pros:** Very explicit
**Cons:** API complexity

**Recommendation:** **Option C** - Smart defaults with override:
1. Auto-detect numeric vs categorical from column dtype
2. Allow `color_scale='discrete'|'continuous'` to override
3. Support `color_palette` for discrete, `color_range` for continuous
4. Extend `VizParameter<T>` to support numeric types for auto-mapping

**Implementation:**
```rust
// Extend VizParameter to handle numeric mappings
pub enum ColorParameter {
    Direct(VizParameter<String>),  // Direct colors
    Mapped {
        column: String,
        scale: ColorScale,
        palette: Option<Vec<String>>,
    }
}

pub enum ColorScale {
    Discrete,
    Continuous,
    Auto,  // Detect from data
}
```

---

### Issue 3: Subgraph viz show → just the subgraph? On update

**Current State:**
- `SubgraphArray` has `.viz` accessor
- Unclear what gets visualized: just subgraph nodes/edges or full parent graph?
- Updates might affect parent graph visualization

**Problem:**
```python
subgraphs = g.components()
sg = subgraphs[0]

# What gets visualized?
sg.viz.show()  # Just nodes/edges in sg? Or full graph with sg highlighted?

# What gets updated?
sg.viz.show(node_color='red')  # Just sg nodes? Or all nodes?
```

**Options:**

**Option A: Subgraph-only visualization**
```python
# Shows ONLY the nodes/edges in the subgraph
sg.viz.show()  # Isolated view
```
**Pros:** Clear, focused view
**Cons:** Loses context of parent graph

**Option B: Highlight in full graph**
```python
# Shows full graph but highlights subgraph
sg.viz.show()  # Full graph with sg highlighted
sg.viz.show(highlight_only=True)  # Explicit
```
**Pros:** Maintains context
**Cons:** Might be unexpected

**Option C: Configurable behavior**
```python
sg.viz.show(mode='isolated')  # Only subgraph
sg.viz.show(mode='highlighted')  # Full graph with highlighting
sg.viz.show(mode='context')  # Full graph with context dimming
```
**Pros:** Flexible
**Cons:** More API surface

**Option D: Different methods**
```python
sg.viz.show_isolated()  # Only subgraph
sg.viz.show_in_context()  # Full graph with highlighting
```
**Pros:** Explicit naming
**Cons:** More methods

**Recommendation:** **Option C with smart default**:
- **Default behavior:** Show isolated subgraph (cleaner, more predictable)
- **Override with:** `context='full'` to show full graph with subgraph highlighted
- **Update behavior:** Updates only affect the subgraph's data source

```python
# Default: isolated view
sg.viz.show()  

# With context
sg.viz.show(context='full', highlight=True)

# Update only affects subgraph
sg.viz.show(node_color='red')  # Colors only sg nodes
```

---

### Issue 4: Quick static viz like viz.color() or something

**Current State:**
- All viz methods create servers and interactive views
- No quick matplotlib-style static visualization

**Problem:**
- Sometimes users just want a quick PNG/SVG output
- No simple one-liner for "show me the graph"

**Options:**

**Option A: Add static methods to VizAccessor**
```python
# Quick static plot
g.viz.plot()  # Opens in matplotlib/PIL
g.viz.plot(save='graph.png')

# Still have interactive
g.viz.show()  # Interactive server
```
**Pros:** Clean separation
**Cons:** Requires matplotlib dependency

**Option B: Add rendering backend parameter**
```python
g.viz.show(backend='static')  # Static image
g.viz.show(backend='interactive')  # Server (default)
```
**Pros:** Unified API
**Cons:** Different backends have different capabilities

**Option C: Separate quick-viz methods**
```python
# Quick colorized static view
g.viz.color('degree')  # Color by degree, display static

# Quick layout static view
g.viz.layout('force')  # Force layout, display static

# Full interactive
g.viz.show()
```
**Pros:** Descriptive method names
**Cons:** Limited functionality per method

**Option D: Render method on existing server**
```python
# Interactive first
g.viz.show()

# Capture current state as static
g.viz.render('snapshot.png')
g.viz.render(format='svg')
```
**Pros:** Captures live state
**Cons:** Requires server running first

**Recommendation:** **Option A + Option D combined**:

Add both quick static plots AND rendering from interactive:

```python
# Quick static (no server needed)
g.viz.plot()  # Opens matplotlib
g.viz.plot(save='graph.png')
g.viz.plot(node_color='degree', layout='force')

# Interactive server
g.viz.show()

# Render current server state
g.viz.render('snapshot.png')  # Captures current viz state
```

**Implementation considerations:**
- Use networkx for quick static plots (already a dev dependency)
- Or generate SVG directly from layout engine
- Avoid heavy matplotlib dependency in core

---

### Issue 5: Loading time on .table() for large graphs

**Current State:**
- `.table()` method converts graph data to table format
- Might be doing expensive operations

**Problem:**
- Performance issue, not directly viz-related but affects viz workflow
- Users want to visualize but get stuck on table generation

**Analysis Needed:**
1. Profile `.table()` method to find bottleneck
2. Check if it's:
   - Data extraction from Rust?
   - Python-side formatting?
   - Display formatting?
   - All attribute access?

**Quick Win Options:**

**Option A: Lazy loading/streaming**
```python
# Don't load all data upfront
table = g.nodes.table(lazy=True)  
# Or stream in chunks
table = g.nodes.table(limit=1000)
```

**Option B: Cache table data**
```python
# Cache table between calls
g.nodes.table()  # Slow first time
g.nodes.table()  # Fast on repeat
```

**Option C: Async table generation**
```python
# Non-blocking
table_future = g.nodes.table_async()
# ... do other work ...
table = await table_future
```

**Recommendation:** 
1. **Profile first** to identify bottleneck
2. **Quick fix:** Add `limit` parameter for pagination
3. **Long term:** Implement streaming/lazy loading

This is not a viz issue per se, so let's defer detailed design until profiling.

---

## Implementation Priority

### High Priority (Core viz UX)
1. **Issue 1:** Make `.show()` update existing servers intelligently
2. **Issue 2:** Add discrete/continuous color mapping support
3. **Issue 3:** Define subgraph visualization behavior

### Medium Priority (Nice to have)
4. **Issue 4:** Add quick static viz methods

### Separate (Performance)
5. **Issue 5:** Profile and optimize `.table()` (not viz-specific)

## Next Steps

1. **Decide on approach for each issue** (user input needed)
2. **Implement Issue 1** (show() update behavior) - easiest win
3. **Implement Issue 2** (color mapping) - high impact
4. **Implement Issue 3** (subgraph viz) - important for clarity
5. **Defer Issue 4** until Issues 1-3 are done
6. **Profile Issue 5** separately

Would you like to discuss any of these options before implementation?
