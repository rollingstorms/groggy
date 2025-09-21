# 🧹 Viz System Cleanup Plan
## Unify Around Working Canvas + Rust Engine

### 🎯 **THE DISCOVERY**
We found the **working viz engine**: `GraphDataSource -> StreamingServer -> HTML5 Canvas`
- ✅ Serves real graph data (nodes/edges)
- ✅ WebSocket communication works
- ✅ Canvas rendering with interactive controls (embedded in streaming server)
- ✅ Accessible via `graph.graph_viz()`
- ✅ Self-contained HTML5 Canvas system in `src/viz/streaming/websocket_server.rs`

### 🎨 **CANVAS SYSTEM LOCATION** (Confirmed Working)
**Primary Canvas Implementation**: `src/viz/streaming/websocket_server.rs`
- Lines ~1467: `<canvas id="graph-canvas" class="graph-canvas"></canvas>`
- Lines ~1500-2000: Complete JavaScript canvas engine with `ctx.getContext('2d')`
- Vector drawing commands: `ctx.arc()`, `ctx.moveTo()`, `ctx.lineTo()`, `ctx.stroke()`
- WebSocket real-time updates
- **Completely self-contained** - no external dependencies

### 🗑️ **DELETE THESE FILES/DIRECTORIES** (Competing/Broken Systems)

#### 1. Python viz.py (Complex wrapper - not needed)
```bash
rm python-groggy/python/groggy/viz.py
```

#### 2. Test/Debug Files (Archaeology artifacts)
```bash
rm test_viz_*.py
rm preview_viz.py
rm working_viz_preview.py
rm test_viz_interactive_embed.py
rm test_viz_server.py
rm test_viz_simple.py
rm test_viz_quick.py
rm visual_test_*.py
rm debug_error_message.py
rm test_graph_data_websocket.py
rm test_graph_websocket.py
rm src/bin/test_graph_viz.rs
rm test_rust_graph_viz.rs
# Keep only: test_new_graph_viz.py (the working test)
```

#### 3. Competing Viz Systems in src/viz/ (**DEPENDENCY VERIFIED - SAFE TO DELETE**)
```bash
rm -rf src/viz/adapters/      # Unused adaptation layer
rm -rf src/viz/core/          # Competing core engine
rm -rf src/viz/frontend/      # Static frontend (unused by working streaming server)
rm -rf src/viz/server/        # Competing server (old)
rm src/viz/server.rs          # Competing server (old)
rm -rf src/viz/themes/        # Unused theme system
rm -rf src/viz/unified/       # Failed unification attempt
```

#### 4. Update src/viz/mod.rs (Remove deleted module declarations)
```rust
// REMOVE these lines from src/viz/mod.rs:
pub mod core;       // DELETE - competing core engine
pub mod themes;     // DELETE - unused theme system
pub mod unified;    // DELETE - failed unification attempt
```

#### 5. Archaeological Reports (Archive for reference)
```bash
mkdir archive/
mv FELLOWSHIP_FEATURE_ARCHAEOLOGY_REPORT.md archive/
mv UNIFIED_VIZ_MIGRATION_PLAN.md archive/
mv GRAPH_VISUALIZATION_SOLUTION.md archive/
```

### ✅ **KEEP AND EXTEND** (Working Foundation - NO IMPORT DEPENDENCIES)

#### 1. Core Working Components (**VERIFIED SAFE**)
```
✅ src/viz/streaming/         # The WebSocket + Canvas server that works
✅ src/viz/layouts/           # Layout algorithms (used by streaming)
✅ src/viz/display/           # Table/array/matrix formatting (ESSENTIAL - used by storage)
✅ src/viz/mod.rs             # Keep but prune unused module declarations
✅ src/api/graph.rs           # GraphDataSource - the working data source
```

#### 2. Python Interface (Working Solution)
```
✅ python-groggy/src/ffi/subgraphs/subgraph.rs::graph_viz()  # Working solution
✅ python-groggy/src/ffi/api/graph.rs::graph_viz()          # Working solution
```

#### 3. Evaluate JavaScript Components
```
⚠️ python-groggy/js-widget/   # Evaluate: replace with iframe to streaming server?
⚠️ python-groggy/python/groggy/static/groggy-viz-core.standalone.js  # Needed?
```

### 🎨 **UNIFIED ARCHITECTURE**

**Single Pipeline**: `Graph -> GraphDataSource -> StreamingServer -> HTML5 Canvas`

**All Backends Use This**:
- **Interactive (`graph.graph_viz()`)**: Direct streaming server
- **Jupyter Widget**: Iframe to streaming server
- **Save File**: Capture canvas state -> PNG/SVG export
- **Served**: Same streaming server
- **Everything**: Same canvas rendering engine

### 🔧 **IMPLEMENTATION STEPS**

#### Step 1: Cleanup (Delete unused systems)
```bash
# Run the deletion commands above
```

#### Step 2: Consolidate src/viz/mod.rs
- Remove exports for deleted systems
- Keep only streaming, layouts, and data_source exports

#### Step 3: Extend Working System
- Add save/export functionality to StreamingServer
- Create Jupyter widget that uses iframe to streaming server
- Unify all backends around this single engine

#### Step 4: Update Documentation
- Remove references to deleted systems
- Document the unified GraphDataSource approach
- Update examples to use `graph.graph_viz()`

### 📊 **BEFORE/AFTER**

**Before**: 9+ competing viz systems, broken Python wrappers
**After**: 1 unified Rust engine, simple Python interface

**File Count Reduction**: ~50+ viz files -> ~10 core files

**Complexity Reduction**: Multiple competing approaches -> Single working pipeline

### 🎯 **RESULT**

- **One engine**: GraphDataSource + StreamingServer + Canvas
- **One interface**: `graph.graph_viz()`
- **All backends**: Flow through same Rust engine
- **Massive cleanup**: Remove 80% of viz complexity
- **Actually works**: Real graph objects in visualization

This unifies everything around the **working solution** we just discovered! 🎉