# 🎨 Unified Visualization Core Migration Plan

## Current Architecture Analysis

You already have an excellent foundation:
- ✅ **VizBackend enum** (Jupyter, Streaming, File, Local)
- ✅ **DataSource trait** (unified data interface)
- ✅ **LayoutEngine trait** (ForceDirectedLayout + others)
- ✅ **Streaming infrastructure** (WebSocket server, virtual scrolling)
- ✅ **Display system** (themes, HTML export)
- ✅ **Jupyter widget** (working TypeScript implementation)

## 🚀 Migration Strategy: Extract + Unify

### Phase 1: Core Engine Extraction (Week 1)

#### 1.1 Create Unified Core Module
```
src/viz/core/
├── mod.rs               // Core module exports
├── engine.rs           // Main VizEngine struct
├── physics.rs          // Extracted physics simulation
├── rendering.rs        // Unified rendering pipeline
├── interaction.rs      // Interaction state management
└── frame.rs            // Frame data structures
```

#### 1.2 Extract Physics Engine
Move from `layouts/mod.rs` ForceDirectedLayout to `core/physics.rs`:
- ✅ Your comprehensive force-directed simulation
- ✅ Barnes-Hut optimization 
- ✅ Collision detection
- ✅ Adaptive cooling

#### 1.3 Create VizEngine Core
```rust
pub struct VizEngine {
    // Core state
    nodes: Vec<VizNode>,
    edges: Vec<VizEdge>,
    
    // Engines
    physics: PhysicsEngine,
    layout: Box<dyn LayoutEngine>,
    
    // State
    positions: HashMap<String, Position>,
    interaction_state: InteractionState,
    
    // Configuration
    config: VizConfig,
}

impl VizEngine {
    pub fn update(&mut self) -> VizFrame {
        // Single update method used by ALL adapters
        self.physics.update(&mut self.positions);
        self.render_frame()
    }
}
```

### Phase 2: Adapter Refactoring (Week 2)

#### 2.1 Streaming Adapter
```rust
// src/viz/adapters/streaming.rs
pub struct StreamingAdapter {
    core: VizEngine,
    server: StreamingServer,
    clients: Vec<WebSocketClient>,
}

impl VizAdapter for StreamingAdapter {
    fn render(&mut self) -> GraphResult<()> {
        let frame = self.core.update();
        self.broadcast_frame(frame)
    }
}
```

#### 2.2 Jupyter Widget Adapter  
```rust
// src/viz/adapters/jupyter.rs
pub struct JupyterAdapter {
    core: VizEngine,
    python_bridge: PyO3Bridge,
}

impl VizAdapter for JupyterAdapter {
    fn render(&mut self) -> GraphResult<WidgetFrame> {
        let frame = self.core.update();
        Ok(WidgetFrame::from_viz_frame(frame))
    }
}
```

#### 2.3 File Export Adapter
```rust
// src/viz/adapters/file.rs  
pub struct FileAdapter {
    core: VizEngine,
    output_path: PathBuf,
    format: ExportFormat, // SVG, PNG, HTML
}
```

### Phase 3: JavaScript Unification (Week 3)

#### 3.1 Extract JavaScript Core
```javascript
// js-widget/src/core/VizCore.js
class GroggyVizCore {
    constructor(nodes, edges, config) {
        this.physics = new PhysicsEngine(config.physics);
        this.renderer = new SVGRenderer(config.rendering);
        this.interactions = new InteractionEngine(config.interaction);
    }
    
    // Single update method for all environments
    update(frameData) {
        this.physics.updatePositions(frameData.positions);
        this.renderer.render(frameData);
        this.interactions.handleEvents(frameData.events);
    }
}
```

#### 3.2 Refactor Widget to Use Core
```javascript
// js-widget/src/widget.ts
export class GroggyGraphView extends DOMWidgetView {
    render() {
        const nodes = this.model.get('nodes');
        const edges = this.model.get('edges');
        
        // Use unified core - same as streaming!
        this.vizCore = new GroggyVizCore(nodes, edges, config);
        this.vizCore.attachToDOM(this.el);
    }
}
```

#### 3.3 Refactor Streaming Client
```javascript
// Streaming client uses same core
class StreamingClient {
    connect() {
        this.vizCore = new GroggyVizCore(nodes, edges, config);
        this.websocket.onmessage = (frame) => {
            this.vizCore.update(frame.data);
        };
    }
}
```

### Phase 4: Unified API (Week 4)

#### 4.1 Single Entry Point
```rust
impl Graph {
    pub fn viz(&self) -> VizModule {
        VizModule::new(self)
    }
}

impl VizModule {
    // Unified API - one method, all backends
    pub fn render(&self, backend: VizBackend) -> GraphResult<VizResult> {
        match backend {
            VizBackend::Jupyter => self.jupyter_adapter().render(),
            VizBackend::Streaming => self.streaming_adapter().render(),
            VizBackend::File => self.file_adapter().render(),
            VizBackend::Local => self.local_adapter().render(),
        }
    }
    
    // Convenience methods
    pub fn widget(&self) -> JupyterWidget { self.render(VizBackend::Jupyter) }
    pub fn serve(&self, port: u16) -> StreamingServer { self.render(VizBackend::Streaming) }
    pub fn save(&self, path: &str) -> GraphResult<()> { self.render(VizBackend::File) }
}
```

## 🎯 Migration Benefits

### ✅ Single Source of Truth
- Physics simulation: One implementation
- Layout algorithms: One implementation  
- Styling system: One implementation
- Interaction logic: One implementation

### ✅ Feature Parity Guaranteed
- New layout? Works in widget AND streaming
- New interaction? Works everywhere
- Performance improvement? Benefits all

### ✅ Elegant User API
```python
# All of these use the SAME core engine
g.viz().widget()                    # Jupyter widget
g.viz().serve(port=8080)           # Streaming server  
g.viz().save("graph.html")         # Static HTML
g.viz().save("graph.svg")          # Vector graphics
```

## 📋 Migration Steps (Detailed)

### Step 1: Extract VizEngine Core
1. Create `src/viz/core/mod.rs`
2. Move physics from `layouts/mod.rs` to `core/physics.rs`
3. Create `VizEngine` struct that wraps your existing engines
4. Test that core produces same output as current system

### Step 2: Create Adapters
1. Create `src/viz/adapters/mod.rs`
2. Wrap existing streaming server in `StreamingAdapter`
3. Create `JupyterAdapter` for widget integration
4. Test adapters produce same output as direct usage

### Step 3: Unify JavaScript
1. Extract common logic from widget TypeScript
2. Create unified `GroggyVizCore` class
3. Refactor widget to use core
4. Create streaming client that uses same core

### Step 4: Update APIs
1. Add unified `render(backend)` method to `VizModule`
2. Keep existing convenience methods as wrappers
3. Update documentation
4. Add migration guide

## 🔧 Implementation Priority

### High Priority (Do First)
1. ✅ **VizEngine core extraction** - Foundation for everything
2. ✅ **Streaming adapter** - Preserve existing functionality  
3. ✅ **Jupyter adapter** - Match current widget behavior

### Medium Priority (Do Second)  
1. **File export adapter** - Enable static output
2. **JavaScript unification** - Reduce duplication
3. **API cleanup** - Elegant user interface

### Low Priority (Polish)
1. **Performance optimization** - Leverage unified core
2. **Additional export formats** - PDF, PNG, etc.
3. **Advanced interactions** - Consistent across backends

## 💡 Migration Tips

### Preserve Backwards Compatibility
```rust
// Keep existing methods working
impl Graph {
    // NEW: Unified approach
    pub fn viz(&self) -> VizModule { /* ... */ }
    
    // OLD: Keep working during migration
    pub fn interactive(&self) -> InteractiveViz { self.viz().widget() }
    pub fn streaming(&self) -> StreamingServer { self.viz().serve() }
}
```

### Test-Driven Migration
1. ✅ **Extract with tests** - Core engine produces same output
2. ✅ **Adapter tests** - Each adapter works identically  
3. ✅ **Integration tests** - End-to-end functionality preserved
4. ✅ **Performance tests** - No regressions

### Incremental Rollout
1. ✅ **Core + Streaming** - Get foundation working
2. ✅ **Add Jupyter** - Match existing widget
3. ✅ **Add File Export** - New capability  
4. ✅ **JavaScript Unification** - Reduce maintenance

## 🧹 Phase 5: Cleanup and Optimization

After the unified core is working with all adapters, clean up the legacy code:

### 5.1 Remove Duplicate Physics Code
- ✅ Remove original `ForceDirectedLayout` from `layouts/mod.rs`
- ✅ Update any remaining references to use `core/physics.rs`
- ✅ Remove deprecated layout implementations

### 5.2 Consolidate JavaScript
- ✅ Remove duplicate physics code from widget TypeScript
- ✅ Remove duplicate physics code from streaming client
- ✅ Ensure all JavaScript uses unified `GroggyVizCore`

### 5.3 API Cleanup
- ✅ Mark old methods as deprecated with migration hints
- ✅ Add comprehensive migration guide
- ✅ Update all documentation to reference unified API

### 5.4 Performance Optimization
- ✅ Benchmark unified core vs. original implementations
- ✅ Optimize hot paths in physics simulation
- ✅ Add performance regression tests
- ✅ Memory usage optimization

### 5.5 Testing and Validation
- ✅ Comprehensive integration tests across all adapters
- ✅ Visual regression tests (screenshot comparison)
- ✅ Performance benchmarks
- ✅ Memory leak detection

## 🎉 End State Vision

```python
# One API, all backends, same engine
import groggy

g = groggy.Graph()
g.add_nodes([...])
g.add_edges([...])

# All use identical physics, layouts, styling
widget = g.viz().widget()          # Jupyter
server = g.viz().serve()           # Interactive streaming  
g.viz().save("output.html")        # Static file
g.viz().save("output.svg")         # Vector graphics
```

**Result**: One engine, four output modes, zero duplication! 🎨