# Phase 2: Streaming Infrastructure + Visualization Integration Implementation

## 🎯 **Phase Overview**

**Timeline**: 2 weeks (following Phase 1 completion)
**Critical Path**: YES - enables large dataset handling and visualization
**Goal**: Integrate streaming capabilities and interactive visualization system while maintaining foundation delegation pattern

## 🏗️ **Core Architecture Principles**

### **Foundation-Only Implementation Rule (Continued from Phase 1)**
```rust
// ✅ CORRECT: Streaming + Visualization logic lives ONLY here
impl BaseTable {
    display_engine: DisplayEngine,           // From Phase 1
    streaming_server: Option<StreamingServer>,   // NEW: Phase 2
    viz_module: VizModule,                   // NEW: Phase 2
    
    fn interactive(&self, config: InteractiveConfig) -> BrowserInterface {
        // FULL IMPLEMENTATION - streaming + virtual scrolling
    }
    
    fn viz(&self) -> &VizModule {
        // FULL IMPLEMENTATION - visualization system
    }
}

impl BaseArray {
    display_engine: DisplayEngine,           // From Phase 1  
    streaming_server: Option<StreamingServer>,   // NEW: Phase 2
    viz_module: VizModule,                   // NEW: Phase 2
    
    // Same pattern as BaseTable
}

// ❌ NEVER: Custom streaming/viz logic in specialized types
impl NodesTable {
    fn interactive(&self, config: InteractiveConfig) -> BrowserInterface {
        self.base_table.interactive(config)   // PURE DELEGATION ONLY
    }
    
    fn viz(&self) -> &VizModule {
        self.base_table.viz()                 // PURE DELEGATION ONLY
    }
}
```

### **Delegation Enforcement Strategy (Phase 2)**
1. **Code Review Checklist**: No streaming/viz logic in specialized types
2. **Build-time Validation**: Automated checks for delegation pattern
3. **API Surface Testing**: Identical methods across all types
4. **Integration Testing**: All 20+ types get streaming + viz via delegation

## 📋 **Week-by-Week Implementation Plan**

### **Week 1: Streaming Infrastructure Foundation**

#### **Day 1-2: DataSource Abstraction & Virtual Scrolling**
```rust
// File: src/core/streaming/data_source.rs
pub trait DataSource: Send + Sync {
    fn total_rows(&self) -> usize;
    fn total_cols(&self) -> usize;  
    fn get_window(&self, start: usize, count: usize) -> DataWindow;
    fn get_schema(&self) -> DataSchema;
    fn supports_streaming(&self) -> bool;
    fn get_column_types(&self) -> Vec<DataType>;
    fn get_column_names(&self) -> Vec<String>;
}

// File: src/core/streaming/virtual_scroller.rs
pub struct VirtualScrollManager {
    window_size: usize,           // Visible rows (default: 50)
    buffer_size: usize,           // Preloaded rows (default: 200)
    current_offset: usize,        // Current scroll position
    data_cache: LRUCache<WindowKey, DataWindow>,
}

impl VirtualScrollManager {
    pub fn new(window_size: usize, buffer_size: usize) -> Self { /* */ }
    pub fn get_visible_window(&self, offset: usize) -> DataWindow { /* */ }
    pub fn preload_buffer(&mut self, data_source: &dyn DataSource) { /* */ }
    pub fn handle_scroll(&mut self, new_offset: usize) -> UpdateResult { /* */ }
}
```

**Tasks**:
- [ ] Create DataSource trait for unified data access
- [ ] Implement DataSource for BaseTable and BaseArray
- [ ] Build VirtualScrollManager with LRU caching
- [ ] Create DataWindow caching system
- [ ] Add streaming configuration management
- **Estimated Time**: 16 hours

#### **Day 3-4: WebSocket Server & Communication**
```rust
// File: src/core/streaming/websocket_server.rs
pub struct StreamingServer {
    websocket_server: WebSocketServer,
    virtual_scroller: VirtualScrollManager,
    html_generator: HtmlTableGenerator,
    data_source: Arc<dyn DataSource>,
    active_connections: HashMap<ConnectionId, ClientState>,
}

impl StreamingServer {
    pub fn new(data_source: Arc<dyn DataSource>, config: StreamingConfig) -> Self { /* */ }
    pub fn start(&self, port: u16) -> Result<ServerHandle> { /* */ }
    pub fn broadcast_update(&self, update: DataUpdate) -> Result<()> { /* */ }
    pub fn handle_client_scroll(&self, conn_id: ConnectionId, offset: usize) -> Result<()> { /* */ }
}

// WebSocket message protocol
#[derive(Serialize, Deserialize)]
pub enum WSMessage {
    InitialData { window: DataWindow, total_rows: usize },
    DataUpdate { new_window: DataWindow, offset: usize },
    ScrollRequest { offset: usize, window_size: usize },
    ThemeChange { theme: String },
    Error { message: String },
}
```

**Tasks**:
- [ ] Implement WebSocket server with tokio-tungstenite
- [ ] Create streaming message protocol
- [ ] Build client connection management
- [ ] Add real-time data synchronization
- [ ] Create HTML table generator for streaming
- **Estimated Time**: 16 hours

#### **Day 5: BaseTable/BaseArray Integration**
```rust
// File: src/storage/table/base.rs - Integration
impl BaseTable {
    // Add new fields (non-breaking)
    streaming_server: Option<StreamingServer>,
    streaming_config: StreamingConfig,
    
    // NEW: Interactive method
    pub fn interactive(&self, config: Option<InteractiveConfig>) -> Result<BrowserInterface> {
        let config = config.unwrap_or_default();
        
        // Create data source from self
        let data_source: Arc<dyn DataSource> = Arc::new(self.clone());
        
        // Launch streaming server
        let server = StreamingServer::new(data_source, config.streaming_config);
        let server_handle = server.start(config.port)?;
        
        // Open browser interface
        let browser_interface = BrowserInterface::new(server_handle, config.browser_config);
        browser_interface.launch()?;
        
        Ok(browser_interface)
    }
    
    // Implement DataSource trait
    fn total_rows(&self) -> usize { self.nrows() }
    fn total_cols(&self) -> usize { self.ncols() }
    fn get_window(&self, start: usize, count: usize) -> DataWindow {
        // Convert BaseTable data to streaming DataWindow format
        DataWindow {
            headers: self.column_names(),
            rows: self.get_rows_range(start, start + count),
            schema: self.get_schema(),
            total_rows: self.nrows(),
            start_offset: start,
        }
    }
}

// File: src/storage/array/base.rs - Same pattern
impl BaseArray {
    streaming_server: Option<StreamingServer>,
    
    pub fn interactive(&self, config: Option<InteractiveConfig>) -> Result<BrowserInterface> {
        // Arrays presented as single-column tables
        let table_view = self.as_table_view();
        table_view.interactive(config)
    }
}
```

**Tasks**:
- [ ] Integrate StreamingServer into BaseTable struct
- [ ] Implement DataSource trait for BaseTable
- [ ] Create interactive() method in BaseTable
- [ ] Add BaseArray streaming support (via table view)
- [ ] Update all BaseTable constructors with streaming fields
- **Estimated Time**: 8 hours

### **Week 2: Visualization Integration**

#### **Day 6-7: VizModule Foundation**
```rust
// File: src/core/visualization/viz_module.rs
pub struct VizModule {
    graph_data: GraphData,
    layout_engine: LayoutEngine,
    renderers: RendererCollection,
    theme_system: VizThemeSystem,
}

impl VizModule {
    pub fn interactive(&self, options: VizOptions) -> Result<InteractiveViz> {
        let layout = self.layout_engine.compute_layout(
            &self.graph_data, 
            options.layout_algorithm
        );
        
        let viz = InteractiveViz::new(
            layout,
            options.theme,
            options.interaction_config
        );
        
        viz.launch_d3_interface(options.port)
    }
    
    pub fn static_viz(&self, options: StaticVizOptions) -> Result<StaticOutput> {
        let renderer = match options.format {
            ExportFormat::PNG => self.renderers.png_renderer(),
            ExportFormat::SVG => self.renderers.svg_renderer(), 
            ExportFormat::PDF => self.renderers.pdf_renderer(),
        };
        
        let layout = self.layout_engine.compute_layout(
            &self.graph_data,
            options.layout_algorithm
        );
        
        renderer.render(layout, options.theme)
    }
}

// File: src/core/visualization/layout_engine.rs
pub struct LayoutEngine {
    algorithms: HashMap<String, Box<dyn LayoutAlgorithm>>,
}

pub trait LayoutAlgorithm {
    fn compute_layout(&self, graph: &GraphData, config: LayoutConfig) -> Layout;
    fn supports_streaming(&self) -> bool;
    fn update_layout(&self, layout: &mut Layout, changes: &GraphChanges) -> Result<()>;
}

pub enum BuiltInLayout {
    ForceDirected,    // Spring-embedder algorithm
    Hierarchical,     // Tree-based positioning
    Circular,         // Nodes arranged in circles  
    Grid,             // Regular grid positioning
}
```

**Tasks**:
- [ ] Create VizModule with layout engine architecture
- [ ] Implement LayoutAlgorithm trait system
- [ ] Build 4 built-in layout algorithms
- [ ] Create GraphData abstraction for visualization
- [ ] Add VizThemeSystem for visualization styling
- **Estimated Time**: 16 hours

#### **Day 8-9: Interactive & Static Renderers**
```rust
// File: src/core/visualization/renderers/interactive.rs
pub struct InteractiveViz {
    d3_server: D3Server,
    websocket_server: VizWebSocketServer,
    layout: Layout,
    theme: VizTheme,
}

impl InteractiveViz {
    pub fn launch_d3_interface(&self, port: u16) -> Result<VizHandle> {
        // Generate D3.js visualization code
        let d3_code = self.generate_d3_visualization();
        
        // Start web server with D3 interface
        let server = self.d3_server.start(port, d3_code)?;
        
        // Launch browser
        open_browser(&format!("http://localhost:{}", port))?;
        
        Ok(VizHandle::new(server))
    }
    
    fn generate_d3_visualization(&self) -> String {
        // Generate complete D3.js code for interactive visualization
        // Includes: zoom, pan, node dragging, edge highlighting
        // Real-time updates via WebSocket
    }
}

// File: src/core/visualization/renderers/static.rs
pub struct PngRenderer {
    dpi: u32,
    width: u32,
    height: u32,
}

pub struct SvgRenderer {
    vector_precision: f64,
}

pub struct PdfRenderer {
    page_size: PageSize,
    margins: Margins,
}

impl StaticRenderer for PngRenderer {
    fn render(&self, layout: Layout, theme: VizTheme) -> Result<Vec<u8>> {
        // Use resvg or similar for high-quality PNG rendering
        // Support high-DPI output for publications
    }
}
```

**Tasks**:
- [ ] Build InteractiveViz with D3.js integration
- [ ] Create D3 code generation for graph visualization  
- [ ] Implement PngRenderer with high-DPI support
- [ ] Build SvgRenderer for vector graphics
- [ ] Create PdfRenderer for publication output
- **Estimated Time**: 16 hours

#### **Day 10: Foundation Integration & Delegation**
```rust
// File: src/storage/table/base.rs - Add visualization
impl BaseTable {
    streaming_server: Option<StreamingServer>,  // Phase 2 Week 1
    viz_module: VizModule,                      // NEW: Phase 2 Week 2
    
    // NEW: Visualization access method
    pub fn viz(&self) -> &VizModule {
        &self.viz_module
    }
    
    // Convert table data to graph format for visualization
    fn as_graph_data(&self) -> GraphData {
        // Convert BaseTable to graph representation
        // Nodes from node_id column, edges from relationships
        GraphData::from_table(self)
    }
}

// File: src/storage/array/base.rs - Same pattern
impl BaseArray {
    viz_module: VizModule,
    
    pub fn viz(&self) -> &VizModule {
        &self.viz_module
    }
}

// PURE DELEGATION for ALL specialized types
impl NodesTable {
    fn interactive(&self, config: Option<InteractiveConfig>) -> Result<BrowserInterface> {
        self.base_table.interactive(config)  // DELEGATION ONLY
    }
    
    fn viz(&self) -> &VizModule {
        self.base_table.viz()                // DELEGATION ONLY
    }
}

// Same delegation pattern for:
// EdgesTable, GraphTable, ComponentsArray, GraphArray, Matrix, etc.
```

**Tasks**:
- [ ] Integrate VizModule into BaseTable and BaseArray
- [ ] Implement viz() method in foundation classes
- [ ] Create delegation methods in ALL specialized types
- [ ] Add GraphData conversion for tables and arrays
- [ ] Update constructors with viz_module initialization
- **Estimated Time**: 8 hours

### **Week 2 Continuation: CSS Framework & Testing**

#### **Day 11-12: Enhanced CSS Framework**
```css
/* File: src/core/display/themes/streaming.css */
/* Streaming table specific styles */
.groggy-streaming-container {
    position: relative;
    height: 400px;
    overflow: hidden;
    border: 1px solid var(--border-color);
    border-radius: 8px;
}

.groggy-virtual-scroller {
    height: 100%;
    overflow-y: auto;
    overflow-x: auto;
}

.groggy-loading-indicator {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}

/* Mobile responsive */
@media (max-width: 768px) {
    .groggy-streaming-container {
        height: 300px;
        font-size: 14px;
    }
    
    .groggy-table {
        min-width: auto;
    }
}

/* File: src/core/display/themes/visualization.css */
/* Visualization specific styles */
.groggy-viz-container {
    width: 100%;
    height: 600px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    position: relative;
}

.groggy-viz-controls {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgba(255, 255, 255, 0.9);
    padding: 10px;
    border-radius: 4px;
}

/* D3.js integration styles */
.groggy-node {
    stroke: var(--node-stroke);
    stroke-width: 2px;
    cursor: pointer;
}

.groggy-edge {
    stroke: var(--edge-stroke);
    stroke-width: 1px;
}

.groggy-node:hover {
    stroke-width: 3px;
    filter: brightness(1.2);
}
```

**Tasks**:
- [ ] Create streaming table CSS framework
- [ ] Build visualization CSS themes
- [ ] Add responsive design for mobile devices
- [ ] Integrate CSS with theme system
- [ ] Test cross-browser compatibility
- **Estimated Time**: 12 hours

#### **Day 13-14: Python FFI Integration**
```rust
// File: python-groggy/src/ffi/storage/table.rs - Add streaming methods
#[pymethods]
impl PyBaseTable {
    // Existing display methods from Phase 1...
    
    /// Launch interactive streaming table in browser
    #[pyo3(signature = (port = 8080, theme = "light"))]
    pub fn interactive(&self, port: u16, theme: &str) -> PyResult<()> {
        let config = InteractiveConfig {
            port,
            theme: theme.to_string(),
            ..Default::default()
        };
        
        self.table.interactive(Some(config))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(())
    }
    
    /// Access visualization module
    pub fn viz(&self) -> PyVizModule {
        PyVizModule {
            inner: self.table.viz().clone()
        }
    }
}

// File: python-groggy/src/ffi/visualization/mod.rs
#[pyclass(name = "VizModule")]
pub struct PyVizModule {
    inner: VizModule,
}

#[pymethods]
impl PyVizModule {
    /// Launch interactive visualization
    #[pyo3(signature = (layout = "force_directed", theme = "light", port = 8081))]
    pub fn interactive(&self, layout: &str, theme: &str, port: u16) -> PyResult<()> {
        let options = VizOptions {
            layout_algorithm: LayoutAlgorithm::from_str(layout)?,
            theme: VizTheme::from_str(theme)?,
            port,
            ..Default::default()
        };
        
        self.inner.interactive(options)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(())
    }
    
    /// Export static visualization
    #[pyo3(signature = (path, format = "png", layout = "force_directed", theme = "light"))]
    pub fn export(&self, path: &str, format: &str, layout: &str, theme: &str) -> PyResult<()> {
        let options = StaticVizOptions {
            format: ExportFormat::from_str(format)?,
            layout_algorithm: LayoutAlgorithm::from_str(layout)?,
            theme: VizTheme::from_str(theme)?,
            output_path: path.to_string(),
            ..Default::default()
        };
        
        let output = self.inner.static_viz(options)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        output.save_to_file(path)?;
        Ok(())
    }
}
```

**Tasks**:
- [ ] Add interactive() method to Python FFI  
- [ ] Create PyVizModule wrapper class
- [ ] Implement visualization export functionality
- [ ] Add error handling and type conversion
- [ ] Create Python configuration classes
- **Estimated Time**: 12 hours

## 🧪 **Testing Strategy**

### **Streaming Tests**
```python
# File: tests/test_streaming.py
def test_virtual_scrolling():
    """Test virtual scrolling with large datasets."""
    g = create_large_graph(nodes=100000, edges=500000)
    
    # Should handle massive table without memory issues
    table = g.nodes.table()
    
    # Test virtual scrolling
    browser = table.interactive(port=8080, theme="dark")
    assert browser.is_running()
    
    # Test data window requests
    window = browser.get_data_window(offset=50000, size=50)
    assert len(window.rows) == 50
    assert window.start_offset == 50000

def test_real_time_updates():
    """Test WebSocket real-time data updates."""
    g = gg.Graph()
    g.add_nodes(1000)
    
    table = g.nodes.table()
    browser = table.interactive()
    
    # Add more nodes - should stream to browser
    g.add_nodes(100)  
    
    # Verify browser receives update
    time.sleep(0.1)  # Allow WebSocket propagation
    assert browser.get_total_rows() == 1100
```

### **Visualization Tests**
```python
# File: tests/test_visualization.py  
def test_interactive_visualization():
    """Test interactive D3.js visualization."""
    g = create_sample_graph()
    table = g.nodes.table()
    
    # Launch interactive viz
    viz_handle = table.viz().interactive(
        layout="force_directed",
        theme="dark", 
        port=8081
    )
    
    assert viz_handle.is_running()
    assert viz_handle.get_url() == "http://localhost:8081"

def test_static_export():
    """Test static visualization export."""
    g = create_sample_graph()
    table = g.nodes.table()
    
    # Export PNG
    table.viz().export(
        "test_viz.png",
        format="png",
        layout="hierarchical",
        theme="publication"
    )
    
    assert os.path.exists("test_viz.png")
    assert get_image_size("test_viz.png") == (800, 600)
```

### **Delegation Tests**  
```python
# File: tests/test_phase2_delegation.py
def test_streaming_delegation():
    """Verify ALL table types get streaming via delegation."""
    g = create_test_graph()
    
    # Test all table types have identical streaming interface
    tables = [
        g.nodes.table(),      # NodesTable
        g.edges.table(),      # EdgesTable
        g.graph_table(),      # GraphTable
    ]
    
    for table in tables:
        # All should have identical streaming methods
        assert hasattr(table, 'interactive')
        assert hasattr(table, 'viz')
        
        # All should work identically
        browser = table.interactive(port=find_free_port())
        assert browser.is_running()
        browser.close()

def test_visualization_delegation():
    """Verify ALL table types get visualization via delegation."""
    g = create_test_graph()
    
    tables = [g.nodes.table(), g.edges.table(), g.graph_table()]
    
    for table in tables:
        viz_module = table.viz()
        assert isinstance(viz_module, VizModule)
        
        # Test export functionality
        output_path = f"test_{table.__class__.__name__}.svg"
        viz_module.export(output_path, format="svg")
        assert os.path.exists(output_path)
```

### **Performance Tests**
```python
# File: tests/test_phase2_performance.py
def test_streaming_performance():
    """Test streaming performance with large datasets."""
    g = create_large_graph(nodes=1000000, edges=5000000)
    table = g.nodes.table()
    
    # Streaming should start quickly even with huge dataset
    start_time = time.time()
    browser = table.interactive()
    startup_time = time.time() - start_time
    
    assert startup_time < 2.0  # Should start in < 2 seconds
    
    # Virtual scrolling should be responsive
    start_time = time.time()
    window = browser.get_data_window(offset=500000, size=100)
    scroll_time = time.time() - start_time
    
    assert scroll_time < 0.1  # Should scroll in < 100ms
    assert len(window.rows) == 100

def test_visualization_performance():
    """Test visualization performance."""
    g = create_large_graph(nodes=10000, edges=50000)
    table = g.nodes.table()
    
    # Layout computation should complete reasonably quickly
    start_time = time.time()
    layout = table.viz().compute_layout("force_directed")
    layout_time = time.time() - start_time
    
    assert layout_time < 5.0  # Should layout in < 5 seconds
    assert len(layout.node_positions) == 10000
```

## 🚨 **Risk Mitigation**

### **Technical Risks**
- **WebSocket Complexity**: Start with simple protocols, add complexity gradually
- **D3.js Integration**: Use established D3 patterns, test cross-browser compatibility  
- **Memory Usage**: Implement LRU caching, monitor memory with large datasets
- **Performance Regression**: Benchmark streaming vs current system continuously

### **Architectural Risks**
- **Delegation Breaking**: Strict code review process to prevent custom logic in specialized types
- **API Surface Drift**: Automated testing for method signature consistency across all types
- **WebSocket Reliability**: Implement reconnection logic and graceful degradation
- **Visualization Complexity**: Start with 2 layout algorithms, expand to 4 after core works

## 🎯 **Phase 2 Success Criteria**

At the end of Phase 2, we should have:

1. **Complete Streaming Infrastructure**:
   - Virtual scrolling for 1M+ row datasets
   - WebSocket server for real-time updates  
   - LRU caching system for data windows
   - Browser interface launching for all table types

2. **Full Visualization System**:
   - Interactive D3.js visualizations with 4 layout algorithms
   - Static export to PNG, SVG, PDF formats
   - Unified theme system across tables and visualizations
   - Real-time visualization updates via WebSocket

3. **Foundation Delegation Architecture**:
   - ALL streaming functionality in BaseTable/BaseArray only
   - ALL visualization functionality in foundation classes only
   - Pure delegation in ALL specialized types (20+ types)
   - Zero code duplication for streaming + visualization

4. **Production-Ready Integration**:
   - Complete Python FFI bindings for streaming + visualization
   - Professional CSS framework with mobile responsiveness
   - Comprehensive test suite for streaming + visualization + delegation
   - Performance benchmarks meeting targets

**Phase 2 Success = All 20+ data structure types have identical, high-performance streaming and visualization capabilities with zero code duplication. Foundation for massive dataset handling and interactive analysis is established.**

---

## 🚀 **Ready to Build Streaming + Visualization on Our Solid Phase 1 Foundation!**

**Phase 2 transforms Groggy from static display to interactive streaming visualization platform! 🎨📊**