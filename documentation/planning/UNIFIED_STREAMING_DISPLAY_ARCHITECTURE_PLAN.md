# Unified Streaming Display Architecture Plan

## 🎯 **Executive Summary**

**Problem**: We currently have parallel, inconsistent display systems for arrays, matrices, and tables with:
- Duplicate representation logic across different data structures  
- No unified streaming/lazy-loading for large datasets
- Inconsistent visual styling and formatting
- Limited HTML output (currently `<pre>` wrapped Unicode)

**Solution**: Build a centralized, streaming-capable display architecture that unifies all data structure representations while enabling interactive browser interfaces for massive datasets.

## 📊 **Current State Assessment**

### **Existing Display Infrastructure**
```
📁 python-groggy/python/groggy/display/
├── __init__.py           # Global display configuration
├── formatters.py         # High-level formatting functions
├── table_display.py      # TableDisplayFormatter class
├── array_display.py      # ArrayDisplayFormatter class  
├── matrix_display.py     # MatrixDisplayFormatter class
├── unicode_chars.py      # Box-drawing characters & symbols
└── truncation.py         # Smart truncation logic
```

### **Current Repr Methods in FFI Layer**
- **Tables**: `__repr__`, `__str__`, `rich_display()`, `_repr_html_()`
- **Arrays**: `__repr__`, `__str__`, basic formatting
- **Matrices**: `__repr__`, `__str__`, `rich_display()`, `_repr_html_()`  
- **TableArray**: `__repr__` only

### **Issues Identified**
1. **Fragmentation**: Each data structure has its own formatting logic
2. **Inconsistency**: Different styling approaches across types
3. **No Streaming**: All data must be loaded for display
4. **Limited HTML**: `_repr_html_()` just wraps Unicode in `<pre>` tags
5. **Performance**: Large datasets cause memory issues in notebooks

## 🏗️ **Unified Architecture Design**

### **Core Principles**
1. **Single Source of Truth**: One display system for all data structures
2. **Lazy Loading**: Only render what's visible + buffer
3. **Progressive Enhancement**: Unicode → HTML → Interactive Browser
4. **Template-Based**: Unified styling and theming
5. **Streaming Ready**: Built for infinite scrolling from day one

### **Three-Tier Display Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DISPLAY TIER 1                            │
│                     Unicode Text Display                           │  
│  • Rich terminal output with box drawing                           │
│  • Used for: __repr__, __str__, print()                           │
│  • Fast, lightweight, always works                                 │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          DISPLAY TIER 2                            │
│                       HTML Table Display                           │
│  • Semantic HTML with CSS styling                                  │  
│  • Used for: _repr_html_(), Jupyter notebooks                     │
│  • Professional appearance, responsive design                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          DISPLAY TIER 3                            │
│                    Interactive Browser Interface                    │
│  • WebSocket-based streaming with virtual scrolling                │
│  • Used for: .interactive(), massive datasets                      │  
│  • Real-time updates, filtering, sorting                          │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔄 **Streaming Data Architecture**

### **Virtual Window System**
```rust
pub struct StreamingDisplayManager {
    // Data source - abstracted for all types
    data_source: Arc<dyn DataSource>,
    
    // Virtual window management
    viewport: ViewportWindow,
    buffer: DataBuffer,
    
    // Rendering pipeline
    renderer: DisplayRenderer,
    
    // Streaming infrastructure
    websocket_server: Option<WebSocketServer>,
}

pub struct ViewportWindow {
    visible_start: usize,      // First visible row
    visible_count: usize,      // Number of visible rows
    buffer_before: usize,      // Rows buffered before viewport  
    buffer_after: usize,       // Rows buffered after viewport
}

pub trait DataSource: Send + Sync {
    fn total_rows(&self) -> usize;
    fn total_cols(&self) -> usize;  
    fn get_window(&self, start: usize, count: usize) -> DataWindow;
    fn get_schema(&self) -> DataSchema;
    fn supports_streaming(&self) -> bool;
}
```

### **Unified Data Abstraction**
```rust
// All data structures implement this trait
impl DataSource for GraphTable {
    fn get_window(&self, start: usize, count: usize) -> DataWindow {
        // Table-specific windowing logic
    }
}

impl DataSource for GraphArray {  
    fn get_window(&self, start: usize, count: usize) -> DataWindow {
        // Array-specific windowing logic (transpose to table format)
    }
}

impl DataSource for GraphMatrix {
    fn get_window(&self, start: usize, count: usize) -> DataWindow {  
        // Matrix-specific windowing logic
    }
}
```

## 📋 **Implementation Plan**

### **Phase 1: Centralized Display System (Week 1-2)**

**Goal**: Consolidate all repr/display logic into a single, unified system

#### 1.1 Core Display Engine
```rust
// New unified display engine in Rust core
pub struct DisplayEngine {
    config: DisplayConfig,
    formatters: HashMap<DataType, Box<dyn DisplayFormatter>>,
}

pub trait DisplayFormatter {
    fn format_unicode(&self, data: &DataWindow, config: &DisplayConfig) -> String;
    fn format_html(&self, data: &DataWindow, config: &DisplayConfig) -> String;
    fn supports_streaming(&self) -> bool;
}

pub struct DisplayConfig {
    // Display limits
    pub max_rows: usize,
    pub max_cols: usize,
    pub max_width: usize,
    pub precision: usize,
    
    // Styling
    pub theme: DisplayTheme,
    pub output_format: OutputFormat,
    
    // Streaming settings
    pub buffer_size: usize,
    pub lazy_threshold: usize,
}

pub enum OutputFormat {
    Unicode,
    Html,
    Interactive,
}
```

#### 1.2 FFI Integration
```rust
// Update all FFI structures to use unified display
#[pymethods]
impl PyGraphTable {
    fn __repr__(&self) -> String {
        self.display_engine.format_unicode(&self.get_data_window(0, 10))
    }
    
    fn _repr_html_(&self) -> String {
        self.display_engine.format_html(&self.get_data_window(0, 100))  
    }
    
    fn rich_display(&self, config: Option<DisplayConfig>) -> String {
        // Unified display method for all types
        let config = config.unwrap_or_default();
        match config.output_format {
            OutputFormat::Unicode => self.__repr__(),
            OutputFormat::Html => self._repr_html_(),
            OutputFormat::Interactive => self.launch_interactive(),
        }
    }
}
```

**Deliverables**:
- ✅ Single `DisplayEngine` handles all data structure formatting
- ✅ Consistent styling across tables, arrays, matrices  
- ✅ Improved HTML output (semantic tables, not `<pre>`)
- ✅ Consolidated Python display module

### **Phase 2: HTML Enhancement & Theming (Week 3-4)**

**Goal**: Professional-grade HTML output with responsive design and theming

#### 2.1 HTML Template System
```rust
pub struct HtmlRenderer {
    template_engine: TemplateEngine,
    css_framework: CssFramework,
    themes: HashMap<String, Theme>,
}

pub struct Theme {
    pub colors: ColorPalette,
    pub typography: Typography,
    pub spacing: SpacingRules,
    pub responsive_breakpoints: Vec<Breakpoint>,
}
```

#### 2.2 Responsive CSS Framework
```css
/* Built-in themes */
.groggy-table.theme-light { /* Light theme styles */ }
.groggy-table.theme-dark { /* Dark theme styles */ }
.groggy-table.theme-publication { /* Academic paper theme */ }
.groggy-table.theme-minimal { /* Clean minimal theme */ }

/* Responsive design */
@media (max-width: 768px) {
    .groggy-table { /* Mobile-friendly adjustments */ }
}
```

**Deliverables**:
- ✅ Semantic HTML table generation
- ✅ Professional CSS framework with 4 built-in themes
- ✅ Mobile-responsive design
- ✅ Accessibility compliance (WCAG)

### **Phase 3: Streaming Infrastructure (Week 5-6)**  

**Goal**: Virtual scrolling and lazy loading for massive datasets

#### 3.1 Virtual Scrolling System
```rust
pub struct VirtualScrollManager {
    window_manager: WindowManager,
    data_cache: LRUCache<WindowKey, DataWindow>,
    prefetch_strategy: PrefetchStrategy,
}

pub struct WindowManager {
    viewport_size: usize,
    buffer_size: usize, 
    total_size: usize,
    current_offset: usize,
}

pub enum PrefetchStrategy {
    Conservative,    // Minimal prefetching
    Aggressive,      // Large prefetch buffers
    Adaptive,        // Learns from scroll patterns
}
```

#### 3.2 WebSocket Communication
```rust
pub struct StreamingServer {
    websocket_server: WebSocketServer,
    display_engine: Arc<DisplayEngine>,
    active_sessions: HashMap<SessionId, StreamingSession>,
}

pub struct StreamingSession {
    data_source: Arc<dyn DataSource>,
    viewport: ViewportWindow, 
    last_update: Instant,
}
```

**Deliverables**:
- ✅ Handle 1M+ row datasets with constant memory usage
- ✅ Sub-100ms response time for data window requests  
- ✅ Smart prefetching based on scroll velocity
- ✅ WebSocket server for real-time communication

### **Phase 4: Interactive Browser Interface (Week 7-8)**

**Goal**: Full browser-based interface with streaming and interactivity

#### 4.1 Python API
```python
# Unified API across all data structures
table = graph.nodes.table()
array = graph.nodes.degree
matrix = graph.adjacency()

# All support the same interface
table.interactive()           # Launch browser interface
array.interactive(theme="dark", port=8080)
matrix.interactive(config=InteractiveConfig(
    max_buffer_size=50000,
    scroll_buffer=100,
    auto_refresh=True
))

# Configuration
config = InteractiveConfig(
    theme="publication",
    max_rows_in_memory=100000,
    websocket_timeout=30,
    enable_filtering=True,
    enable_sorting=True
)
```

#### 4.2 Frontend Architecture
```typescript
// React-based streaming table component
interface StreamingTableProps {
    websocketUrl: string;
    dataSchema: DataSchema;
    theme: ThemeConfig;
    initialData: DataWindow;
}

class StreamingTable extends React.Component<StreamingTableProps> {
    private virtualScroller: VirtualScroller;
    private websocket: WebSocket;
    private dataCache: Map<number, TableRow>;
    
    handleScroll = (scrollTop: number) => {
        const visibleRange = this.calculateVisibleRange(scrollTop);
        this.requestDataIfNeeded(visibleRange);
    }
    
    requestDataIfNeeded = (range: RowRange) => {
        // Smart data fetching with caching
        const missing = this.findMissingRows(range);
        if (missing.length > 0) {
            this.websocket.send({
                type: 'get_rows',
                start: missing[0],
                count: missing.length
            });
        }
    }
}
```

**Deliverables**:
- ✅ One-line `.interactive()` API launches browser
- ✅ Smooth 60fps scrolling through massive datasets
- ✅ Real-time data updates via WebSocket
- ✅ Cross-browser compatibility (Chrome, Firefox, Safari, Edge)

## 🎨 **Visual Design System**

### **Theme Architecture**
```rust
pub struct VisualTheme {
    // Color palette
    pub primary: ColorSet,
    pub secondary: ColorSet, 
    pub background: ColorSet,
    pub text: ColorSet,
    pub borders: ColorSet,
    
    // Typography
    pub fonts: FontStack,
    pub sizes: FontSizes,
    
    // Layout
    pub spacing: SpacingScale,
    pub borders: BorderStyles,
    pub shadows: ShadowStyles,
}

pub enum BuiltInTheme {
    Light,          // Clean light theme
    Dark,           // Professional dark theme  
    Publication,    // Academic paper style
    Minimal,        // Ultra-clean minimal
    HighContrast,   // Accessibility-focused
}
```

### **Responsive Breakpoints**
```css
/* Mobile-first responsive design */
.groggy-display {
    --breakpoint-xs: 480px;   /* Small phones */
    --breakpoint-sm: 768px;   /* Large phones, small tablets */
    --breakpoint-md: 1024px;  /* Tablets, small laptops */
    --breakpoint-lg: 1200px;  /* Desktops */
    --breakpoint-xl: 1600px;  /* Large desktops */
}
```

## 🚀 **Migration Strategy**

### **Backward Compatibility**
```python
# All existing code continues to work unchanged
print(table)                    # Uses new unified display
table._repr_html_()            # Enhanced HTML output
str(array)                     # Unified formatting

# New capabilities are purely additive
table.interactive()            # NEW: Browser interface
table.rich_display(theme="dark")  # NEW: Themed output
```

### **Progressive Enhancement**
1. **Phase 1**: Improve existing `__repr__` and `_repr_html_` methods
2. **Phase 2**: Add `.rich_display()` with theming
3. **Phase 3**: Add `.interactive()` for streaming
4. **Phase 4**: Advanced features (sorting, filtering, etc.)

## 📊 **Success Metrics**

### **Performance Targets**
- **Dataset Size**: Handle 10M+ rows smoothly
- **Memory Usage**: O(viewport_size), not O(total_size)
- **Response Time**: <50ms for window requests
- **Startup Time**: <1s to launch interactive interface
- **Scroll Performance**: 60fps smooth scrolling

### **Quality Targets**  
- **API Simplicity**: One method call to launch (`.interactive()`)
- **Visual Consistency**: Identical styling across all data types
- **Browser Support**: Works in all modern browsers
- **Accessibility**: Full keyboard navigation and screen reader support
- **Mobile Ready**: Touch-friendly responsive design

### **Developer Experience**
- **Zero Config**: Works out-of-the-box with sensible defaults
- **Highly Configurable**: Themes, layouts, performance tuning
- **Framework Agnostic**: Works in Jupyter, VSCode, standalone Python
- **Fast Iteration**: Hot-reload theming and configuration

## 🛠️ **Technical Stack**

### **Backend (Rust)**
```toml
[dependencies]
# Display engine
askama = "0.12"              # HTML templating
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Streaming infrastructure  
tokio-tungstenite = "0.20"   # WebSocket server
tower-http = "0.4"           # Static file serving
hyper = "0.14"               # HTTP server

# Performance
lru = "0.12"                 # LRU cache for data windows
rayon = "1.7"                # Parallel processing
```

### **Frontend (TypeScript + React)**
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0", 
    "react-window": "^1.8.8",        // Virtual scrolling
    "react-virtualized-auto-sizer": "^1.0.20",
    "@tanstack/react-table": "^8.10.0",  // Table utilities
    "styled-components": "^6.0.0",    // CSS-in-JS theming
    "ws": "^8.14.0"                   // WebSocket client
  },
  "devDependencies": {
    "typescript": "^5.2.0",
    "vite": "^4.4.0",                // Fast build tool
    "vitest": "^0.34.0"              // Testing
  }
}
```

### **Build Pipeline**
```yaml
# Unified build system
name: Build Display System
steps:
  - name: Build Rust Core
    run: cargo build --release
    
  - name: Build Frontend Assets  
    run: |
      cd frontend
      npm install
      npm run build
      
  - name: Embed Assets in Binary
    run: cargo build --features embed-assets
    
  - name: Run Integration Tests
    run: |
      cargo test
      npm run test:e2e
```

## 🔮 **Future Extensibility**

### **Plugin Architecture**
```rust
pub trait DisplayPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn supports_data_type(&self, data_type: &DataType) -> bool;
    fn render(&self, data: &DataWindow, config: &DisplayConfig) -> RenderResult;
}

// Example plugins
struct GraphVisualizationPlugin;  // Node-link diagrams
struct HeatmapPlugin;             // Matrix heatmaps  
struct StatisticalSummaryPlugin;  // Statistical overlays
```

### **Integration Points**
```python
# Future AI integration
table.interactive(
    ai_assistant=True,           # Enable AI-powered insights
    auto_insights=True,          # Automatic pattern detection
    smart_filtering=True         # AI-suggested filters
)

# Future time-travel features  
historical_view = table.interactive(
    time_travel=True,            # Enable version history
    commit_browser=True          # Browse graph history
)
```

## 🎨 **Compact Display Format Enhancement**

### **Immediate Priority: Tighter Repr Formatting**

**Problem**: Current display templates justify to full width (e.g., 120 chars), creating unnecessarily wide tables with lots of whitespace.

**Solution**: Use minimum required width based on actual content, with smart truncation for very long values.

### **Current vs. Proposed Approach**

#### **Current (Full Width Distribution)**
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ #  │ name                    │ city                    │ age                     │ score                   │ joined              │
├────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────┤
│  0 │ Alice                   │ NYC                     │                      25 │                   91.50 │ 2024-02-15          │
│  1 │ Bob                     │ Paris                   │                      30 │                   87.00 │ 2023-11-20          │
└────┴─────────────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────┘
```
*Lots of wasted whitespace, unnecessarily wide*

#### **Proposed (Compact Min-Width)**
```
┌───┬───────┬───────┬─────┬───────┬────────────┐
│ # │ name  │ city  │ age │ score │ joined     │
├───┼───────┼───────┼─────┼───────┼────────────┤
│ 0 │ Alice │ NYC   │  25 │ 91.50 │ 2024-02-15 │
│ 1 │ Bob   │ Paris │  30 │ 87.00 │ 2023-11-20 │
└───┴───────┴───────┴─────┴───────┴────────────┘
```
*Compact, only uses space needed for content*

### **Implementation Strategy**

#### **1. New Width Calculation Function**
```python
def calculate_compact_widths(headers: List[str], data: List[List[Any]], max_cell_width: int = 20) -> List[int]:
    """
    Calculate minimum required column widths with optional cell truncation.
    """
    if not headers:
        return []
    
    column_widths = []
    for col_idx, header in enumerate(headers):
        # Get all values in this column
        column_values = [header]  # Include header
        for row in data:
            if col_idx < len(row):
                column_values.append(str(row[col_idx]))
        
        # Find the widest value in this column
        max_width_in_col = max(len(val) for val in column_values) if column_values else 1
        
        # Apply cell truncation limit
        final_width = min(max_width_in_col, max_cell_width)
        final_width = max(final_width, 3)  # At least 3 chars
        
        column_widths.append(final_width)
    
    return column_widths
```

#### **2. Type-Aware Truncation**
```python
def truncate_cell_value(value: Any, max_width: int, dtype: str = 'object') -> str:
    """Smart truncation for individual cell values based on data type."""
    str_value = str(value)
    
    if len(str_value) <= max_width:
        return str_value
    
    # Type-aware truncation strategies
    if dtype in ['string', 'str', 'object']:
        return str_value[:max_width-1] + "…"
    elif dtype in ['float', 'float64', 'float32']:
        # Try reducing precision first
        if isinstance(value, (int, float)):
            for precision in [2, 1, 0]:
                formatted = f"{float(value):.{precision}f}"
                if len(formatted) <= max_width:
                    return formatted
        return f"{float(value):.1e}"  # Scientific notation
    elif dtype in ['int', 'int64', 'int32']:
        if isinstance(value, int) and len(str_value) > max_width:
            return f"{value:.1e}"
        return str_value[:max_width]
    else:
        return str_value[:max_width-1] + "…" if max_width > 1 else str_value[:max_width]
```

#### **3. Updated Display Formatters**
```python
class TableDisplayFormatter:
    def __init__(self, max_rows: int = 10, max_cols: int = 8, 
                 compact: bool = True, max_cell_width: int = 20, precision: int = 2):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.compact = compact          # NEW: Enable compact mode
        self.max_cell_width = max_cell_width  # NEW: Max width per cell
        self.precision = precision

    def format(self, table_data: Dict[str, Any]) -> str:
        # ... existing code ...
        
        if self.compact:
            # NEW: Use compact width calculation
            col_widths = calculate_compact_widths(
                truncated_headers, truncated_data, self.max_cell_width
            )
        else:
            # OLD: Use full-width distribution (backward compatibility)
            col_widths = calculate_column_widths(
                truncated_headers, truncated_data, self.max_width
            )
        
        # ... rest of formatting logic ...
```

### **Configuration Options**
```python
# Global settings
DEFAULT_COMPACT_MODE = True
DEFAULT_MAX_CELL_WIDTH = 20

def configure_display(max_rows=None, max_cols=None, 
                     compact=None, max_cell_width=None, precision=None):
    """Configure global display settings for all data structures."""
    global DEFAULT_COMPACT_MODE, DEFAULT_MAX_CELL_WIDTH
    
    if compact is not None:
        DEFAULT_COMPACT_MODE = compact
    if max_cell_width is not None:
        DEFAULT_MAX_CELL_WIDTH = max_cell_width
    # ... existing parameters ...

# Per-structure configuration
table_formatter = TableDisplayFormatter(
    compact=True,           # Enable tight formatting
    max_cell_width=15,     # Truncate cells longer than 15 chars
    precision=2            # Float precision
)
```

## ✅ **Definition of Done**

### **Core Functionality**
- [ ] All data structures (tables, arrays, matrices) use unified display system
- [ ] **Compact mode enabled by default** with minimal-width formatting
- [ ] **Smart truncation** with type-aware strategies (float precision, string ellipsis)
- [ ] `.interactive()` launches browser interface for all types
- [ ] Virtual scrolling handles 10M+ rows with constant memory usage
- [ ] Professional HTML output with semantic markup
- [ ] 4 built-in themes (light, dark, publication, minimal)

### **Performance & Quality**
- [ ] <50ms response time for data window requests
- [ ] <1s startup time for interactive interface
- [ ] 60fps smooth scrolling performance
- [ ] **Compact repr output** uses only minimum required width
- [ ] Mobile-responsive design
- [ ] WCAG accessibility compliance

### **Developer Experience**
- [ ] Zero-config operation with sensible defaults
- [ ] **Backward compatibility** with `compact=False` option
- [ ] Comprehensive theming and configuration options
- [ ] Complete API documentation with examples
- [ ] Integration tests covering all major browsers
- [ ] Migration guide for updating existing code

---

## 🎯 **Next Steps**

1. **Review and Approve** this unified architecture plan
2. **Phase 1 Kickoff**: Start with compact display formatting (immediate improvement)
3. **Implement Compact Mode**: Update existing formatters with min-width calculations
4. **Create Epic Tasks**: Break down each phase into implementable tasks
5. **Set Up Infrastructure**: Prepare build pipeline and testing environment
6. **Begin Phase 2**: Continue with centralized display system

**Ready to transform static data displays into streaming, interactive experiences! 🚀**

*Starting with compact formatting for immediate improvement, then building toward impossible streaming interfaces.*