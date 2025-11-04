# Visualization Module Refactor Plan

## Core Philosophy: Three Distinct Modes

The viz module should provide three distinct, purpose-built visualization modes:

### 1. `.stream()` - Interactive Real-time Visualization
**Purpose:** Live, interactive exploration with WebSocket streaming
**Was:** `.show()`
**Behavior:**
- Creates/reuses streaming server
- Updates running server with new parameters
- Interactive controls (zoom, pan, filter)
- Real-time layout updates

```python
# Initial streaming viz
g.viz.stream(layout='force', node_color='degree')

# Update existing stream with new parameters
g.viz.stream(node_color='community')  # Smart update

# Explicit server control
g.viz.stream(port=8080)  # Specify port
g.viz.stream(open_browser=True)  # Open in browser vs iframe
```

### 2. `.plot()` - Static Snapshot Visualization  
**Purpose:** Quick visual inspection, embeddable static output
**Behavior:**
- Generates static HTML (with embedded SVG/Canvas, no streaming)
- Or exports PNG/SVG image
- Can be saved to file or displayed inline (Jupyter)
- Uses same styling parameters as stream

```python
# Display inline (Jupyter) or open in browser
g.viz.plot(layout='force', node_color='degree')

# Save to file
g.viz.plot(save='graph.png', layout='force')
g.viz.plot(save='graph.svg', layout='circular')
g.viz.plot(save='graph.html', layout='honeycomb')  # Static HTML

# Quick color visualization
g.viz.plot(node_color='community')  # Auto layout
```

### 3. `.diagram()` - Minimal Explanatory Diagrams
**Purpose:** LaTeX-quality educational/explanatory graphics
**Philosophy:** "The LaTeX of graphs" - minimal, brutalist, clear
**Style:**
- Black and white or minimal color
- Clean lines, clear labels
- Designed for papers, documentation, teaching
- Small, focused, information-dense
- ASCII-art inspired aesthetic option

```python
# Minimal diagram with labels
g.viz.diagram(style='minimal')  # Black & white, clean

# Brutalist ASCII-inspired
g.viz.diagram(style='brutalist')  # Retro terminal aesthetic

# Educational diagram with annotations
g.viz.diagram(
    annotate=['node_degree', 'edge_weight'],
    style='textbook'
)

# Export for documentation
g.viz.diagram(save='concept.svg', dpi=300)  # High-res for papers
g.viz.diagram(save='concept.png', style='retro')  # Pixel-art style

# Small embeddable diagrams
g.viz.diagram(size='small')  # 400x300
g.viz.diagram(size='icon')   # 100x100 for README badges
```

## API Migration Strategy

### Phase 1: Add new methods alongside old (Backward Compat)
```python
# New methods
g.viz.stream()  # New streaming
g.viz.plot()    # New static
g.viz.diagram() # New minimal

# Keep old methods with deprecation warnings
g.viz.show()    # Deprecated → calls stream() with warning
g.viz.server()  # Deprecated → calls stream(open_browser=True) with warning
g.viz.update()  # Deprecated → calls stream() with warning
```

### Phase 2: Update docs and examples
- Update all notebooks to use new API
- Update README and documentation
- Add migration guide

### Phase 3: Remove deprecated methods (Next major version)
- Remove `.show()`, `.server()`, `.update()`
- Clean up tests

## Technical Implementation Plan

### 1. Refactor VizAccessor Structure

**Current:**
```rust
impl VizAccessor {
    fn show() -> PyResult<PyObject>
    fn server() -> PyResult<PyObject>
    fn update() -> PyResult<()>
}
```

**New:**
```rust
impl VizAccessor {
    // === New Primary Methods ===
    fn stream(&self, py: Python, kwargs: Option<&PyDict>) -> PyResult<PyObject>
    fn plot(&self, py: Python, kwargs: Option<&PyDict>) -> PyResult<PyObject>
    fn diagram(&self, py: Python, kwargs: Option<&PyDict>) -> PyResult<PyObject>
    
    // === Deprecated (Phase 1) ===
    #[deprecated(since = "0.6.0", note = "Use stream() instead")]
    fn show(&self, py: Python, kwargs: Option<&PyDict>) -> PyResult<PyObject>
    
    #[deprecated(since = "0.6.0", note = "Use stream(open_browser=True) instead")]
    fn server(&self, py: Python, kwargs: Option<&PyDict>) -> PyResult<PyObject>
    
    #[deprecated(since = "0.6.0", note = "Use stream() to update existing server")]
    fn update(&self, py: Python, kwargs: Option<&PyDict>) -> PyResult<()>
}
```

### 2. Stream Implementation (Smart Update)

```rust
fn stream(&self, py: Python, kwargs: Option<&PyDict>) -> PyResult<PyObject> {
    let (layout, viz_config) = self.parse_viz_kwargs(kwargs)?;
    
    // Check if server already exists for this data source
    if let Some(server_info) = self.get_server_info() {
        // Server exists → UPDATE mode
        self.send_parameter_update(server_info.port, viz_config)?;
        self.display_iframe(py, server_info.port)
    } else {
        // No server → CREATE mode
        self.create_streaming_server(py, layout, viz_config)
    }
}
```

### 3. Plot Implementation (Static Output)

```rust
fn plot(&self, py: Python, kwargs: Option<&PyDict>) -> PyResult<PyObject> {
    let (layout, viz_config) = self.parse_viz_kwargs(kwargs)?;
    let save_path = self.extract_save_param(kwargs)?;
    
    // Generate static layout (no server needed)
    let positions = self.compute_static_layout(layout)?;
    
    match save_path {
        Some(path) if path.ends_with(".png") => {
            self.render_png(positions, viz_config, path)
        }
        Some(path) if path.ends_with(".svg") => {
            self.render_svg(positions, viz_config, path)
        }
        Some(path) if path.ends_with(".html") => {
            self.render_static_html(positions, viz_config, path)
        }
        None => {
            // Display inline (Jupyter) or open in browser
            self.display_static_html(py, positions, viz_config)
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unsupported file format. Use .png, .svg, or .html"
        ))
    }
}
```

### 4. Diagram Implementation (Minimal Style)

```rust
fn diagram(&self, py: Python, kwargs: Option<&PyDict>) -> PyResult<PyObject> {
    let style = self.extract_string_param(kwargs, "style", "minimal")?;
    let size = self.extract_string_param(kwargs, "size", "medium")?;
    let save_path = self.extract_save_param(kwargs)?;
    
    // Use specific diagram styles
    let diagram_config = match style.as_str() {
        "minimal" => DiagramStyle::minimal(),
        "brutalist" => DiagramStyle::brutalist(),
        "textbook" => DiagramStyle::textbook(),
        "retro" => DiagramStyle::retro(),
        _ => DiagramStyle::minimal()
    };
    
    // Generate diagram with minimal aesthetic
    let diagram = self.generate_diagram(diagram_config, size)?;
    
    match save_path {
        Some(path) => self.save_diagram(diagram, path),
        None => self.display_diagram(py, diagram)
    }
}

struct DiagramStyle {
    color_scheme: ColorScheme,
    line_width: f64,
    node_size: f64,
    font_family: String,
    background: String,
}

impl DiagramStyle {
    fn minimal() -> Self {
        Self {
            color_scheme: ColorScheme::BlackWhite,
            line_width: 1.0,
            node_size: 8.0,
            font_family: "monospace".into(),
            background: "white".into(),
        }
    }
    
    fn brutalist() -> Self {
        Self {
            color_scheme: ColorScheme::Terminal,
            line_width: 2.0,
            node_size: 10.0,
            font_family: "Courier New".into(),
            background: "#000".into(),
        }
    }
    
    fn textbook() -> Self {
        Self {
            color_scheme: ColorScheme::Grayscale,
            line_width: 1.5,
            node_size: 12.0,
            font_family: "serif".into(),
            background: "white".into(),
        }
    }
    
    fn retro() -> Self {
        Self {
            color_scheme: ColorScheme::CGA,  // 4-color CGA palette
            line_width: 1.0,
            node_size: 8.0,
            font_family: "monospace".into(),
            background: "#0a0a0a".into(),
        }
    }
}
```

## Styling Philosophy Differences

### Stream: Rich & Interactive
- Full color palettes
- Smooth animations
- Hover effects
- Dynamic sizing
- Real-time updates

### Plot: Practical & Clear
- Standard matplotlib-style colors
- Static but polished
- Good for reports
- Embeddable

### Diagram: Minimal & Conceptual
- Emphasis on structure over aesthetics
- High contrast
- Printable (black & white works)
- Small file sizes
- ASCII-art inspired options
- Terminal-friendly

## Example Usage Patterns

```python
import groggy as gr

g = gr.karate_club()

# === 1. Exploration: Use stream ===
g.viz.stream(layout='force')  # Interactive exploration
g.viz.stream(node_color='club')  # Update with coloring
g.viz.stream(node_size='betweenness')  # Update with sizing

# === 2. Documentation: Use plot ===
g.viz.plot(save='karate_network.png', layout='circular')
g.viz.plot(save='community_viz.svg', node_color='louvain')

# === 3. Teaching/Papers: Use diagram ===
g.viz.diagram(style='minimal', save='concept.svg')
g.viz.diagram(style='textbook', annotate=['degree'], save='figure1.png')
g.viz.diagram(style='brutalist')  # Terminal-aesthetic display

# === 4. Quick inspection ===
g.viz.plot()  # Quick look
g.components()[0].viz.diagram(style='minimal')  # Focused component view
```

## File Organization

```
src/viz/
├── mod.rs                      # Main viz module
├── streaming/                  # Stream backend
│   ├── server.rs
│   ├── websocket.rs
│   └── updates.rs
├── static/                     # Plot backend  
│   ├── renderer.rs
│   ├── html_generator.rs
│   ├── svg_export.rs
│   └── png_export.rs
├── diagram/                    # Diagram backend (NEW)
│   ├── mod.rs
│   ├── styles.rs              # Minimal, brutalist, textbook, retro
│   ├── layout_simple.rs       # Simplified layouts for diagrams
│   └── annotations.rs         # Educational annotations
└── realtime/                   # Shared layout & config
    ├── mod.rs
    └── ...
```

## Migration Timeline

**Week 1: Foundation**
- [ ] Create `diagram/` module structure
- [ ] Implement `DiagramStyle` variants
- [ ] Add `.diagram()` method to `VizAccessor`

**Week 2: Stream Refactor**
- [ ] Rename `.show()` → `.stream()` internally
- [ ] Implement smart update logic
- [ ] Add deprecation warnings to old methods

**Week 3: Plot Enhancement**
- [ ] Implement `.plot()` with save options
- [ ] Add static HTML generation
- [ ] Support inline Jupyter display

**Week 4: Testing & Docs**
- [ ] Update all tests
- [ ] Update documentation
- [ ] Create migration guide
- [ ] Update notebooks

## Benefits of This Refactor

1. **Clarity:** Each method has a clear, single purpose
2. **Discoverability:** Names match user intent (stream, plot, diagram)
3. **Flexibility:** Right tool for the right job
4. **Style:** Diagram mode embodies groggy's minimal philosophy
5. **Education:** Diagram mode perfect for teaching graph concepts
6. **Performance:** Static methods don't need server overhead

## Questions for Consideration

1. Should `diagram()` support color at all, or strictly grayscale?
2. Should we support ASCII art diagrams for terminal output?
3. What default size should diagrams be? (Small for embeds vs large for print)
4. Should `stream()` have a `mode='present'` for full-screen presentations?
5. Should `plot()` support animated GIFs for simple transitions?

Let's discuss and then implement!
