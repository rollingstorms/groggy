# Viz Module Implementation Roadmap

## Current Status

✅ **What We Have**:
- Complete Python API layer (`python-groggy/python/groggy/viz.py`)
- Complete Rust visualization module (`src/viz/mod.rs`)
- Complete FFI bindings (`python-groggy/src/ffi/viz/mod.rs`)
- Working DataSource trait and layout algorithms
- Streaming server infrastructure
- Frontend HTML/CSS/JavaScript files

❌ **What's Missing (The Critical Gaps)**:
1. **Graph.viz() method** - No way to access VizModule from Graph
2. **Graph → DataSource implementation** - Graph doesn't implement DataSource trait
3. **FFI registration** - VizModule classes not exposed to Python
4. **Static export implementation** - Returns NotImplemented error

## Exact Implementation Plan

### Step 1: Add Graph.viz() Method [CRITICAL]

**File**: `src/api/graph.rs`
**Action**: Add viz() method that returns VizModule

```rust
impl Graph {
    /// Get visualization module for this graph
    pub fn viz(&self) -> crate::viz::VizModule {
        use crate::viz::VizModule;
        use std::sync::Arc;
        
        // Create DataSource from current graph
        let data_source = Arc::new(GraphDataSource::new(self));
        VizModule::new(data_source)
    }
}
```

### Step 2: Implement DataSource for Graph [CRITICAL]

**File**: `src/api/graph.rs` (add to end of file)
**Action**: Create GraphDataSource wrapper

```rust
use crate::viz::streaming::data_source::{DataSource, GraphNode, GraphEdge, GraphMetadata, DataWindow, DataSchema, NodePosition, LayoutAlgorithm, Position};

/// DataSource implementation for Graph
#[derive(Debug)]
struct GraphDataSource {
    // Store graph reference
    node_count: usize,
    edge_count: usize,
    nodes: Vec<(NodeId, HashMap<String, AttrValue>)>,
    edges: Vec<(EdgeId, NodeId, NodeId, HashMap<String, AttrValue>)>,
}

impl GraphDataSource {
    fn new(graph: &Graph) -> Self {
        // Extract current graph data
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();
        
        // Get all nodes with attributes
        let mut nodes = Vec::new();
        for node_id in graph.all_nodes() {
            let attrs = graph.get_all_node_attrs(node_id).unwrap_or_default();
            nodes.push((node_id, attrs));
        }
        
        // Get all edges with attributes  
        let mut edges = Vec::new();
        for edge_id in graph.all_edges() {
            if let Ok((src, dst)) = graph.edge_endpoints(edge_id) {
                let attrs = graph.get_all_edge_attrs(edge_id).unwrap_or_default();
                edges.push((edge_id, src, dst, attrs));
            }
        }
        
        Self { node_count, edge_count, nodes, edges }
    }
}

impl DataSource for GraphDataSource {
    fn total_rows(&self) -> usize { self.node_count }
    fn total_cols(&self) -> usize { 4 } // id, label, attrs, type
    fn supports_graph_view(&self) -> bool { true }
    
    fn get_graph_nodes(&self) -> Vec<GraphNode> {
        self.nodes.iter().map(|(id, attrs)| {
            GraphNode {
                id: id.to_string(),
                label: attrs.get("label").and_then(|v| v.as_text()).cloned(),
                attributes: attrs.clone(),
                position: None,
            }
        }).collect()
    }
    
    fn get_graph_edges(&self) -> Vec<GraphEdge> {
        self.edges.iter().map(|(id, src, dst, attrs)| {
            GraphEdge {
                id: id.to_string(),
                source: src.to_string(),
                target: dst.to_string(),
                label: attrs.get("label").and_then(|v| v.as_text()).cloned(),
                weight: attrs.get("weight").and_then(|v| v.as_float()).copied(),
                attributes: attrs.clone(),
            }
        }).collect()
    }
    
    fn get_graph_metadata(&self) -> GraphMetadata {
        GraphMetadata {
            node_count: self.node_count,
            edge_count: self.edge_count,
            is_directed: true, // TODO: Get from graph
            has_weights: true, // TODO: Check actual edges
            attribute_types: HashMap::new(),
        }
    }
    
    // Required DataSource methods
    fn get_window(&self, start: usize, count: usize) -> DataWindow {
        // Convert nodes to tabular format for streaming
        let headers = vec!["id".to_string(), "label".to_string()];
        let rows: Vec<Vec<AttrValue>> = self.nodes.iter()
            .skip(start)
            .take(count)
            .map(|(id, attrs)| vec![
                AttrValue::Int(*id as i64),
                attrs.get("label").cloned().unwrap_or(AttrValue::Text("".to_string()))
            ])
            .collect();
            
        DataWindow::new(headers, rows, self.get_schema(), self.node_count, start)
    }
    
    fn get_schema(&self) -> DataSchema {
        DataSchema {
            columns: vec![
                ColumnSchema { name: "id".to_string(), data_type: DataType::Integer },
                ColumnSchema { name: "label".to_string(), data_type: DataType::String },
            ],
            primary_key: Some("id".to_string()),
            source_type: "graph".to_string(),
        }
    }
    
    fn supports_streaming(&self) -> bool { true }
    fn get_column_types(&self) -> Vec<DataType> { vec![DataType::Integer, DataType::String] }
    fn get_column_names(&self) -> Vec<String> { vec!["id".to_string(), "label".to_string()] }
    fn get_source_id(&self) -> String { "graph".to_string() }
    fn get_version(&self) -> u64 { 1 }
}
```

### Step 3: Fix FFI Module Registration [CRITICAL]

**File**: `python-groggy/src/ffi/mod.rs`
**Action**: Add viz module to Python registration

```rust
// Add this to the module registration:
pub mod viz;

// In the Python module function:
#[pymodule]
fn _groggy(py: Python, m: &PyModule) -> PyResult<()> {
    // ... existing registrations ...
    
    // Add viz classes
    m.add_class::<viz::PyVizConfig>()?;
    m.add_class::<viz::PyVizModule>()?;
    m.add_class::<viz::PyInteractiveViz>()?;
    m.add_class::<viz::PyInteractiveVizSession>()?;
    m.add_class::<viz::PyStaticViz>()?;
    
    Ok(())
}
```

### Step 4: Add viz() Method to FFI Graph [CRITICAL]

**File**: `python-groggy/src/ffi/api/graph.rs`
**Action**: Add viz() method to PyGraph

```rust
#[pymethods]
impl PyGraph {
    // ... existing methods ...
    
    /// Get visualization module for this graph
    pub fn viz(&self) -> PyResult<crate::ffi::viz::PyVizModule> {
        let viz_module = self.inner.viz();
        Ok(crate::ffi::viz::PyVizModule { inner: viz_module })
    }
}
```

### Step 5: Implement Static Export [HIGH PRIORITY]

**File**: `src/viz/mod.rs`
**Action**: Replace NotImplemented with basic SVG export

```rust
pub fn static_viz(&self, options: StaticOptions) -> GraphResult<StaticViz> {
    use std::fs;
    use std::io::Write;
    
    match options.format {
        ExportFormat::SVG => {
            // Generate basic SVG
            let nodes = self.data_source.get_graph_nodes();
            let edges = self.data_source.get_graph_edges();
            let positions = self.data_source.compute_layout(options.layout);
            
            let svg = generate_svg(&nodes, &edges, &positions, &options);
            
            // Write to file
            let mut file = fs::File::create(&options.filename)?;
            file.write_all(svg.as_bytes())?;
            
            Ok(StaticViz {
                file_path: options.filename,
                size_bytes: svg.len(),
            })
        },
        _ => Err(GraphError::NotImplemented { 
            feature: "PNG/PDF export".to_string(),
            tracking_issue: None
        })
    }
}

fn generate_svg(nodes: &[GraphNode], edges: &[GraphEdge], positions: &[NodePosition], options: &StaticOptions) -> String {
    let mut svg = format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
        options.width, options.height
    );
    
    // Draw edges
    for edge in edges {
        let src_pos = positions.iter().find(|p| p.node_id == edge.source);
        let dst_pos = positions.iter().find(|p| p.node_id == edge.target);
        
        if let (Some(src), Some(dst)) = (src_pos, dst_pos) {
            svg.push_str(&format!(
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="1"/>"#,
                src.position.x, src.position.y, dst.position.x, dst.position.y
            ));
        }
    }
    
    // Draw nodes
    for (node, pos) in nodes.iter().zip(positions.iter()) {
        svg.push_str(&format!(
            r#"<circle cx="{}" cy="{}" r="5" fill="blue"/>"#,
            pos.position.x, pos.position.y
        ));
        
        if let Some(label) = &node.label {
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" font-size="12">{}</text>"#,
                pos.position.x + 7, pos.position.y + 3, label
            ));
        }
    }
    
    svg.push_str("</svg>");
    svg
}
```

### Step 6: Implement Basic Layout Algorithm [HIGH PRIORITY]

**File**: `src/api/graph.rs` (add to GraphDataSource)
**Action**: Add compute_layout implementation

```rust
impl DataSource for GraphDataSource {
    // ... existing methods ...
    
    fn compute_layout(&self, algorithm: LayoutAlgorithm) -> Vec<NodePosition> {
        match algorithm {
            LayoutAlgorithm::Circular { radius, start_angle } => {
                let radius = radius.unwrap_or(100.0);
                let node_count = self.nodes.len();
                
                self.nodes.iter().enumerate().map(|(i, (id, _))| {
                    let angle = start_angle + (i as f64 * 2.0 * std::f64::consts::PI / node_count as f64);
                    NodePosition {
                        node_id: id.to_string(),
                        position: Position {
                            x: 200.0 + radius * angle.cos(),
                            y: 200.0 + radius * angle.sin(),
                        },
                    }
                }).collect()
            },
            LayoutAlgorithm::Grid { columns, cell_size } => {
                self.nodes.iter().enumerate().map(|(i, (id, _))| {
                    let col = i % columns;
                    let row = i / columns;
                    NodePosition {
                        node_id: id.to_string(),
                        position: Position {
                            x: col as f64 * cell_size,
                            y: row as f64 * cell_size,
                        },
                    }
                }).collect()
            },
            _ => {
                // Simple random layout for other algorithms
                use rand::Rng;
                let mut rng = rand::thread_rng();
                
                self.nodes.iter().map(|(id, _)| {
                    NodePosition {
                        node_id: id.to_string(),
                        position: Position {
                            x: rng.gen_range(50.0..350.0),
                            y: rng.gen_range(50.0..350.0),
                        },
                    }
                }).collect()
            }
        }
    }
}
```

## Implementation Order

### Phase 1: Core Integration (1-2 hours)
1. Add GraphDataSource implementation to `src/api/graph.rs`
2. Add Graph.viz() method to `src/api/graph.rs`
3. Add viz() FFI method to `python-groggy/src/ffi/api/graph.rs`
4. Register viz classes in `python-groggy/src/ffi/mod.rs`

### Phase 2: Basic Functionality (2-3 hours)
5. Implement basic SVG export in `src/viz/mod.rs`
6. Implement circular and grid layout algorithms
7. Fix any compilation errors and missing imports

### Phase 3: Testing (1 hour)
8. Build and test with `maturin develop`
9. Run Python integration tests
10. Verify end-to-end workflow works

## Expected Result

After implementation, this should work:

```python
import groggy as gr

# Create graph
g = gr.Graph()
node_a = g.add_node(label="Alice")
node_b = g.add_node(label="Bob")
g.add_edge(node_a, node_b, weight=1.0)

# Interactive visualization
session = g.viz().interactive(port=8080, layout="circular")
print(f"Visualization at: {session.url()}")
session.stop()

# Static export  
result = g.viz().static("test.svg", format="svg", layout="circular")
print(f"Exported to: {result.file_path}")
```

## Critical Dependencies

All required Rust modules already exist:
- ✅ `src/viz/mod.rs` - Main viz module
- ✅ `src/viz/streaming/` - Streaming infrastructure
- ✅ `python-groggy/src/ffi/viz/mod.rs` - FFI bindings
- ✅ `python-groggy/python/groggy/viz.py` - Python API

The only missing pieces are the connections between them.