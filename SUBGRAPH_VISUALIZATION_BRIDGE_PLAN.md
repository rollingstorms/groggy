# Subgraph Visualization Bridge - Complete Implementation Plan

## Problem Analysis

We have **comprehensive visualization infrastructure** (Phases 1-13 complete) but it's disconnected from subgraphs:

- ✅ **Built**: Streaming server, layout algorithms, themes, static export, interactive features
- ❌ **Missing**: Bridge between subgraphs and the DataSource trait that powers our viz system
- ❌ **Missing**: Working static export (returns NotImplemented)
- ❌ **Missing**: Subgraph.viz() API

## Solution Architecture

Create a **thread-safe data extraction layer** that bridges subgraphs to our existing viz infrastructure.

---

## TODO 1: Create SubgraphDataSource Wrapper

**File**: `src/subgraphs/visualization.rs` (NEW FILE)

**Problem**: Subgraph uses `Rc<RefCell<Graph>>` but DataSource needs `Send + Sync`
**Solution**: Extract data at creation time, store in thread-safe structures

```rust
//! Visualization bridge for subgraphs
//! Extracts subgraph data into thread-safe structures for visualization

use crate::viz::streaming::data_source::{
    DataSource, GraphNode, GraphEdge, GraphMetadata, NodePosition, Position, LayoutAlgorithm
};
use crate::core::{StreamingDataWindow, StreamingDataSchema, DisplayDataWindow, DisplayDataSchema};
use crate::viz::display::{DataType, ColumnSchema};
use crate::types::{NodeId, EdgeId, AttrValue};
use crate::subgraphs::Subgraph;
use std::collections::HashMap;

/// Thread-safe wrapper that implements DataSource for Subgraph data
/// Extracts all data at creation time to avoid Rc<RefCell> threading issues
#[derive(Debug, Clone)]
pub struct SubgraphDataSource {
    // Pre-extracted graph data (thread-safe)
    nodes: Vec<ExtractedNode>,
    edges: Vec<ExtractedEdge>,
    metadata: GraphMetadata,
    source_id: String,
}

#[derive(Debug, Clone)]
struct ExtractedNode {
    id: NodeId,
    label: Option<String>,
    attributes: HashMap<String, AttrValue>,
}

#[derive(Debug, Clone)]
struct ExtractedEdge {
    id: EdgeId,
    source: NodeId,
    target: NodeId,
    label: Option<String>,
    weight: Option<f64>,
    attributes: HashMap<String, AttrValue>,
}

impl SubgraphDataSource {
    /// Create from subgraph by extracting all data immediately
    pub fn from_subgraph(subgraph: &Subgraph) -> Self {
        let graph = subgraph.graph.borrow();
        
        // Extract nodes
        let mut nodes = Vec::new();
        for &node_id in subgraph.node_set() {
            let attributes = graph.get_node_attributes(node_id).unwrap_or_default();
            let label = attributes.get("label")
                .and_then(|v| match v {
                    AttrValue::String(s) => Some(s.clone()),
                    AttrValue::Text(s) => Some(s.clone()),
                    _ => None,
                })
                .or_else(|| Some(format!("Node {}", node_id)));
            
            nodes.push(ExtractedNode {
                id: node_id,
                label,
                attributes,
            });
        }
        
        // Extract edges
        let mut edges = Vec::new();
        for &edge_id in subgraph.edge_set() {
            if let Ok((source, target)) = graph.get_edge_endpoints(edge_id) {
                let attributes = graph.get_edge_attributes(edge_id).unwrap_or_default();
                let label = attributes.get("label")
                    .and_then(|v| match v {
                        AttrValue::String(s) => Some(s.clone()),
                        AttrValue::Text(s) => Some(s.clone()),
                        _ => None,
                    });
                let weight = attributes.get("weight")
                    .and_then(|v| match v {
                        AttrValue::Float(f) => Some(*f),
                        AttrValue::Int(i) => Some(*i as f64),
                        _ => None,
                    });
                
                edges.push(ExtractedEdge {
                    id: edge_id,
                    source,
                    target,
                    label,
                    weight,
                    attributes,
                });
            }
        }
        
        let metadata = GraphMetadata {
            node_count: nodes.len(),
            edge_count: edges.len(),
            is_directed: true, // TODO: Get from graph if available
            subgraph_info: Some(format!("Subgraph: {} nodes, {} edges", nodes.len(), edges.len())),
        };
        
        Self {
            nodes,
            edges,
            metadata,
            source_id: format!("subgraph_{}", subgraph.subgraph_id()),
        }
    }
}

// Implement DataSource trait (thread-safe)
impl DataSource for SubgraphDataSource {
    fn total_rows(&self) -> usize {
        self.nodes.len()
    }
    
    fn total_cols(&self) -> usize {
        4 // id, label, type, attributes
    }
    
    fn get_window(&self, start: usize, count: usize) -> StreamingDataWindow {
        let end = std::cmp::min(start + count, self.nodes.len());
        let mut rows = Vec::new();
        
        for i in start..end {
            if let Some(node) = self.nodes.get(i) {
                let row = vec![
                    AttrValue::Int(node.id as i64),
                    node.label.as_ref().map(|s| AttrValue::String(s.clone())).unwrap_or(AttrValue::Null),
                    AttrValue::String("node".to_string()),
                    AttrValue::String(format!("{} attributes", node.attributes.len())),
                ];
                rows.push(row);
            }
        }
        
        StreamingDataWindow::new(
            vec!["id".to_string(), "label".to_string(), "type".to_string(), "attributes".to_string()],
            rows,
            self.get_schema(),
            self.nodes.len(),
            start,
        )
    }
    
    fn get_schema(&self) -> StreamingDataSchema {
        StreamingDataSchema {
            columns: vec![
                ColumnSchema { name: "id".to_string(), data_type: DataType::Int },
                ColumnSchema { name: "label".to_string(), data_type: DataType::String },
                ColumnSchema { name: "type".to_string(), data_type: DataType::String },
                ColumnSchema { name: "attributes".to_string(), data_type: DataType::String },
            ],
            primary_key: Some("id".to_string()),
            source_type: "subgraph".to_string(),
        }
    }
    
    fn supports_streaming(&self) -> bool { true }
    
    fn get_column_types(&self) -> Vec<DataType> {
        vec![DataType::Int, DataType::String, DataType::String, DataType::String]
    }
    
    fn get_column_names(&self) -> Vec<String> {
        vec!["id".to_string(), "label".to_string(), "type".to_string(), "attributes".to_string()]
    }
    
    fn get_source_id(&self) -> String {
        self.source_id.clone()
    }
    
    fn get_version(&self) -> u64 {
        // Use hash of node/edge counts as version
        (self.nodes.len() as u64) * 1000 + (self.edges.len() as u64)
    }
    
    // GRAPH-SPECIFIC METHODS (the key bridge!)
    fn get_graph_nodes(&self) -> Vec<GraphNode> {
        self.nodes.iter().map(|node| GraphNode {
            id: node.id.to_string(),
            label: node.label.clone(),
            attributes: node.attributes.clone(),
            position: None, // Will be set by layout algorithm
        }).collect()
    }
    
    fn get_graph_edges(&self) -> Vec<GraphEdge> {
        self.edges.iter().map(|edge| GraphEdge {
            id: edge.id.to_string(),
            source: edge.source.to_string(),
            target: edge.target.to_string(),
            label: edge.label.clone(),
            weight: edge.weight,
            attributes: edge.attributes.clone(),
        }).collect()
    }
    
    fn get_graph_metadata(&self) -> GraphMetadata {
        self.metadata.clone()
    }
    
    fn compute_layout(&self, algorithm: LayoutAlgorithm) -> Vec<NodePosition> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        
        match algorithm {
            LayoutAlgorithm::Circular { radius, start_angle } => {
                let r = radius.unwrap_or(200.0);
                let start = start_angle.unwrap_or(0.0);
                let angle_step = 2.0 * std::f64::consts::PI / self.nodes.len() as f64;
                
                self.nodes.iter().enumerate().map(|(i, node)| {
                    let angle = start + (i as f64 * angle_step);
                    NodePosition {
                        node_id: node.id.to_string(),
                        position: Position {
                            x: 300.0 + r * angle.cos(),
                            y: 300.0 + r * angle.sin(),
                        },
                    }
                }).collect()
            },
            LayoutAlgorithm::Grid { columns, cell_size } => {
                self.nodes.iter().enumerate().map(|(i, node)| {
                    let row = i / columns;
                    let col = i % columns;
                    NodePosition {
                        node_id: node.id.to_string(),
                        position: Position {
                            x: 50.0 + (col as f64 * cell_size),
                            y: 50.0 + (row as f64 * cell_size),
                        },
                    }
                }).collect()
            },
            LayoutAlgorithm::ForceDirected { iterations, spring_strength, repulsion_strength } => {
                // Basic force-directed layout
                self.compute_force_directed_layout(iterations.unwrap_or(100))
            },
            _ => {
                // Default: circular layout
                let radius = 200.0;
                let angle_step = 2.0 * std::f64::consts::PI / self.nodes.len() as f64;
                
                self.nodes.iter().enumerate().map(|(i, node)| {
                    let angle = i as f64 * angle_step;
                    NodePosition {
                        node_id: node.id.to_string(),
                        position: Position {
                            x: 300.0 + radius * angle.cos(),
                            y: 300.0 + radius * angle.sin(),
                        },
                    }
                }).collect()
            }
        }
    }
}

impl SubgraphDataSource {
    /// Simple force-directed layout implementation
    fn compute_force_directed_layout(&self, iterations: usize) -> Vec<NodePosition> {
        let mut positions: Vec<Position> = self.nodes.iter().enumerate().map(|(i, _)| {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / self.nodes.len() as f64;
            Position {
                x: 300.0 + 100.0 * angle.cos(),
                y: 300.0 + 100.0 * angle.sin(),
            }
        }).collect();
        
        // Build edge lookup for forces
        let edges: Vec<(usize, usize)> = self.edges.iter().filter_map(|edge| {
            let src_idx = self.nodes.iter().position(|n| n.id == edge.source)?;
            let dst_idx = self.nodes.iter().position(|n| n.id == edge.target)?;
            Some((src_idx, dst_idx))
        }).collect();
        
        // Simple force simulation
        for _iter in 0..iterations {
            let mut forces = vec![Position { x: 0.0, y: 0.0 }; positions.len()];
            
            // Repulsive forces between all nodes
            for i in 0..positions.len() {
                for j in (i + 1)..positions.len() {
                    let dx = positions[j].x - positions[i].x;
                    let dy = positions[j].y - positions[i].y;
                    let dist = (dx * dx + dy * dy).sqrt().max(1.0);
                    let force = 5000.0 / (dist * dist);
                    
                    forces[i].x -= force * dx / dist;
                    forces[i].y -= force * dy / dist;
                    forces[j].x += force * dx / dist;
                    forces[j].y += force * dy / dist;
                }
            }
            
            // Attractive forces for connected nodes
            for &(i, j) in &edges {
                let dx = positions[j].x - positions[i].x;
                let dy = positions[j].y - positions[i].y;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);
                let force = dist * 0.01;
                
                forces[i].x += force * dx / dist;
                forces[i].y += force * dy / dist;
                forces[j].x -= force * dx / dist;
                forces[j].y -= force * dy / dist;
            }
            
            // Apply forces with damping
            for i in 0..positions.len() {
                positions[i].x += forces[i].x * 0.1;
                positions[i].y += forces[i].y * 0.1;
            }
        }
        
        self.nodes.iter().enumerate().map(|(i, node)| {
            NodePosition {
                node_id: node.id.to_string(),
                position: positions[i],
            }
        }).collect()
    }
}
```

---

## TODO 2: Add viz() Method to Subgraph

**File**: `src/subgraphs/subgraph.rs`

**Add to imports:**
```rust
use crate::viz::VizModule;
```

**Add method to impl Subgraph:**
```rust
impl Subgraph {
    // ... existing methods ...
    
    /// Create visualization module for this subgraph
    /// Extracts subgraph data into thread-safe structures and returns VizModule
    pub fn viz(&self) -> VizModule {
        let data_source = crate::subgraphs::visualization::SubgraphDataSource::from_subgraph(self);
        
        use std::sync::Arc;
        let data_source: Arc<dyn crate::viz::streaming::data_source::DataSource> = Arc::new(data_source);
        VizModule::new(data_source)
    }
}
```

**Add to mod.rs:**
```rust
// In src/subgraphs/mod.rs
pub mod visualization;
```

---

## TODO 3: Fix Static Export Implementation

**File**: `src/viz/mod.rs`

**Replace the NotImplemented static_viz method:**

```rust
impl VizModule {
    /// Generate static visualization export (SVG, HTML, PNG)
    pub fn static_viz(&self, options: StaticOptions) -> GraphResult<StaticViz> {
        match options.format {
            ExportFormat::HTML => {
                self.generate_static_html(&options)
            },
            ExportFormat::SVG => {
                self.generate_static_svg(&options)
            },
            ExportFormat::PNG => {
                // For now, generate SVG and suggest conversion
                Err(GraphError::NotImplemented { 
                    feature: "PNG export (use SVG and convert externally)".to_string(),
                    tracking_issue: Some("Convert SVG to PNG using external tools".to_string())
                })
            },
            ExportFormat::PDF => {
                // For now, generate SVG and suggest conversion
                Err(GraphError::NotImplemented { 
                    feature: "PDF export (use SVG and convert externally)".to_string(),
                    tracking_issue: Some("Convert SVG to PDF using external tools".to_string())
                })
            }
        }
    }
    
    /// Generate static HTML with embedded visualization
    fn generate_static_html(&self, options: &StaticOptions) -> GraphResult<StaticViz> {
        use std::fs;
        
        // Get graph data
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        let positions = self.data_source.compute_layout(options.layout.clone());
        let metadata = self.data_source.get_graph_metadata();
        
        // Create HTML with embedded SVG
        let svg_content = self.generate_svg_content(&nodes, &edges, &positions, options);
        
        let html = format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Subgraph Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ margin-bottom: 20px; }}
        .stats {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .stat {{ background: #e3f2fd; padding: 10px; border-radius: 4px; }}
        .viz-container {{ border: 1px solid #ddd; border-radius: 4px; overflow: hidden; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Subgraph Visualization</h1>
            <p>Generated from Groggy subgraph data</p>
        </div>
        <div class="stats">
            <div class="stat"><strong>Nodes:</strong> {}</div>
            <div class="stat"><strong>Edges:</strong> {}</div>
            <div class="stat"><strong>Layout:</strong> {:?}</div>
        </div>
        <div class="viz-container">
            {}
        </div>
    </div>
</body>
</html>"#, metadata.node_count, metadata.edge_count, options.layout, svg_content);
        
        // Write to file
        fs::write(&options.filename, &html)
            .map_err(|e| GraphError::internal(&format!("Failed to write HTML file: {}", e), "generate_static_html"))?;
        
        Ok(StaticViz {
            file_path: options.filename.clone(),
            size_bytes: html.len(),
        })
    }
    
    /// Generate static SVG file
    fn generate_static_svg(&self, options: &StaticOptions) -> GraphResult<StaticViz> {
        use std::fs;
        
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        let positions = self.data_source.compute_layout(options.layout.clone());
        
        let svg = self.generate_svg_content(&nodes, &edges, &positions, options);
        
        // Write to file
        fs::write(&options.filename, &svg)
            .map_err(|e| GraphError::internal(&format!("Failed to write SVG file: {}", e), "generate_static_svg"))?;
        
        Ok(StaticViz {
            file_path: options.filename.clone(),
            size_bytes: svg.len(),
        })
    }
    
    /// Generate SVG content (shared by HTML and SVG export)
    fn generate_svg_content(&self, nodes: &[GraphNode], edges: &[GraphEdge], positions: &[NodePosition], options: &StaticOptions) -> String {
        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">
<defs>
    <style>
        .node {{ fill: #1976d2; stroke: #fff; stroke-width: 2px; cursor: pointer; }}
        .node:hover {{ fill: #1565c0; }}
        .edge {{ stroke: #666; stroke-width: 1.5px; }}
        .label {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; fill: #333; }}
        .edge-label {{ font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; fill: #666; }}
    </style>
</defs>
<rect width="100%" height="100%" fill="#fafafa"/>
"#, options.width, options.height, options.width, options.height);

        // Draw edges first (so they appear behind nodes)
        for edge in edges {
            if let (Some(src_pos), Some(dst_pos)) = (
                positions.iter().find(|p| p.node_id == edge.source),
                positions.iter().find(|p| p.node_id == edge.target)
            ) {
                svg.push_str(&format!(
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="edge"/>"#,
                    src_pos.position.x, src_pos.position.y, 
                    dst_pos.position.x, dst_pos.position.y
                ));
                
                // Add edge label if present
                if let Some(label) = &edge.label {
                    let mid_x = (src_pos.position.x + dst_pos.position.x) / 2.0;
                    let mid_y = (src_pos.position.y + dst_pos.position.y) / 2.0;
                    svg.push_str(&format!(
                        r#"<text x="{}" y="{}" class="edge-label">{}</text>"#,
                        mid_x, mid_y - 5.0, label
                    ));
                }
            }
        }
        
        // Draw nodes
        for pos in positions {
            if let Some(node) = nodes.iter().find(|n| n.id == pos.node_id) {
                svg.push_str(&format!(
                    r#"<circle cx="{}" cy="{}" r="8" class="node" data-id="{}"/>"#,
                    pos.position.x, pos.position.y, node.id
                ));
                
                // Add node label
                if let Some(label) = &node.label {
                    svg.push_str(&format!(
                        r#"<text x="{}" y="{}" class="label">{}</text>"#,
                        pos.position.x, pos.position.y - 15.0, label
                    ));
                }
            }
        }
        
        svg.push_str("</svg>");
        svg
    }
}
```

---

## TODO 4: Add Python viz() Method

**File**: `python-groggy/src/ffi/subgraphs/subgraph.rs`

**Add to imports:**
```rust
use crate::ffi::viz::PyVizModule;
```

**Add to #[pymethods] impl PySubgraph:**
```rust
#[pymethods]
impl PySubgraph {
    // ... existing methods ...
    
    /// Get visualization module for this subgraph
    /// 
    /// Returns a VizModule that can be used for interactive() or static_viz() calls
    /// 
    /// # Example
    /// ```python
    /// # Interactive visualization
    /// session = subgraph.viz().interactive(port=8080)
    /// print(f"Visualization at: {session.url()}")
    /// 
    /// # Static export
    /// result = subgraph.viz().static_viz(filename="subgraph.svg", format="svg")
    /// print(f"Exported to: {result.file_path}")
    /// ```
    fn viz(&self) -> PyResult<PyVizModule> {
        let viz_module = self.inner.viz();
        Ok(PyVizModule {
            inner: viz_module,
        })
    }
}
```

---

## TODO 5: End-to-End Test

**File**: `test_subgraph_visualization_complete.py` (NEW FILE)

```python
#!/usr/bin/env python3
"""
Complete test of subgraph visualization functionality
Tests both interactive and static export capabilities
"""

import groggy as gr
import time
import os

def test_subgraph_viz_complete():
    print("🧪 Testing Complete Subgraph Visualization...")
    
    # Create a test graph
    print("📊 Creating test graph...")
    g = gr.Graph()
    
    # Add nodes with attributes
    alice = g.add_node(label="Alice", type="person", age=30)
    bob = g.add_node(label="Bob", type="person", age=25)
    charlie = g.add_node(label="Charlie", type="person", age=35)
    diana = g.add_node(label="Diana", type="person", age=28)
    
    # Add edges with weights
    g.add_edge(alice, bob, weight=0.8, relationship="friend")
    g.add_edge(bob, charlie, weight=0.6, relationship="colleague")
    g.add_edge(charlie, diana, weight=0.9, relationship="friend")
    g.add_edge(diana, alice, weight=0.7, relationship="neighbor")
    
    print(f"✅ Created graph with {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Create a subgraph (filter for people over 27)
    print("🔍 Creating subgraph...")
    subgraph = g.filter_nodes("age > 27")
    print(f"✅ Created subgraph with {subgraph.node_count()} nodes, {subgraph.edge_count()} edges")
    
    # Test 1: Static SVG Export
    print("\n📤 Testing static SVG export...")
    try:
        result = subgraph.viz().static_viz(
            filename="test_subgraph.svg",
            format="svg",
            layout="circular",
            width=600,
            height=600
        )
        print(f"✅ SVG exported to: {result.file_path}")
        print(f"📦 File size: {result.size_bytes} bytes")
        
        # Verify file exists and has content
        if os.path.exists(result.file_path) and os.path.getsize(result.file_path) > 0:
            print("✅ SVG file created successfully")
            with open(result.file_path, 'r') as f:
                content = f.read()
                if '<svg' in content and '<circle' in content:
                    print("✅ SVG contains expected graph elements")
                else:
                    print("❌ SVG missing graph elements")
        else:
            print("❌ SVG file not created or empty")
            
    except Exception as e:
        print(f"❌ SVG export failed: {e}")
    
    # Test 2: Static HTML Export
    print("\n📤 Testing static HTML export...")
    try:
        result = subgraph.viz().static_viz(
            filename="test_subgraph.html",
            format="html",
            layout="grid",
            width=800,
            height=600
        )
        print(f"✅ HTML exported to: {result.file_path}")
        print(f"📦 File size: {result.size_bytes} bytes")
        
        # Verify file exists and has content
        if os.path.exists(result.file_path) and os.path.getsize(result.file_path) > 0:
            print("✅ HTML file created successfully")
            with open(result.file_path, 'r') as f:
                content = f.read()
                if '<html' in content and '<svg' in content and 'Subgraph Visualization' in content:
                    print("✅ HTML contains expected content and embedded SVG")
                else:
                    print("❌ HTML missing expected content")
        else:
            print("❌ HTML file not created or empty")
            
    except Exception as e:
        print(f"❌ HTML export failed: {e}")
    
    # Test 3: Interactive Visualization
    print("\n🌐 Testing interactive visualization...")
    try:
        session = subgraph.viz().interactive(
            port=8081,
            layout="force_directed",
            auto_open=False  # Don't open browser in test
        )
        print(f"✅ Interactive session started at: {session.url()}")
        
        # Let it run for a moment
        time.sleep(2)
        
        # Check if server is responding
        import requests
        try:
            response = requests.get(session.url(), timeout=5)
            if response.status_code == 200:
                print("✅ Interactive server responding correctly")
            else:
                print(f"❌ Interactive server returned status: {response.status_code}")
        except Exception as e:
            print(f"❌ Interactive server not responding: {e}")
        
        # Stop the session
        session.stop()
        print("✅ Interactive session stopped")
        
    except Exception as e:
        print(f"❌ Interactive visualization failed: {e}")
    
    # Test 4: Different Layouts
    print("\n🎨 Testing different layout algorithms...")
    layouts = ["circular", "grid", "force_directed"]
    for layout in layouts:
        try:
            result = subgraph.viz().static_viz(
                filename=f"test_subgraph_{layout}.svg",
                format="svg",
                layout=layout,
                width=500,
                height=500
            )
            print(f"✅ {layout} layout exported successfully")
        except Exception as e:
            print(f"❌ {layout} layout failed: {e}")
    
    print("\n🎉 Subgraph visualization test complete!")
    print("\nGenerated files:")
    for filename in ["test_subgraph.svg", "test_subgraph.html"] + [f"test_subgraph_{layout}.svg" for layout in layouts]:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  📄 {filename} ({size} bytes)")

if __name__ == "__main__":
    test_subgraph_viz_complete()
```

---

## Implementation Order

1. **Start with TODO 1**: Create the SubgraphDataSource wrapper (solves the threading issue)
2. **Then TODO 2**: Add viz() method to Subgraph (connects to existing infrastructure) 
3. **Then TODO 3**: Fix static export (makes files actually generate)
4. **Then TODO 4**: Add Python FFI support (enables Python API)
5. **Finally TODO 5**: Test everything works end-to-end

This plan reuses **all** our existing visualization infrastructure (streaming server, layout algorithms, themes) and just bridges it to subgraphs through a thread-safe data extraction layer.

Ready to start with TODO 1?