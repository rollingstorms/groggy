//! Visualization bridge for subgraphs
//! Extracts subgraph data into thread-safe structures for visualization

use crate::core::{StreamingDataSchema, StreamingDataWindow};
use crate::subgraphs::Subgraph;
use crate::types::{AttrValue, EdgeId, NodeId};
use crate::viz::display::{ColumnSchema, DataType};
use crate::viz::streaming::data_source::{
    DataSource, GraphEdge, GraphMetadata, GraphNode, LayoutAlgorithm, NodePosition, Position,
};

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
    /// Create from any SubgraphOperations implementor by extracting all data immediately
    pub fn from_subgraph_operations<T: crate::traits::SubgraphOperations>(subgraph: &T) -> Self {
        let graph_ref = subgraph.graph_ref();
        let graph = graph_ref.borrow();

        // Extract nodes
        let mut nodes = Vec::new();
        for &node_id in subgraph.node_set() {
            let attributes = graph.get_node_attrs(node_id).unwrap_or_default();
            let label = attributes
                .get("label")
                .and_then(|v| match v {
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
            if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                let attributes = graph.get_edge_attrs(edge_id).unwrap_or_default();
                let label = attributes.get("label").and_then(|v| match v {
                    AttrValue::Text(s) => Some(s.clone()),
                    _ => None,
                });
                let weight = attributes.get("weight").and_then(|v| match v {
                    AttrValue::Float(f) => Some(*f as f64),
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
            has_weights: edges.iter().any(|e| e.weight.is_some()),
            attribute_types: HashMap::new(),
        };

        let node_count = nodes.len();
        Self {
            nodes,
            edges,
            metadata,
            source_id: format!("subgraph_{}", node_count),
        }
    }

    /// Create from specific Subgraph type (convenience method)
    pub fn from_subgraph(subgraph: &Subgraph) -> Self {
        Self::from_subgraph_operations(subgraph)
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
                    node.label
                        .as_ref()
                        .map(|s| AttrValue::Text(s.clone()))
                        .unwrap_or(AttrValue::Null),
                    AttrValue::Text("node".to_string()),
                    AttrValue::Text(format!("{} attributes", node.attributes.len())),
                ];
                rows.push(row);
            }
        }

        StreamingDataWindow::new(
            vec![
                "id".to_string(),
                "label".to_string(),
                "type".to_string(),
                "attributes".to_string(),
            ],
            rows,
            self.get_schema(),
            self.nodes.len(),
            start,
        )
    }

    fn get_schema(&self) -> StreamingDataSchema {
        StreamingDataSchema {
            columns: vec![
                ColumnSchema {
                    name: "id".to_string(),
                    data_type: DataType::Integer,
                },
                ColumnSchema {
                    name: "label".to_string(),
                    data_type: DataType::String,
                },
                ColumnSchema {
                    name: "type".to_string(),
                    data_type: DataType::String,
                },
                ColumnSchema {
                    name: "attributes".to_string(),
                    data_type: DataType::String,
                },
            ],
            primary_key: Some("id".to_string()),
            source_type: "subgraph".to_string(),
        }
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn get_column_types(&self) -> Vec<DataType> {
        vec![
            DataType::Integer,
            DataType::String,
            DataType::String,
            DataType::String,
        ]
    }

    fn get_column_names(&self) -> Vec<String> {
        vec![
            "id".to_string(),
            "label".to_string(),
            "type".to_string(),
            "attributes".to_string(),
        ]
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
        self.nodes
            .iter()
            .map(|node| GraphNode {
                id: node.id.to_string(),
                label: node.label.clone(),
                attributes: node.attributes.clone(),
                position: None, // Will be set by layout algorithm
            })
            .collect()
    }

    fn get_graph_edges(&self) -> Vec<GraphEdge> {
        self.edges
            .iter()
            .map(|edge| GraphEdge {
                id: edge.id.to_string(),
                source: edge.source.to_string(),
                target: edge.target.to_string(),
                label: edge.label.clone(),
                weight: edge.weight,
                attributes: edge.attributes.clone(),
            })
            .collect()
    }

    fn get_graph_metadata(&self) -> GraphMetadata {
        self.metadata.clone()
    }

    fn compute_layout(&self, algorithm: LayoutAlgorithm) -> Vec<NodePosition> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        match algorithm {
            LayoutAlgorithm::Circular {
                radius,
                start_angle,
            } => {
                let r = radius.unwrap_or(200.0);
                let start = start_angle;
                let angle_step = 2.0 * std::f64::consts::PI / self.nodes.len() as f64;

                self.nodes
                    .iter()
                    .enumerate()
                    .map(|(i, node)| {
                        let angle = start + (i as f64 * angle_step);
                        NodePosition {
                            node_id: node.id.to_string(),
                            position: Position {
                                x: 300.0 + r * angle.cos(),
                                y: 300.0 + r * angle.sin(),
                            },
                        }
                    })
                    .collect()
            }
            LayoutAlgorithm::Grid { columns, cell_size } => self
                .nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    let row = i / columns;
                    let col = i % columns;
                    NodePosition {
                        node_id: node.id.to_string(),
                        position: Position {
                            x: 50.0 + (col as f64 * cell_size),
                            y: 50.0 + (row as f64 * cell_size),
                        },
                    }
                })
                .collect(),
            LayoutAlgorithm::ForceDirected {
                charge: _,
                distance: _,
                iterations: _,
            } => {
                // Basic force-directed layout - simplified version
                let positions: Vec<Position> = self
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let angle =
                            (i as f64 / self.nodes.len() as f64) * 2.0 * std::f64::consts::PI;
                        Position {
                            x: 300.0 + 100.0 * angle.cos(),
                            y: 300.0 + 100.0 * angle.sin(),
                        }
                    })
                    .collect();

                self.nodes
                    .iter()
                    .enumerate()
                    .map(|(i, node)| NodePosition {
                        node_id: node.id.to_string(),
                        position: positions[i],
                    })
                    .collect()
            }
            LayoutAlgorithm::Hierarchical {
                direction: _,
                layer_spacing: _,
                node_spacing: _,
            } => {
                // Default: circular layout
                let radius = 200.0;
                let angle_step = 2.0 * std::f64::consts::PI / self.nodes.len() as f64;

                self.nodes
                    .iter()
                    .enumerate()
                    .map(|(i, node)| {
                        let angle = i as f64 * angle_step;
                        NodePosition {
                            node_id: node.id.to_string(),
                            position: Position {
                                x: 300.0 + radius * angle.cos(),
                                y: 300.0 + radius * angle.sin(),
                            },
                        }
                    })
                    .collect()
            }
            LayoutAlgorithm::Honeycomb {
                cell_size,
                energy_optimization: _,
                iterations: _,
            } => {
                // Honeycomb/hexagonal grid layout
                let sqrt3 = 3.0_f64.sqrt();
                let hex_width = cell_size * sqrt3;
                let hex_height = cell_size * 1.5;

                self.nodes
                    .iter()
                    .enumerate()
                    .map(|(i, node)| {
                        let row = (i as f64 / 10.0).floor() as usize; // 10 nodes per row
                        let col = i % 10;

                        let x_offset = if row % 2 == 1 { hex_width / 2.0 } else { 0.0 };
                        let x = 50.0 + (col as f64 * hex_width) + x_offset;
                        let y = 50.0 + (row as f64 * hex_height);

                        NodePosition {
                            node_id: node.id.to_string(),
                            position: Position { x, y },
                        }
                    })
                    .collect()
            }
        }
    }
}

impl SubgraphDataSource {}
