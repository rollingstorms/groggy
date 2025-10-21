use crate::api::graph::Graph;
use crate::types::{AttrValue, AttrValueType};
use crate::viz::display::{ColumnSchema, DataType};
use crate::viz::layouts::{CircularLayout, ForceDirectedLayout, HoneycombLayout, LayoutEngine};
use crate::viz::streaming::data_source::{
    DataSchema, DataSource, DataWindow, DataWindowMetadata, GraphEdge, GraphMetadata, GraphNode,
    HierarchicalDirection, LayoutAlgorithm, NodePosition, Position,
};
use std::collections::HashMap;

/// DataSource adapter that exposes a `Graph` through the realtime/streaming pipeline.
#[derive(Debug, Clone)]
pub struct GraphDataSource {
    node_count: usize,
    edge_count: usize,
    is_directed: bool,
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    attribute_types: HashMap<String, String>,
}

impl GraphDataSource {
    /// Create a new adapter using the "name" attribute as the preferred label.
    pub fn new(graph: &Graph) -> Self {
        Self::new_with_label(graph, "name")
    }

    /// Create a new adapter using an explicit attribute name for node labels.
    pub fn new_with_label(graph: &Graph, node_label_attr: &str) -> Self {
        let mut attribute_types = HashMap::new();

        // DON'T sort - preserve the graph's original ordering so indices match
        let node_ids_vec = graph.node_ids();

        let nodes = node_ids_vec
            .into_iter()
            .map(|node_id| {
                let mut attributes = HashMap::new();
                if let Ok(node_attrs) = graph.get_node_attrs(node_id) {
                    for (key, value) in node_attrs {
                        record_attribute_type(&mut attribute_types, "node", &key, &value);
                        attributes.insert(key, value);
                    }
                }

                let label = attributes
                    .get(node_label_attr)
                    .and_then(attr_to_string)
                    .unwrap_or_else(|| format!("Node {}", node_id));

                let position = extract_position(&attributes);

                GraphNode {
                    id: node_id.to_string(),
                    label: Some(label),
                    attributes,
                    position,
                }
            })
            .collect::<Vec<_>>();

        // DON'T sort - preserve the graph's original ordering so indices match
        let edge_ids_vec = graph.edge_ids();

        let edges = edge_ids_vec
            .into_iter()
            .filter_map(|edge_id| match graph.edge_endpoints(edge_id) {
                Ok((source, target)) => {
                    let mut attributes = HashMap::new();
                    if let Ok(edge_attrs) = graph.get_edge_attrs(edge_id) {
                        for (key, value) in edge_attrs {
                            record_attribute_type(&mut attribute_types, "edge", &key, &value);
                            attributes.insert(key, value);
                        }
                    }

                    let weight = extract_weight(&attributes);

                    Some(GraphEdge {
                        id: edge_id.to_string(),
                        source: source.to_string(),
                        target: target.to_string(),
                        label: None,
                        weight,
                        attributes,
                    })
                }
                Err(_) => None,
            })
            .collect::<Vec<_>>();

        let node_count = nodes.len();
        let edge_count = edges.len();
        let is_directed = graph.is_directed();

        Self {
            node_count,
            edge_count,
            is_directed,
            nodes,
            edges,
            attribute_types,
        }
    }

    fn graph_nodes(&self) -> Vec<crate::viz::streaming::data_source::GraphNode> {
        self.nodes.clone()
    }

    fn graph_edges(&self) -> Vec<crate::viz::streaming::data_source::GraphEdge> {
        self.edges.clone()
    }

    fn viz_nodes(&self) -> Vec<crate::viz::streaming::data_source::GraphNode> {
        self.nodes.clone()
    }

    fn viz_edges(&self) -> Vec<crate::viz::streaming::data_source::GraphEdge> {
        self.edges.clone()
    }
}

impl DataSource for GraphDataSource {
    fn total_rows(&self) -> usize {
        self.node_count
    }

    fn total_cols(&self) -> usize {
        4 // id, label, x, y
    }

    fn get_window(&self, start: usize, count: usize) -> DataWindow {
        let end = (start + count).min(self.nodes.len());
        let headers = vec![
            "id".to_string(),
            "label".to_string(),
            "x".to_string(),
            "y".to_string(),
        ];

        let rows = self.nodes[start..end]
            .iter()
            .map(|node| {
                let (x, y) = node
                    .position
                    .as_ref()
                    .map(|pos| {
                        (
                            AttrValue::Float(pos.x as f32),
                            AttrValue::Float(pos.y as f32),
                        )
                    })
                    .unwrap_or((AttrValue::Null, AttrValue::Null));

                vec![
                    AttrValue::Text(node.id.clone()),
                    AttrValue::Text(node.label.clone().unwrap_or_default()),
                    x,
                    y,
                ]
            })
            .collect::<Vec<_>>();

        DataWindow {
            headers,
            rows,
            schema: self.get_schema(),
            total_rows: self.nodes.len(),
            start_offset: start,
            metadata: DataWindowMetadata {
                created_at: std::time::SystemTime::now(),
                is_cached: false,
                load_time_ms: 0,
                extra: HashMap::new(),
            },
        }
    }

    fn get_schema(&self) -> DataSchema {
        DataSchema {
            columns: vec![
                ColumnSchema {
                    name: "id".to_string(),
                    data_type: DataType::String,
                },
                ColumnSchema {
                    name: "label".to_string(),
                    data_type: DataType::String,
                },
                ColumnSchema {
                    name: "x".to_string(),
                    data_type: DataType::Float,
                },
                ColumnSchema {
                    name: "y".to_string(),
                    data_type: DataType::Float,
                },
            ],
            primary_key: Some("id".to_string()),
            source_type: "graph".to_string(),
        }
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn get_column_types(&self) -> Vec<DataType> {
        vec![
            DataType::String,
            DataType::String,
            DataType::Float,
            DataType::Float,
        ]
    }

    fn get_column_names(&self) -> Vec<String> {
        vec![
            "id".to_string(),
            "label".to_string(),
            "x".to_string(),
            "y".to_string(),
        ]
    }

    fn get_source_id(&self) -> String {
        format!("graph-{}", std::ptr::addr_of!(*self) as usize)
    }

    fn get_version(&self) -> u64 {
        1
    }

    fn supports_graph_view(&self) -> bool {
        true
    }

    fn get_graph_nodes(&self) -> Vec<GraphNode> {
        self.graph_nodes()
    }

    fn get_graph_edges(&self) -> Vec<GraphEdge> {
        self.graph_edges()
    }

    fn get_graph_metadata(&self) -> GraphMetadata {
        GraphMetadata {
            node_count: self.node_count,
            edge_count: self.edge_count,
            is_directed: self.is_directed,
            has_weights: self.edges.iter().any(|edge| edge.weight.is_some()),
            attribute_types: self.attribute_types.clone(),
        }
    }

    fn compute_layout(&self, algorithm: LayoutAlgorithm) -> Vec<NodePosition> {
        let nodes = self.viz_nodes();
        let edges = self.viz_edges();

        match algorithm {
            LayoutAlgorithm::ForceDirected {
                charge,
                distance,
                iterations,
            } => {
                let layout = ForceDirectedLayout::default()
                    .with_charge(charge)
                    .with_distance(distance)
                    .with_iterations(iterations);

                layout
                    .compute_layout(&nodes, &edges)
                    .map(|positions| {
                        positions
                            .into_iter()
                            .map(|(node_id, position)| NodePosition { node_id, position })
                            .collect()
                    })
                    .unwrap_or_else(|_| fallback_positions(&nodes))
            }
            LayoutAlgorithm::Circular { radius, .. } => {
                let layout = CircularLayout { radius };
                layout
                    .compute_layout(&nodes, &edges)
                    .map(|positions| {
                        positions
                            .into_iter()
                            .map(|(node_id, position)| NodePosition { node_id, position })
                            .collect()
                    })
                    .unwrap_or_else(|_| fallback_positions(&nodes))
            }
            LayoutAlgorithm::Honeycomb {
                cell_size,
                energy_optimization,
                iterations,
            } => {
                let layout = HoneycombLayout {
                    cell_size,
                    energy_optimization,
                    iterations,
                    ..Default::default()
                };

                layout
                    .compute_layout(&nodes, &edges)
                    .map(|positions| {
                        positions
                            .into_iter()
                            .map(|(node_id, position)| NodePosition { node_id, position })
                            .collect()
                    })
                    .unwrap_or_else(|_| fallback_positions(&nodes))
            }
            LayoutAlgorithm::Grid { columns, cell_size } => grid_layout(&nodes, columns, cell_size),
            LayoutAlgorithm::Hierarchical {
                direction,
                layer_spacing,
                node_spacing,
            } => hierarchical_layout(&nodes, direction, layer_spacing, node_spacing),
        }
    }
}

fn record_attribute_type(
    targets: &mut HashMap<String, String>,
    kind: &str,
    key: &str,
    value: &AttrValue,
) {
    let dtype: AttrValueType = value.dtype();
    targets
        .entry(format!("{}.{}", kind, key))
        .or_insert_with(|| format!("{:?}", dtype));
}

fn attr_to_string(value: &AttrValue) -> Option<String> {
    match value {
        AttrValue::Text(text) => Some(text.clone()),
        AttrValue::CompactText(text) => Some(text.as_str().to_string()),
        AttrValue::Int(i) => Some(i.to_string()),
        AttrValue::SmallInt(i) => Some(i.to_string()),
        AttrValue::Float(f) => Some(f.to_string()),
        AttrValue::Bool(b) => Some(b.to_string()),
        AttrValue::Json(json) => Some(json.clone()),
        _ => None,
    }
}

fn extract_position(attributes: &HashMap<String, AttrValue>) -> Option<Position> {
    let x = attributes
        .get("x")
        .and_then(attr_to_f64)
        .or_else(|| attributes.get("pos_x").and_then(attr_to_f64));
    let y = attributes
        .get("y")
        .and_then(attr_to_f64)
        .or_else(|| attributes.get("pos_y").and_then(attr_to_f64));

    match (x, y) {
        (Some(x), Some(y)) => Some(Position { x, y }),
        _ => attributes.get("position").and_then(|attr| match attr {
            AttrValue::FloatVec(values) if values.len() >= 2 => Some(Position {
                x: values[0] as f64,
                y: values[1] as f64,
            }),
            AttrValue::IntVec(values) if values.len() >= 2 => Some(Position {
                x: values[0] as f64,
                y: values[1] as f64,
            }),
            _ => None,
        }),
    }
}

fn extract_weight(attributes: &HashMap<String, AttrValue>) -> Option<f64> {
    attributes
        .get("weight")
        .and_then(attr_to_f64)
        .or_else(|| attributes.get("value").and_then(attr_to_f64))
}

fn attr_to_f64(value: &AttrValue) -> Option<f64> {
    match value {
        AttrValue::Float(v) => Some(*v as f64),
        AttrValue::Int(v) => Some(*v as f64),
        AttrValue::SmallInt(v) => Some(*v as f64),
        _ => None,
    }
}

fn fallback_positions(nodes: &[GraphNode]) -> Vec<NodePosition> {
    nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let angle = i as f64 * 2.0 * std::f64::consts::PI / nodes.len().max(1) as f64;
            let radius = 200.0;
            NodePosition {
                node_id: node.id.clone(),
                position: Position {
                    x: radius * angle.cos(),
                    y: radius * angle.sin(),
                },
            }
        })
        .collect()
}

fn grid_layout(nodes: &[GraphNode], columns: usize, cell_size: f64) -> Vec<NodePosition> {
    if columns == 0 {
        return fallback_positions(nodes);
    }

    nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let col = index % columns;
            let row = index / columns;
            NodePosition {
                node_id: node.id.clone(),
                position: Position {
                    x: col as f64 * cell_size,
                    y: row as f64 * cell_size,
                },
            }
        })
        .collect()
}

fn hierarchical_layout(
    nodes: &[GraphNode],
    direction: HierarchicalDirection,
    layer_spacing: f64,
    node_spacing: f64,
) -> Vec<NodePosition> {
    nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let idx = index as f64;
            let (x, y) = match direction {
                HierarchicalDirection::TopDown => (0.0, idx * layer_spacing),
                HierarchicalDirection::BottomUp => (0.0, -idx * layer_spacing),
                HierarchicalDirection::LeftRight => (idx * layer_spacing, 0.0),
                HierarchicalDirection::RightLeft => (-idx * layer_spacing, 0.0),
            };
            NodePosition {
                node_id: node.id.clone(),
                position: Position {
                    x: x + idx * node_spacing * 0.5,
                    y,
                },
            }
        })
        .collect()
}

// -- Utility trait implementations --------------------------------------------------------------

impl GraphDataSource {
    /// Convenience method for tests and tooling to access node data by ID.
    pub fn node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// Convenience method to access edge data by ID.
    pub fn edge(&self, id: &str) -> Option<&GraphEdge> {
        self.edges.iter().find(|edge| edge.id == id)
    }
}
