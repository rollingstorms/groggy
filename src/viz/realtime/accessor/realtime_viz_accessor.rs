//! RealtimeVizAccessor trait and DataSource implementation
//!
//! This module provides the bridge between DataSource and Realtime Engine.
//! This is the core of Phase 1 - a clean, testable adapter.

use super::engine_messages::{ControlMsg, Edge, EngineSnapshot, GraphMeta, Node, NodePosition};
use crate::errors::GraphResult;
use crate::types::{AttrValue, EdgeId, NodeId};
use crate::viz::streaming::data_source::{DataSource, LayoutAlgorithm};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Trait for accessing data sources in realtime visualization
pub trait RealtimeVizAccessor: Send + Sync {
    /// Get initial snapshot of all data
    fn initial_snapshot(&self) -> GraphResult<EngineSnapshot>;

    /// Apply control message (optional server-side reactions)
    fn apply_control(&self, control: ControlMsg) -> GraphResult<()>;

    /// Get current node count
    fn node_count(&self) -> usize;

    /// Get current edge count
    fn edge_count(&self) -> usize;

    /// Check if positions are available
    fn has_positions(&self) -> bool;
}

/// Implementation of RealtimeVizAccessor for DataSource
pub struct DataSourceRealtimeAccessor {
    /// DataSource being accessed
    data_source: Arc<dyn DataSource>,
    /// Layout algorithm
    layout_algorithm: RwLock<LayoutAlgorithm>,
    /// Embedding dimensions
    embedding_dimensions: RwLock<usize>,
}

impl DataSourceRealtimeAccessor {
    /// Create new accessor from DataSource
    pub fn new(data_source: Arc<dyn DataSource>) -> Self {
        Self {
            data_source,
            layout_algorithm: RwLock::new(LayoutAlgorithm::ForceDirected {
                charge: -30.0,
                distance: 30.0,
                iterations: 100,
            }),
            embedding_dimensions: RwLock::new(2),
        }
    }

    /// Create new accessor with specific layout
    pub fn with_layout(data_source: Arc<dyn DataSource>, layout: LayoutAlgorithm) -> Self {
        Self {
            data_source,
            layout_algorithm: RwLock::new(layout),
            embedding_dimensions: RwLock::new(2),
        }
    }

    /// Convert DataSource nodes to engine nodes
    fn convert_nodes(&self) -> Vec<Node> {
        eprintln!(
            "🔧 DEBUG: Converting nodes - supports_graph_view: {}",
            self.data_source.supports_graph_view()
        );

        if !self.data_source.supports_graph_view() {
            eprintln!("⚠️  DEBUG: DataSource does not support graph view, returning empty nodes");
            return Vec::new();
        }

        let graph_nodes = self.data_source.get_graph_nodes();
        eprintln!(
            "📊 DEBUG: DataSource returned {} graph nodes",
            graph_nodes.len()
        );

        let engine_nodes: Vec<Node> = graph_nodes
            .into_iter()
            .map(|graph_node| {
                let node_id = graph_node.id.parse().unwrap_or(0);
                eprintln!(
                    "🔧 DEBUG: Converting node '{}' to id {}",
                    graph_node.id, node_id
                );
                Node {
                    id: node_id,
                    attributes: graph_node.attributes,
                }
            })
            .collect();

        eprintln!(
            "✅ DEBUG: Converted {} nodes for engine",
            engine_nodes.len()
        );
        engine_nodes
    }

    /// Convert DataSource edges to engine edges
    fn convert_edges(&self) -> Vec<Edge> {
        eprintln!(
            "🔧 DEBUG: Converting edges - supports_graph_view: {}",
            self.data_source.supports_graph_view()
        );

        if !self.data_source.supports_graph_view() {
            eprintln!("⚠️  DEBUG: DataSource does not support graph view, returning empty edges");
            return Vec::new();
        }

        let graph_edges = self.data_source.get_graph_edges();
        eprintln!(
            "📊 DEBUG: DataSource returned {} graph edges",
            graph_edges.len()
        );

        let engine_edges: Vec<Edge> = graph_edges
            .into_iter()
            .enumerate()
            .map(|(idx, graph_edge)| {
                let source_id = graph_edge.source.parse().unwrap_or(0);
                let target_id = graph_edge.target.parse().unwrap_or(0);
                eprintln!(
                    "🔧 DEBUG: Converting edge {} '{}' -> '{}' to {} -> {}",
                    idx, graph_edge.source, graph_edge.target, source_id, target_id
                );
                Edge {
                    id: idx as EdgeId,
                    source: source_id,
                    target: target_id,
                    attributes: graph_edge.attributes,
                }
            })
            .collect();

        eprintln!(
            "✅ DEBUG: Converted {} edges for engine",
            engine_edges.len()
        );
        engine_edges
    }

    /// Generate or compute positions for nodes
    fn compute_positions(&self, nodes: &[Node]) -> Vec<NodePosition> {
        if nodes.is_empty() {
            return Vec::new();
        }

        // Try to use DataSource layout computation
        let positions = self
            .data_source
            .compute_layout(self.layout_algorithm.read().unwrap().clone());

        if !positions.is_empty() {
            // Convert DataSource positions to engine positions
            positions
                .into_iter()
                .map(|pos| NodePosition {
                    node_id: pos.node_id.parse().unwrap_or(0),
                    coords: vec![pos.position.x, pos.position.y],
                })
                .collect()
        } else {
            // Generate default positions if DataSource can't provide layout
            self.generate_default_positions(nodes)
        }
    }

    /// Generate default positions for nodes (fallback)
    fn generate_default_positions(&self, nodes: &[Node]) -> Vec<NodePosition> {
        use std::f64::consts::PI;

        nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let angle = 2.0 * PI * i as f64 / nodes.len() as f64;
                let radius = 100.0;
                NodePosition {
                    node_id: node.id,
                    coords: vec![radius * angle.cos(), radius * angle.sin()],
                }
            })
            .collect()
    }

    /// Create graph metadata
    fn create_meta(&self, node_count: usize, edge_count: usize, has_positions: bool) -> GraphMeta {
        let ds_meta = self.data_source.get_graph_metadata();

        GraphMeta {
            node_count,
            edge_count,
            dimensions: *self.embedding_dimensions.read().unwrap(),
            layout_method: format!("{:?}", *self.layout_algorithm.read().unwrap()),
            embedding_method: "default".to_string(),
            has_positions,
        }
    }
}

impl RealtimeVizAccessor for DataSourceRealtimeAccessor {
    fn initial_snapshot(&self) -> GraphResult<EngineSnapshot> {
        eprintln!("🔧 DEBUG: DataSourceRealtimeAccessor creating initial snapshot");

        // Convert DataSource data to engine format
        let nodes = self.convert_nodes();
        let edges = self.convert_edges();
        let positions = self.compute_positions(&nodes);
        let has_positions = !positions.is_empty();
        let meta = self.create_meta(nodes.len(), edges.len(), has_positions);

        eprintln!(
            "✅ DEBUG: Snapshot created - {} nodes, {} edges, {} positions",
            nodes.len(),
            edges.len(),
            positions.len()
        );

        Ok(EngineSnapshot {
            nodes,
            edges,
            positions,
            meta,
        })
    }

    fn apply_control(&self, control: ControlMsg) -> GraphResult<()> {
        eprintln!(
            "🎮 DEBUG: DataSourceRealtimeAccessor received control: {:?}",
            control
        );

        match control {
            ControlMsg::ChangeEmbedding { method, k, params } => {
                *self.embedding_dimensions.write().unwrap() = k;
                eprintln!(
                    "📐 DEBUG: Updated embedding to {} with {} dimensions",
                    method, k
                );

                if method == "rotation" {
                    let rotation_x = params
                        .get("rotation_x")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    let rotation_y = params
                        .get("rotation_y")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);

                    eprintln!(
                        "🔄 DEBUG: Rotation control acknowledged (x={}, y={})",
                        rotation_x, rotation_y
                    );
                }

                // Engine is authoritative for runtime updates; accessor only acknowledges control inputs.
            }
            ControlMsg::ChangeLayout { algorithm, params } => {
                eprintln!(
                    "🎯 DEBUG: Layout control received for {} with params {:?}",
                    algorithm, params
                );

                *self.layout_algorithm.write().unwrap() = match algorithm.as_str() {
                    "honeycomb" => LayoutAlgorithm::Honeycomb {
                        cell_size: params
                            .get("cell_size")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(40.0),
                        energy_optimization: params
                            .get("energy_optimization")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(true),
                        iterations: params
                            .get("iterations")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(500),
                    },
                    "force_directed" => LayoutAlgorithm::ForceDirected {
                        charge: params
                            .get("charge")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(-30.0),
                        distance: params
                            .get("distance")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(30.0),
                        iterations: params
                            .get("iterations")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(100),
                    },
                    _ => self.layout_algorithm.read().unwrap().clone(),
                };

                // Engine recomputes layout and will stream envelopes downstream.
            }
            _ => {
                eprintln!(
                    "⚠️  DEBUG: Accessor received unsupported control message: {:?}",
                    control
                );
            }
        }

        Ok(())
    }

    fn node_count(&self) -> usize {
        if self.data_source.supports_graph_view() {
            self.data_source.get_graph_metadata().node_count
        } else {
            self.data_source.total_rows()
        }
    }

    fn edge_count(&self) -> usize {
        if self.data_source.supports_graph_view() {
            self.data_source.get_graph_metadata().edge_count
        } else {
            0
        }
    }

    fn has_positions(&self) -> bool {
        // Check if DataSource can provide layout
        !self
            .data_source
            .compute_layout(self.layout_algorithm.read().unwrap().clone())
            .is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::display::{ColumnSchema, DataType};
    use crate::viz::streaming::data_source::{
        DataSchema, DataWindow, GraphEdge, GraphMetadata, GraphNode,
    };
    use std::collections::HashMap;

    /// Mock DataSource for testing
    #[derive(Debug)]
    struct MockDataSource {
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
    }

    impl MockDataSource {
        fn new() -> Self {
            let mut node1_attrs = HashMap::new();
            node1_attrs.insert("label".to_string(), AttrValue::Text("Node 1".to_string()));

            let mut node2_attrs = HashMap::new();
            node2_attrs.insert("label".to_string(), AttrValue::Text("Node 2".to_string()));

            let mut edge_attrs = HashMap::new();
            edge_attrs.insert("weight".to_string(), AttrValue::Float(1.0));

            Self {
                nodes: vec![
                    GraphNode {
                        id: "0".to_string(),
                        label: Some("Node 1".to_string()),
                        attributes: node1_attrs,
                        position: None,
                    },
                    GraphNode {
                        id: "1".to_string(),
                        label: Some("Node 2".to_string()),
                        attributes: node2_attrs,
                        position: None,
                    },
                ],
                edges: vec![GraphEdge {
                    id: "0".to_string(),
                    source: "0".to_string(),
                    target: "1".to_string(),
                    label: Some("Edge 1".to_string()),
                    weight: Some(1.0),
                    attributes: edge_attrs,
                }],
            }
        }
    }

    impl DataSource for MockDataSource {
        fn total_rows(&self) -> usize {
            self.nodes.len()
        }
        fn total_cols(&self) -> usize {
            3
        }
        fn supports_streaming(&self) -> bool {
            true
        }
        fn supports_graph_view(&self) -> bool {
            true
        }

        fn get_window(&self, _start: usize, _count: usize) -> DataWindow {
            DataWindow::new(
                vec!["id".to_string(), "label".to_string()],
                vec![],
                DataSchema {
                    columns: vec![],
                    primary_key: Some("id".to_string()),
                    source_type: "mock".to_string(),
                },
                self.nodes.len(),
                0,
            )
        }

        fn get_schema(&self) -> DataSchema {
            DataSchema {
                columns: vec![],
                primary_key: Some("id".to_string()),
                source_type: "mock".to_string(),
            }
        }

        fn get_column_types(&self) -> Vec<DataType> {
            vec![DataType::Integer, DataType::Text]
        }

        fn get_column_names(&self) -> Vec<String> {
            vec!["id".to_string(), "label".to_string()]
        }

        fn get_source_id(&self) -> String {
            "mock_source".to_string()
        }
        fn get_version(&self) -> u64 {
            1
        }

        fn get_graph_nodes(&self) -> Vec<GraphNode> {
            self.nodes.clone()
        }

        fn get_graph_edges(&self) -> Vec<GraphEdge> {
            self.edges.clone()
        }

        fn get_graph_metadata(&self) -> GraphMetadata {
            GraphMetadata {
                node_count: self.nodes.len(),
                edge_count: self.edges.len(),
                is_directed: false,
                has_weights: true,
                attribute_types: HashMap::new(),
            }
        }

        fn compute_layout(
            &self,
            _algorithm: LayoutAlgorithm,
        ) -> Vec<crate::viz::streaming::data_source::NodePosition> {
            vec![]
        }
    }

    #[test]
    fn test_initial_snapshot() {
        let mock_ds = Arc::new(MockDataSource::new());
        let accessor = DataSourceRealtimeAccessor::new(mock_ds);

        let snapshot = accessor.initial_snapshot().unwrap();

        assert_eq!(snapshot.nodes.len(), 2);
        assert_eq!(snapshot.edges.len(), 1);
        assert_eq!(snapshot.positions.len(), 2); // Default positions generated
        assert_eq!(snapshot.meta.node_count, 2);
        assert_eq!(snapshot.meta.edge_count, 1);
    }

    #[test]
    fn test_control_messages() {
        let mock_ds = Arc::new(MockDataSource::new());
        let mut accessor = DataSourceRealtimeAccessor::new(mock_ds);

        // Test embedding change
        let control = ControlMsg::ChangeEmbedding {
            method: "pca".to_string(),
            k: 3,
            params: HashMap::new(),
        };

        accessor.apply_control(control).unwrap();
        assert_eq!(accessor.embedding_dimensions, 3);
    }
}
