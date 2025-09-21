use super::server::StreamingServer;
use super::types::{
    WSMessage, StreamingResult, StreamingError,
    NodeAnalytics, CentralityMeasures, TooltipAttribute, AttributeDisplayType,
    NodeTooltipData, TooltipMetric, MetricFormat, EdgeTooltipData,
    SelectionAnalytics, BoundingBox, SelectionType
};

// --- Domain-specific request handlers ---
impl StreamingServer {
    pub async fn handle_node_click_request(&self, node_id: &str) -> StreamingResult<WSMessage> {
        if !self.data_source.supports_graph_view() {
            return Ok(WSMessage::Error {
                message: "Graph view not supported".to_string(),
                error_code: "NO_GRAPH_SUPPORT".to_string(),
            });
        }
        
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        
        // Find the clicked node
        let clicked_node = match nodes.iter().find(|n| n.id == node_id) {
            Some(node) => node,
            None => return Ok(WSMessage::Error {
                message: format!("Node '{}' not found", node_id),
                error_code: "NODE_NOT_FOUND".to_string(),
            }),
        };
        
        // Find connected neighbors and edges
        let mut neighbors = Vec::new();
        let mut connected_edges = Vec::new();
        
        for edge in &edges {
            if edge.source == node_id || edge.target == node_id {
                connected_edges.push(edge.into());
                
                // Add neighbor node
                let neighbor_id = if edge.source == node_id { &edge.target } else { &edge.source };
                if let Some(neighbor) = nodes.iter().find(|n| n.id == *neighbor_id) {
                    neighbors.push(neighbor.into());
                }
            }
        }
        
        // Compute basic analytics for the node
        let analytics = NodeAnalytics {
            degree: connected_edges.len(),
            in_degree: Some(edges.iter().filter(|e| e.target == node_id).count()),
            out_degree: Some(edges.iter().filter(|e| e.source == node_id).count()),
            centrality_measures: CentralityMeasures {
                betweenness: None, // TODO: Implement centrality calculations
                closeness: None,
                eigenvector: None,
                page_rank: None,
            },
            clustering_coefficient: None, // TODO: Implement clustering coefficient
            community_id: None, // TODO: Implement community detection
        };
        
        Ok(WSMessage::NodeClickResponse {
            node_id: node_id.to_string(),
            node_data: clicked_node.into(),
            neighbors,
            connected_edges,
            analytics,
        })
    }

    pub async fn handle_node_hover_request(&self, node_id: &str) -> StreamingResult<WSMessage> {
        if !self.data_source.supports_graph_view() {
            return Ok(WSMessage::Error {
                message: "Graph view not supported".to_string(),
                error_code: "NO_GRAPH_SUPPORT".to_string(),
            });
        }
        
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        
        let hovered_node = match nodes.iter().find(|n| n.id == node_id) {
            Some(node) => node,
            None => return Ok(WSMessage::Error {
                message: format!("Node '{}' not found", node_id),
                error_code: "NODE_NOT_FOUND".to_string(),
            }),
        };
        
        // Create rich tooltip data
        let degree = edges.iter().filter(|e| e.source == node_id || e.target == node_id).count();
        
        let mut primary_attributes = Vec::new();
        let mut secondary_attributes = Vec::new();
        
        // Add key attributes as primary
        if let Some(label) = &hovered_node.label {
            primary_attributes.push(TooltipAttribute {
                name: "Label".to_string(),
                value: label.clone(),
                display_type: AttributeDisplayType::Text,
            });
        }
        
        // Add other attributes as secondary
        for (key, value) in &hovered_node.attributes {
            secondary_attributes.push(TooltipAttribute {
                name: key.clone(),
                value: format!("{:?}", value), // Basic formatting
                display_type: AttributeDisplayType::Text,
            });
        }
        
        let tooltip_data = NodeTooltipData {
            title: hovered_node.label.clone().unwrap_or_else(|| node_id.to_string()),
            subtitle: Some(format!("Node ID: {}", node_id)),
            primary_attributes,
            secondary_attributes,
            metrics: vec![
                TooltipMetric {
                    name: "Degree".to_string(),
                    value: degree as f64,
                    format: MetricFormat::Integer,
                    context: None,
                }
            ],
        };
        
        Ok(WSMessage::NodeHoverResponse {
            node_id: node_id.to_string(),
            tooltip_data,
        })
    }

    pub async fn handle_edge_click_request(&self, edge_id: &str) -> StreamingResult<WSMessage> {
        if !self.data_source.supports_graph_view() {
            return Ok(WSMessage::Error {
                message: "Graph view not supported".to_string(),
                error_code: "NO_GRAPH_SUPPORT".to_string(),
            });
        }
        
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        
        let clicked_edge = match edges.iter().find(|e| e.id == edge_id) {
            Some(edge) => edge,
            None => return Ok(WSMessage::Error {
                message: format!("Edge '{}' not found", edge_id),
                error_code: "EDGE_NOT_FOUND".to_string(),
            }),
        };
        
        // Find source and target nodes
        let source_node = nodes.iter().find(|n| n.id == clicked_edge.source)
            .ok_or_else(|| StreamingError::Client(format!("Source node '{}' not found", clicked_edge.source)))?;
        let target_node = nodes.iter().find(|n| n.id == clicked_edge.target)
            .ok_or_else(|| StreamingError::Client(format!("Target node '{}' not found", clicked_edge.target)))?;
        
        // TODO: Compute path information (shortest path analysis)
        let path_info = None;
        
        Ok(WSMessage::EdgeClickResponse {
            edge_id: edge_id.to_string(),
            edge_data: clicked_edge.into(),
            source_node: source_node.into(),
            target_node: target_node.into(),
            path_info,
        })
    }

    pub async fn handle_edge_hover_request(&self, edge_id: &str) -> StreamingResult<WSMessage> {
        if !self.data_source.supports_graph_view() {
            return Ok(WSMessage::Error {
                message: "Graph view not supported".to_string(),
                error_code: "NO_GRAPH_SUPPORT".to_string(),
            });
        }
        
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        
        let hovered_edge = match edges.iter().find(|e| e.id == edge_id) {
            Some(edge) => edge,
            None => return Ok(WSMessage::Error {
                message: format!("Edge '{}' not found", edge_id),
                error_code: "EDGE_NOT_FOUND".to_string(),
            }),
        };
        
        // Get node labels for display
        let source_label = nodes.iter().find(|n| n.id == hovered_edge.source)
            .and_then(|n| n.label.as_ref())
            .unwrap_or(&hovered_edge.source);
        let target_label = nodes.iter().find(|n| n.id == hovered_edge.target)
            .and_then(|n| n.label.as_ref())
            .unwrap_or(&hovered_edge.target);
        
        let mut attributes = Vec::new();
        for (key, value) in &hovered_edge.attributes {
            attributes.push(TooltipAttribute {
                name: key.clone(),
                value: format!("{:?}", value),
                display_type: AttributeDisplayType::Text,
            });
        }
        
        let tooltip_data = EdgeTooltipData {
            title: hovered_edge.label.clone().unwrap_or_else(|| format!("{} → {}", source_label, target_label)),
            source_label: source_label.to_string(),
            target_label: target_label.to_string(),
            weight_display: hovered_edge.weight.map(|w| format!("{:.3}", w)),
            attributes,
            path_info: None, // TODO: Add path analysis
        };
        
        Ok(WSMessage::EdgeHoverResponse {
            edge_id: edge_id.to_string(),
            tooltip_data,
        })
    }

    pub async fn handle_nodes_selection_request(&self, node_ids: &[String]) -> StreamingResult<WSMessage> {
        if !self.data_source.supports_graph_view() {
            return Ok(WSMessage::Error {
                message: "Graph view not supported".to_string(),
                error_code: "NO_GRAPH_SUPPORT".to_string(),
            });
        }
        
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        
        // Find selected nodes
        let mut selected_nodes = Vec::new();
        for node_id in node_ids {
            if let Some(node) = nodes.iter().find(|n| n.id == *node_id) {
                selected_nodes.push(node.into());
            }
        }
        
        if selected_nodes.is_empty() {
            return Ok(WSMessage::Error {
                message: "No valid nodes found in selection".to_string(),
                error_code: "NO_VALID_NODES".to_string(),
            });
        }
        
        // Count edges between selected nodes
        let edge_count = edges.iter()
            .filter(|e| node_ids.contains(&e.source) && node_ids.contains(&e.target))
            .count();
        
        // Basic analytics
        let total_degree: usize = node_ids.iter()
            .map(|id| edges.iter().filter(|e| e.source == *id || e.target == *id).count())
            .sum();
        let avg_degree = if node_ids.is_empty() { 0.0 } else { total_degree as f64 / node_ids.len() as f64 };
        
        let selection_analytics = SelectionAnalytics {
            node_count: selected_nodes.len(),
            edge_count,
            connected_components: 1, // TODO: Implement component analysis
            avg_degree,
            total_weight: None, // TODO: Sum edge weights
            communities_represented: vec![], // TODO: Community analysis
        };
        
        let bulk_operations = vec![
            "Export Selection".to_string(),
            "Analyze Subgraph".to_string(),
            "Find Shortest Paths".to_string(),
            "Community Detection".to_string(),
        ];
        
        Ok(WSMessage::NodesSelectionResponse {
            selected_nodes,
            selection_analytics,
            bulk_operations,
        })
    }
}