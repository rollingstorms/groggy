use super::core::FastGraph;
use std::collections::HashSet;

/// High-level graph operations (subgraphs, algorithms, etc.)
/// This module contains ONLY complex operations that operate on the full graph structure
impl FastGraph {
    /// Create subgraph with parallel node filtering - HIGH-LEVEL OPERATION ONLY
    pub fn parallel_subgraph_by_node_ids(&self, node_ids: &HashSet<String>) -> FastGraph {
        let mut subgraph = FastGraph::new(self.is_directed);

        // Add filtered nodes with attributes
        for node_id in node_ids {
            if let Some(node_idx) = self.node_id_to_index.get(node_id) {
                if let Some(node_data) = self.get_node_weight(*node_idx) {
                    // Add node directly to internal graph with all attributes
                    let new_node_idx = subgraph.add_node_to_graph_public(node_data.clone());
                    subgraph
                        .node_id_to_index
                        .insert(node_data.id.clone(), new_node_idx);
                    subgraph
                        .node_index_to_id
                        .insert(new_node_idx, node_data.id.clone());

                    // Copy node attributes from columnar store
                    let node_attrs = self.columnar_store.get_node_attributes(node_idx.index());
                    for (attr_name, attr_value) in node_attrs {
                        subgraph.columnar_store.set_node_attribute(
                            new_node_idx.index(),
                            &attr_name,
                            attr_value,
                        );
                    }
                }
            }
        }

        // Add edges between filtered nodes
        for edge_idx in self.get_edge_indices() {
            if let Some((source_idx, target_idx)) = self.get_edge_endpoints(edge_idx) {
                let source_id = self.node_index_to_id.get(&source_idx);
                let target_id = self.node_index_to_id.get(&target_idx);

                if let (Some(source_id), Some(target_id)) = (source_id, target_id) {
                    if node_ids.contains(source_id.as_str())
                        && node_ids.contains(target_id.as_str())
                    {
                        if let Some(edge_data) = self.get_edge_weight(edge_idx) {
                            // Get new node indices in subgraph
                            let new_source_idx = subgraph.node_id_to_index[source_id];
                            let new_target_idx = subgraph.node_id_to_index[target_id];

                            // Add edge to subgraph
                            let new_edge_idx = subgraph.add_edge_to_graph_public(
                                new_source_idx,
                                new_target_idx,
                                edge_data.clone(),
                            );

                            // Copy edge attributes from columnar store
                            let edge_attrs = self.columnar_store.get_edge_attributes(edge_idx.index());
                            for (attr_name, attr_value) in edge_attrs {
                                subgraph.columnar_store.set_edge_attribute(
                                    new_edge_idx.index(),
                                    &attr_name,
                                    attr_value,
                                );
                            }
                        }
                    }
                }
            }
        }

        subgraph
    }

    /// Create subgraph by filtering nodes with specific attributes - HIGH-LEVEL OPERATION
    pub fn create_subgraph_by_node_filter(
        &self,
        filters: &std::collections::HashMap<String, serde_json::Value>,
    ) -> FastGraph {
        // Use columnar store to get filtered nodes
        let filtered_indices = self.columnar_store.filter_nodes_sparse(filters);
        
        // Convert indices to node IDs
        let filtered_node_ids: HashSet<String> = filtered_indices
            .into_iter()
            .filter_map(|idx| {
                let node_idx = petgraph::graph::NodeIndex::new(idx);
                self.node_index_to_id.get(&node_idx).cloned()
            })
            .collect();

        // Create subgraph with filtered nodes
        self.parallel_subgraph_by_node_ids(&filtered_node_ids)
    }

    /// Degree-based node filtering - HIGH-LEVEL OPERATION
    pub fn filter_nodes_by_degree(&self, min_degree: usize, max_degree: usize) -> Vec<String> {
        let mut filtered_nodes = Vec::new();

        for (node_id, &node_idx) in &self.node_id_to_index {
            let degree = if self.is_directed {
                // For directed graphs, use total degree (in + out)
                self.graph
                    .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                    .count()
                    + self
                        .graph
                        .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                        .count()
            } else {
                // For undirected graphs, use regular degree
                self.graph.neighbors(node_idx).count()
            };

            if degree >= min_degree && degree <= max_degree {
                filtered_nodes.push(node_id.clone());
            }
        }

        filtered_nodes
    }

    /// Get node degree - HIGH-LEVEL OPERATION
    pub fn get_node_degree(&self, node_id: &str) -> Option<usize> {
        if let Some(&node_idx) = self.node_id_to_index.get(node_id) {
            let degree = if self.is_directed {
                self.graph
                    .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                    .count()
                    + self
                        .graph
                        .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                        .count()
            } else {
                self.graph.neighbors(node_idx).count()
            };
            Some(degree)
        } else {
            None
        }
    }
}
