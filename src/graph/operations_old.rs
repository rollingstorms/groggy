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

    /// High-performance filtering using sparse intersection - HIGH-LEVEL OPERATION
    pub fn filter_nodes_sparse(&self, filters: &std::collections::HashMap<String, serde_json::Value>) -> Vec<String> {
        let filtered_indices = self.columnar_store.filter_nodes_sparse(filters);
        
        // Convert indices to node IDs
        filtered_indices
            .into_iter()
            .filter_map(|idx| {
                let node_idx = petgraph::graph::NodeIndex::new(idx);
                self.node_index_to_id.get(&node_idx).cloned()
            })
            .collect()
    }
}

impl FastGraph {
    /// Create subgraph with parallel node filtering
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
                }
            }
        }

        // Add edges between filtered nodes in parallel
        let edges_to_add: Vec<_> = self
            .get_edge_indices()
            .par_iter()
            .filter_map(|edge_idx| {
                if let Some((source_idx, target_idx)) = self.get_edge_endpoints(*edge_idx) {
                    let source_id = self.node_index_to_id.get(&source_idx)?;
                    let target_id = self.node_index_to_id.get(&target_idx)?;

                    if node_ids.contains(source_id.as_str())
                        && node_ids.contains(target_id.as_str())
                    {
                        let edge_data = self.get_edge_weight(*edge_idx)?;
                        Some((source_id.clone(), target_id.clone(), edge_data.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        // Add edges to subgraph with attributes
        for (source, target, edge_data) in edges_to_add {
            let source_idx = *subgraph.node_id_to_index.get(&source).unwrap();
            let target_idx = *subgraph.node_id_to_index.get(&target).unwrap();
            // Add edge directly with all attributes
            let _ = subgraph.add_edge_to_graph_public(source_idx, target_idx, edge_data);
        }

        subgraph
    }

    /// Find connected component starting from a node
    pub fn connected_component(&self, start_node_id: &str) -> Option<FastGraph> {
        let start_idx = self.node_id_to_index.get(start_node_id)?;

        let mut visited = HashSet::new();
        let mut queue = vec![*start_idx];
        visited.insert(*start_idx);

        // BFS to find all connected nodes
        while let Some(current_idx) = queue.pop() {
            // Check all neighbors
            for neighbor_idx in self.get_neighbors_public(current_idx) {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    queue.push(neighbor_idx);
                }
            }

            // Also check incoming edges (for undirected behavior)
            for edge_ref in self.get_edges_directed(current_idx, petgraph::Direction::Incoming) {
                let source_idx = edge_ref.source();
                if !visited.contains(&source_idx) {
                    visited.insert(source_idx);
                    queue.push(source_idx);
                }
            }
        }

        // Convert node indices to IDs
        let node_ids: HashSet<String> = visited
            .iter()
            .filter_map(|idx| self.node_index_to_id.get(idx).cloned())
            .collect();

        Some(self.parallel_subgraph_by_node_ids(&node_ids))
    }

    /// Get degree of a node
    pub fn node_degree(&self, node_id: &str) -> Option<usize> {
        let node_idx = self.node_id_to_index.get(node_id)?;

        let in_degree = self
            .get_edges_directed(*node_idx, petgraph::Direction::Incoming)
            .len();
        let out_degree = self
            .get_edges_directed(*node_idx, petgraph::Direction::Outgoing)
            .len();

        Some(in_degree + out_degree)
    }

    /// Get all nodes with degree greater than threshold
    pub fn high_degree_nodes(&self, min_degree: usize) -> Vec<(String, usize)> {
        self.node_id_to_index
            .iter()
            .filter_map(|(node_id, _node_idx)| {
                let degree = self.node_degree(node_id)?;
                if degree >= min_degree {
                    Some((node_id.clone(), degree))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Internal method to remove a node
    pub fn remove_node_internal(&mut self, node_id: String) -> bool {
        if let Some(node_idx) = self.node_id_to_index.remove(&node_id) {
            // Remove from reverse mapping
            self.node_index_to_id.remove(&node_idx);

            // Remove from columnar storage
            self.columnar_store.remove_node_legacy(node_idx.index());

            // Remove from graph (this also removes all connected edges)
            self.graph.remove_node(node_idx);

            true
        } else {
            false // Node didn't exist
        }
    }

    /// Internal method to remove multiple nodes efficiently
    pub fn remove_nodes_internal(&mut self, node_ids: Vec<String>) -> usize {
        let mut removed_count = 0;
        let mut indices_to_remove = Vec::new();

        // Collect node indices to remove
        for node_id in &node_ids {
            if let Some(node_idx) = self.node_id_to_index.get(node_id) {
                indices_to_remove.push((*node_idx, node_id.clone()));
            }
        }

        // Remove nodes (this automatically removes connected edges)
        for (node_idx, node_id) in indices_to_remove {
            self.node_id_to_index.remove(&node_id);
            self.node_index_to_id.remove(&node_idx);
            self.columnar_store.remove_node_legacy(node_idx.index());
            self.graph.remove_node(node_idx);
            removed_count += 1;
        }

        removed_count
    }

    /// Internal method to remove an edge between two nodes
    pub fn remove_edge_internal(&mut self, source: String, target: String) -> bool {
        let source_idx = match self.node_id_to_index.get(&source) {
            Some(idx) => *idx,
            None => return false,
        };
        let target_idx = match self.node_id_to_index.get(&target) {
            Some(idx) => *idx,
            None => return false,
        };

        if let Some(edge_idx) = self.graph.find_edge(source_idx, target_idx) {
            self.graph.remove_edge(edge_idx);
            true
        } else {
            false // Edge didn't exist
        }
    }

    /// Internal method to remove multiple edges efficiently
    pub fn remove_edges_internal(&mut self, edge_pairs: Vec<(String, String)>) -> usize {
        let mut removed_count = 0;
        let mut edges_to_remove = Vec::new();

        // Collect edge indices to remove
        for (source, target) in &edge_pairs {
            if let (Some(source_idx), Some(target_idx)) = (
                self.node_id_to_index.get(source),
                self.node_id_to_index.get(target),
            ) {
                if let Some(edge_idx) = self.graph.find_edge(*source_idx, *target_idx) {
                    edges_to_remove.push(edge_idx);
                }
            }
        }

        // Remove edges
        for edge_idx in edges_to_remove {
            self.graph.remove_edge(edge_idx);
            removed_count += 1;
        }

        removed_count
    }

    /// Internal method to add an edge
    pub fn add_edge_internal(
        &mut self,
        source: String,
        target: String,
        attributes: Option<&PyDict>,
    ) -> PyResult<()> {
        // Ensure both nodes exist, create if they don't
        if !self.node_id_to_index.contains_key(&source) {
            self.add_node(source.clone(), None)?;
        }
        if !self.node_id_to_index.contains_key(&target) {
            self.add_node(target.clone(), None)?;
        }

        let source_idx = *self.node_id_to_index.get(&source).unwrap();
        let target_idx = *self.node_id_to_index.get(&target).unwrap();

        // Check if edge already exists
        if self.graph.find_edge(source_idx, target_idx).is_some() {
            return Ok(()); // Edge already exists, skip
        }

        let attrs = if let Some(py_attrs) = attributes {
            python_dict_to_json_map(py_attrs)?
        } else {
            HashMap::new()
        };
        let edge_data = EdgeData {
            source: source.clone(),
            target: target.clone(),
            attr_uids: std::collections::HashSet::new(),
        };

        let edge_idx = self.graph.add_edge(source_idx, target_idx, edge_data);
        self.edge_index_to_endpoints
            .insert(edge_idx, (source, target));

        // Store attributes in columnar format
        for (attr_name, attr_value) in attrs {
            self.columnar_store
                .set_edge_attribute(edge_idx.index(), &attr_name, attr_value);
        }

        Ok(())
    }

    /// Internal method to add edges in bulk
    pub fn add_edges_internal(&mut self, edge_data: &PyList) -> PyResult<()> {
        let mut edges_to_add = Vec::new();
        let mut nodes_to_create = std::collections::HashSet::new();

        // First pass: collect edges and nodes to create
        for item in edge_data {
            let tuple = item.downcast::<pyo3::types::PyTuple>()?;
            let source: String = tuple.get_item(0)?.extract()?;
            let target: String = tuple.get_item(1)?.extract()?;

            if !self.node_id_to_index.contains_key(&source) {
                nodes_to_create.insert(source.clone());
            }
            if !self.node_id_to_index.contains_key(&target) {
                nodes_to_create.insert(target.clone());
            }

            let attributes = if tuple.len() > 2 {
                let py_attrs = tuple.get_item(2)?.downcast::<PyDict>()?;
                python_dict_to_json_map(py_attrs)?
            } else {
                HashMap::new()
            };

            edges_to_add.push((source, target, attributes));
        }

        // Create missing nodes
        for node_id in nodes_to_create {
            self.add_node(node_id, None)?;
        }

        // Add all edges
        for (source, target, attributes) in edges_to_add {
            let source_idx = *self.node_id_to_index.get(&source).unwrap();
            let target_idx = *self.node_id_to_index.get(&target).unwrap();

            let edge_data = EdgeData {
                source: source.clone(),
                target: target.clone(),
                attr_uids: std::collections::HashSet::new(),
            };

            let edge_idx = self.graph.add_edge(source_idx, target_idx, edge_data);
            self.edge_index_to_endpoints
                .insert(edge_idx, (source, target));

            // Store attributes in columnar format
            for (attr_name, attr_value) in attributes {
                self.columnar_store
                    .set_edge_attribute(edge_idx.index(), &attr_name, attr_value);
            }
        }

        Ok(())
    }

    /// Bulk add nodes with attributes efficiently
    /// Reduces per-node overhead from 38μs to ~5μs by batching columnar operations
    pub fn bulk_add_nodes(&mut self, nodes_data: Vec<(String, HashMap<String, serde_json::Value>)>) -> Vec<petgraph::graph::NodeIndex> {
        let mut node_indices = Vec::with_capacity(nodes_data.len());
        let mut batch_data = Vec::with_capacity(nodes_data.len());
        
        // First pass: Add nodes to graph structure and collect batch data
        for (node_id, attributes) in nodes_data {
            // Create NodeData
            let node_data = super::types::NodeData {
                id: node_id.clone(),
                attr_uids: std::collections::HashSet::new(), // Will be populated by batch operation
            };
            
            // Add to internal graph
            let node_idx = self.graph.add_node(node_data);
            node_indices.push(node_idx);
            
            // Update mappings
            self.node_id_to_index.insert(node_id.clone(), node_idx);
            self.node_index_to_id.insert(node_idx, node_id);
            
            // Prepare for batch attribute setting
            let attrs_vec: Vec<(String, serde_json::Value)> = attributes.into_iter().collect();
            if !attrs_vec.is_empty() {
                batch_data.push((node_idx.index(), attrs_vec));
            }
        }
        
        // Second pass: Batch set all attributes in columnar store
        if !batch_data.is_empty() {
            let attr_name_to_uid = self.columnar_store.batch_set_node_attributes(batch_data);
            
            // Third pass: Update attr_uids in NodeData
            for (&node_idx, _) in self.node_id_to_index.iter() {
                if let Some(node_data) = self.graph.node_weight_mut(node_idx) {
                    // Collect attr_uids for this node
                    let mut attr_uids = std::collections::HashSet::new();
                    for (_, attr_uid) in &attr_name_to_uid {
                        // Check if this node has this attribute
                        if let Some(attr_map) = self.columnar_store.node_attributes.get(attr_uid) {
                            if attr_map.contains_key(&node_idx.index()) {
                                attr_uids.insert(attr_uid.clone());
                            }
                        }
                    }
                    node_data.attr_uids = attr_uids;
                }
            }
        }
        
        node_indices
    }

    /// Bulk add edges with attributes efficiently
    pub fn bulk_add_edges(&mut self, edges_data: Vec<(String, String, HashMap<String, serde_json::Value>)>) -> Vec<petgraph::graph::EdgeIndex> {
        let mut edge_indices = Vec::with_capacity(edges_data.len());
        let mut batch_data = Vec::with_capacity(edges_data.len());
        
        // First pass: Add edges to graph structure and collect batch data
        for (source_id, target_id, attributes) in edges_data {
            // Get node indices
            if let (Some(&source_idx), Some(&target_idx)) = 
                (self.node_id_to_index.get(&source_id), self.node_id_to_index.get(&target_id)) {
                
                // Create EdgeData
                let edge_data = super::types::EdgeData {
                    attr_uids: std::collections::HashSet::new(), // Will be populated by batch operation
                };
                
                // Add to internal graph
                let edge_idx = self.graph.add_edge(source_idx, target_idx, edge_data);
                edge_indices.push(edge_idx);
                
                // Prepare for batch attribute setting
                let attrs_vec: Vec<(String, serde_json::Value)> = attributes.into_iter().collect();
                if !attrs_vec.is_empty() {
                    batch_data.push((edge_idx.index(), attrs_vec));
                }
            }
        }
        
        // Second pass: Batch set all attributes in columnar store
        if !batch_data.is_empty() {
            let attr_name_to_uid = self.columnar_store.batch_set_edge_attributes(batch_data);
            
            // Third pass: Update attr_uids in EdgeData
            for edge_idx in &edge_indices {
                if let Some(edge_data) = self.graph.edge_weight_mut(*edge_idx) {
                    // Collect attr_uids for this edge
                    let mut attr_uids = std::collections::HashSet::new();
                    for (_, attr_uid) in &attr_name_to_uid {
                        // Check if this edge has this attribute
                        if let Some(attr_map) = self.columnar_store.edge_attributes.get(attr_uid) {
                            if attr_map.contains_key(&edge_idx.index()) {
                                attr_uids.insert(attr_uid.clone());
                            }
                        }
                    }
                    edge_data.attr_uids = attr_uids;
                }
            }
        }
        
        edge_indices
    }

    /// High-performance filtering using sparse intersection instead of bitmaps
    pub fn filter_nodes_sparse(&self, filters: &HashMap<String, serde_json::Value>) -> Vec<String> {
        let filtered_indices = self.columnar_store.filter_nodes_sparse(filters);
        
        // Convert indices to node IDs
        filtered_indices
            .into_iter()
            .filter_map(|idx| {
                let node_idx = petgraph::graph::NodeIndex::new(idx);
                self.node_index_to_id.get(&node_idx).cloned()
            })
            .collect()
    }

    /// Optimized bulk node creation from Python interface
    pub fn create_nodes_from_list(&mut self, py_nodes: &PyList) -> PyResult<Vec<String>> {
        let mut nodes_data = Vec::new();
        
        // Extract all node data first
        for py_node in py_nodes {
            let py_dict = py_node.downcast::<PyDict>()?;
            
            // Extract node_id
            let node_id: String = py_dict.get_item("id")?.unwrap().extract()?;
            
            // Extract attributes
            let mut attributes = HashMap::new();
            for (key, value) in py_dict.iter() {
                let key_str: String = key.extract()?;
                if key_str != "id" {
                    let json_value = crate::utils::python_to_json_value(value)?;
                    attributes.insert(key_str, json_value);
                }
            }
            
            nodes_data.push((node_id, attributes));
        }
        
        // Bulk add all nodes
        let _node_indices = self.bulk_add_nodes(nodes_data.clone());
        
        // Return node IDs
        Ok(nodes_data.into_iter().map(|(id, _)| id).collect())
    }

    /// Optimized bulk edge creation from Python interface  
    pub fn create_edges_from_list(&mut self, py_edges: &PyList) -> PyResult<()> {
        let mut edges_data = Vec::new();
        
        // Extract all edge data first
        for py_edge in py_edges {
            let py_dict = py_edge.downcast::<PyDict>()?;
            
            // Extract source and target
            let source_id: String = py_dict.get_item("source")?.unwrap().extract()?;
            let target_id: String = py_dict.get_item("target")?.unwrap().extract()?;
            
            // Extract attributes
            let mut attributes = HashMap::new();
            for (key, value) in py_dict.iter() {
                let key_str: String = key.extract()?;
                if key_str != "source" && key_str != "target" {
                    let json_value = crate::utils::python_to_json_value(value)?;
                    attributes.insert(key_str, json_value);
                }
            }
            
            edges_data.push((source_id, target_id, attributes));
        }
        
        // Bulk add all edges
        let _edge_indices = self.bulk_add_edges(edges_data);
        
        Ok(())
    }
}
