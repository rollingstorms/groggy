//! Mutable workspace - the "working graph" that tracks changes.

use std::collections::HashMap;
use crate::types::{NodeIndex, EdgeIndex, AttrName, AttrValue, StateId};
use crate::core::pool::GraphPool;
use crate::core::change_tracker::ChangeTracker;
use crate::errors::GraphResult;

/// Mutable workspace - the "working graph"
#[derive(Debug)]
pub struct GraphSpace {
    /// Current graph structure
    pub pool: GraphPool,
    /// Which state we're based on
    pub base_state: StateId,
    /// Uncommitted changes tracking
    pub changes: ChangeTracker,
}

impl GraphSpace {
    /// Create a new graph space based on a state
    pub fn new(base_state: StateId) -> Self {
        Self {
            pool: GraphPool::new(),
            base_state,
            changes: ChangeTracker::new(),
        }
    }

    /// Create a graph space with initial capacity
    pub fn with_capacity(base_state: StateId, node_capacity: usize, edge_capacity: usize) -> Self {
        Self {
            pool: GraphPool::with_capacity(node_capacity, edge_capacity),
            base_state,
            changes: ChangeTracker::new(),
        }
    }

    /// Add a new node and return its index
    pub fn add_node(&mut self) -> NodeIndex {
        let node = self.pool.add_node();
        self.changes.track_node_active_change(node, true);
        node
    }

    /// Add multiple nodes and return their indices
    pub fn add_nodes(&mut self, count: usize) -> Vec<NodeIndex> {
        let mut nodes = Vec::with_capacity(count);
        for _ in 0..count {
            nodes.push(self.add_node());
        }
        nodes
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, source: NodeIndex, target: NodeIndex) -> GraphResult<EdgeIndex> {
        let edge = self.pool.add_edge(source, target)?;
        self.changes.track_edge_active_change(edge, true);
        Ok(edge)
    }

    /// Add multiple edges
    pub fn add_edges(&mut self, edges: &[(NodeIndex, NodeIndex)]) -> GraphResult<Vec<EdgeIndex>> {
        let mut edge_ids = Vec::with_capacity(edges.len());
        for &(source, target) in edges {
            edge_ids.push(self.add_edge(source, target)?);
        }
        Ok(edge_ids)
    }

    /// Remove a node (mark as inactive)
    pub fn remove_node(&mut self, node: NodeIndex) -> GraphResult<()> {
        self.pool.remove_node(node)?;
        self.changes.track_node_active_change(node, false);
        Ok(())
    }

    /// Remove multiple nodes
    pub fn remove_nodes(&mut self, nodes: &[NodeIndex]) -> GraphResult<()> {
        for &node in nodes {
            self.remove_node(node)?;
        }
        Ok(())
    }

    /// Remove an edge (mark as inactive)
    pub fn remove_edge(&mut self, edge: EdgeIndex) -> GraphResult<()> {
        self.pool.remove_edge(edge)?;
        self.changes.track_edge_active_change(edge, false);
        Ok(())
    }

    /// Remove multiple edges
    pub fn remove_edges(&mut self, edges: &[EdgeIndex]) -> GraphResult<()> {
        for &edge in edges {
            self.remove_edge(edge)?;
        }
        Ok(())
    }

    /// Get the number of active nodes
    pub fn node_count(&self) -> usize {
        self.pool.active_nodes.len()
    }

    /// Get the number of active edges
    pub fn edge_count(&self) -> usize {
        self.pool.active_edges.len()
    }

    /// Get all active node IDs
    pub fn get_node_ids(&self) -> Vec<NodeIndex> {
        self.pool.get_node_ids()
    }

    /// Get all active edge IDs
    pub fn get_edge_ids(&self) -> Vec<EdgeIndex> {
        self.pool.get_edge_ids()
    }

    /// Check if a node exists and is active
    pub fn has_node(&self, node: NodeIndex) -> bool {
        self.pool.has_node(node)
    }

    /// Check if an edge exists and is active
    pub fn has_edge(&self, edge: EdgeIndex) -> bool {
        self.pool.has_edge(edge)
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, node: NodeIndex) -> GraphResult<Vec<NodeIndex>> {
        self.pool.get_neighbors(node)
    }

    /// Set a node attribute
    pub fn set_node_attribute(&mut self, node: NodeIndex, attr_name: AttrName, value: AttrValue) -> GraphResult<()> {
        self.pool.set_node_attribute(node, attr_name.clone(), value.clone())?;
        self.changes.track_node_attr_change(node, attr_name, value);
        Ok(())
    }

    /// Set multiple node attributes at once
    pub fn set_node_attributes(&mut self, node: NodeIndex, attributes: HashMap<AttrName, AttrValue>) -> GraphResult<()> {
        for (attr_name, value) in attributes {
            self.set_node_attribute(node, attr_name, value)?;
        }
        Ok(())
    }

    /// Set an edge attribute
    pub fn set_edge_attribute(&mut self, edge: EdgeIndex, attr_name: AttrName, value: AttrValue) -> GraphResult<()> {
        self.pool.set_edge_attribute(edge, attr_name.clone(), value.clone())?;
        self.changes.track_edge_attr_change(edge, attr_name, value);
        Ok(())
    }

    /// Set multiple edge attributes at once
    pub fn set_edge_attributes(&mut self, edge: EdgeIndex, attributes: HashMap<AttrName, AttrValue>) -> GraphResult<()> {
        for (attr_name, value) in attributes {
            self.set_edge_attribute(edge, attr_name, value)?;
        }
        Ok(())
    }

    /// Get a node attribute
    pub fn get_node_attribute(&self, node: NodeIndex, attr_name: &AttrName) -> GraphResult<Option<&AttrValue>> {
        self.pool.get_node_attribute(node, attr_name)
    }

    /// Get an edge attribute
    pub fn get_edge_attribute(&self, edge: EdgeIndex, attr_name: &AttrName) -> GraphResult<Option<&AttrValue>> {
        self.pool.get_edge_attribute(edge, attr_name)
    }

    /// Get all attributes for a node
    pub fn get_node_attributes(&self, node: NodeIndex) -> GraphResult<HashMap<AttrName, AttrValue>> {
        self.pool.get_node_attributes(node)
    }

    /// Get all attributes for an edge
    pub fn get_edge_attributes(&self, edge: EdgeIndex) -> GraphResult<HashMap<AttrName, AttrValue>> {
        self.pool.get_edge_attributes(edge)
    }

    /// Get edge endpoints
    pub fn get_edge_endpoints(&self, edge: EdgeIndex) -> GraphResult<(NodeIndex, NodeIndex)> {
        self.pool.get_edge_endpoints(edge)
    }

    /// Check if there are uncommitted changes
    pub fn has_uncommitted_changes(&self) -> bool {
        self.changes.has_changes()
    }

    /// Get the number of uncommitted changes
    pub fn uncommitted_change_count(&self) -> usize {
        self.changes.change_count()
    }

    /// Get all nodes that have been modified
    pub fn get_modified_nodes(&self) -> Vec<NodeIndex> {
        self.changes.get_changed_nodes()
    }

    /// Get all edges that have been modified
    pub fn get_modified_edges(&self) -> Vec<EdgeIndex> {
        self.changes.get_changed_edges()
    }

    /// Reset to the base state, discarding all changes
    pub fn reset(&mut self) {
        self.changes.clear();
        // Note: In a full implementation, we'd also need to restore the pool
        // to match the base state. This would require loading from the history.
    }

    /// Get the base state this workspace is derived from
    pub fn get_base_state(&self) -> StateId {
        self.base_state
    }

    /// Update the base state (typically after a commit)
    pub fn set_base_state(&mut self, new_base: StateId) {
        self.base_state = new_base;
        self.changes.clear();
    }

    /// Create a snapshot of current changes for committing
    pub fn create_change_snapshot(&self) -> crate::core::delta::DeltaObject {
        self.changes.to_delta_object()
    }

    /// Apply changes from another workspace
    pub fn merge_changes(&mut self, other: &GraphSpace) -> GraphResult<()> {
        // This is a simplified merge - in practice, we'd need conflict resolution
        self.changes.merge(&other.changes);
        
        // Apply the changes to our pool as well
        // Note: This is a simplified implementation
        for node in other.changes.get_changed_nodes() {
            if other.pool.has_node(node) && !self.pool.has_node(node) {
                // Add missing nodes
                while self.pool.next_node_id <= node {
                    self.pool.add_node();
                }
            }
        }
        
        Ok(())
    }

    /// Add a node with a specific ID
    pub fn add_node_with_id(&mut self, node_id: NodeIndex) -> GraphResult<()> {
        // Ensure we have enough capacity
        while self.pool.next_node_id <= node_id {
            self.pool.add_node();
        }
        Ok(())
    }

    /// Add an edge with a specific ID  
    pub fn add_edge_with_id(&mut self, _edge_id: EdgeIndex, source: NodeIndex, target: NodeIndex) -> GraphResult<()> {
        // This is a simplified implementation - real version would need more logic
        self.add_edge(source, target)?;
        Ok(())
    }

    /// Remove a node attribute
    pub fn remove_node_attribute(&mut self, node: NodeIndex, attr_name: &AttrName) -> GraphResult<()> {
        if let Some(attr_vec) = self.pool.node_attrs.get_mut(attr_name) {
            if node < attr_vec.len() {
                attr_vec[node] = AttrValue::Bool(false); // Reset to default
                self.changes.track_node_change(node);
            }
        }
        Ok(())
    }

    /// Remove an edge attribute
    pub fn remove_edge_attribute(&mut self, edge: EdgeIndex, attr_name: &AttrName) -> GraphResult<()> {
        if let Some(attr_vec) = self.pool.edge_attrs.get_mut(attr_name) {
            if edge < attr_vec.len() {
                attr_vec[edge] = AttrValue::Bool(false); // Reset to default
                self.changes.track_edge_change(edge);
            }
        }
        Ok(())
    }

    /// Check if there are uncommitted changes
    pub fn has_changes(&self) -> bool {
        self.changes.has_changes()
    }

    /// Clear all uncommitted changes
    pub fn clear_changes(&mut self) {
        self.changes.clear();
    }
}

impl Default for GraphSpace {
    fn default() -> Self {
        Self::new(0) // Default to state 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_space_operations() {
        let mut space = GraphSpace::new(0);
        
        let node1 = space.add_node();
        let node2 = space.add_node();
        
        assert_eq!(space.node_count(), 2);
        assert!(space.has_uncommitted_changes());
        
        let edge = space.add_edge(node1, node2).unwrap();
        assert_eq!(space.edge_count(), 1);
        
        space.set_node_attribute(node1, "name".to_string(), AttrValue::Text("Alice".to_string())).unwrap();
        
        let modified_nodes = space.get_modified_nodes();
        assert!(modified_nodes.contains(&node1));
        assert!(modified_nodes.contains(&node2));
    }

    #[test]
    fn test_change_tracking() {
        let mut space = GraphSpace::new(0);
        
        let node = space.add_node();
        space.set_node_attribute(node, "value".to_string(), AttrValue::Int(42)).unwrap();
        
        assert!(space.has_uncommitted_changes());
        assert_eq!(space.uncommitted_change_count(), 2); // node creation + attribute
        
        space.reset();
        assert!(!space.has_uncommitted_changes());
    }
}
