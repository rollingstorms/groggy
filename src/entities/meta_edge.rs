//! MetaEdge Entity - Edge representing aggregated edge relationships
//!
//! This module provides the concrete implementation for meta-edges, which are
//! special edges that represent aggregated relationships between meta-nodes
//! and other nodes. MetaEdges implement all regular edge operations plus 
//! meta-specific operations.

use crate::api::graph::Graph;
use crate::errors::GraphResult;
use crate::traits::{GraphEntity, EdgeOperations, MetaEdgeOperations};
use crate::types::{EntityId, EdgeId, AttrValue};
use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashMap;

/// A meta-edge representing aggregated edge relationships
///
/// Meta-edges are edges that represent multiple aggregated relationships,
/// typically created during subgraph collapse operations. They have all
/// the capabilities of regular edges plus meta-specific operations.
#[derive(Debug, Clone)]
pub struct MetaEdge {
    /// The unique identifier for this meta-edge
    id: EdgeId,
    /// Reference to the graph containing this meta-edge
    graph: Rc<RefCell<Graph>>,
}

impl MetaEdge {
    /// Create a new MetaEdge wrapper
    ///
    /// # Arguments
    /// * `id` - The EdgeId for this meta-edge
    /// * `graph` - Reference to the graph containing this meta-edge
    ///
    /// # Returns
    /// A new MetaEdge instance that implements all edge traits plus meta-operations
    ///
    /// # Errors
    /// Returns GraphError if the edge doesn't exist or isn't a valid meta-edge
    pub fn new(id: EdgeId, graph: Rc<RefCell<Graph>>) -> GraphResult<Self> {
        // Validate that the edge exists
        {
            let graph_borrowed = graph.borrow();
            if !graph_borrowed.contains_edge(id) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Edge {} does not exist in graph", id
                )));
            }
            
            // Note: We don't strictly validate that it's a meta-edge here because
            // meta-edges are identified by attributes (entity_type="meta") which
            // might be set after creation. The is_meta_edge() method handles detection.
        }
        
        Ok(MetaEdge { id, graph })
    }
    
    /// Get the EdgeId for this meta-edge
    ///
    /// # Returns
    /// The EdgeId that uniquely identifies this meta-edge
    pub fn id(&self) -> EdgeId {
        self.id
    }
}

impl GraphEntity for MetaEdge {
    fn entity_id(&self) -> EntityId {
        EntityId::Edge(self.id) // Meta-edges use the same EntityId as regular edges
    }
    
    fn entity_type(&self) -> &'static str {
        "meta_edge"
    }
    
    fn graph_ref(&self) -> Rc<RefCell<Graph>> {
        self.graph.clone()
    }
    
    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // For meta-edges, related entities are the endpoint nodes
        // These could be regular nodes or meta-nodes
        let (source, target) = self.endpoints()?;
        let mut entities: Vec<Box<dyn GraphEntity>> = Vec::new();
        
        // Check if source is a meta-node
        let graph = self.graph.borrow();
        if graph.is_meta_node(source) {
            drop(graph);
            let source_meta = super::MetaNode::new(source, self.graph.clone())?;
            entities.push(Box::new(source_meta));
        } else {
            drop(graph);
            let source_node = super::Node::new(source, self.graph.clone())?;
            entities.push(Box::new(source_node));
        }
        
        // Check if target is a meta-node
        let graph = self.graph.borrow();
        if graph.is_meta_node(target) {
            drop(graph);
            let target_meta = super::MetaNode::new(target, self.graph.clone())?;
            entities.push(Box::new(target_meta));
        } else {
            drop(graph);
            let target_node = super::Node::new(target, self.graph.clone())?;
            entities.push(Box::new(target_node));
        }
        
        Ok(entities)
    }
    
    fn summary(&self) -> String {
        let graph = self.graph.borrow();
        
        // Get endpoints
        let endpoints = match graph.edge_endpoints(self.id) {
            Ok((source, target)) => format!("{} → {}", source, target),
            Err(_) => "? → ?".to_string(),
        };
        
        // Get edge count if available
        let edge_count_info = if let Some(count) = self.edge_count() {
            format!(", aggregates {} edges", count)
        } else {
            String::new()
        };
        
        // Get a few key attributes if available (excluding meta-specific ones)
        let mut attr_summary = String::new();
        if let Ok(attrs) = graph.get_edge_attrs(self.id) {
            let key_attrs: Vec<_> = attrs.keys()
                .filter(|&k| k != "entity_type" && k != "edge_count" && k != "source" && k != "target")
                .take(2)
                .collect();
            
            if !key_attrs.is_empty() {
                let attr_strs: Vec<String> = key_attrs
                    .iter()
                    .filter_map(|&key| {
                        graph.get_edge_attr(self.id, key).ok().flatten()
                            .map(|value| format!("{}={}", key, value))
                    })
                    .collect();
                
                if !attr_strs.is_empty() {
                    attr_summary = format!(", {}", attr_strs.join(", "));
                }
            }
        }
        
        format!(
            "MetaEdge(id={}, {}{}{}) ", 
            self.id, endpoints, edge_count_info, attr_summary
        )
    }
}

impl EdgeOperations for MetaEdge {
    fn edge_id(&self) -> EdgeId {
        self.id
    }
    
    // EdgeOperations trait provides default implementations for endpoints(), source(), target()
    // that delegate to our graph_ref() and edge_id(), so we don't need to override them
}

impl MetaEdgeOperations for MetaEdge {
    fn is_meta_edge(&self) -> bool {
        let graph = self.graph.borrow();
        match graph.get_edge_attr(self.id, &"entity_type".into()) {
            Ok(Some(attr_value)) => attr_value.as_text() == Some("meta"),
            _ => false,
        }
    }
    
    fn edge_count(&self) -> Option<i64> {
        let graph = self.graph.borrow();
        match graph.get_edge_attr(self.id, &"edge_count".into()) {
            Ok(Some(attr_value)) => attr_value.as_int(),
            _ => None,
        }
    }
    
    fn aggregated_from(&self) -> GraphResult<Option<Vec<EdgeId>>> {
        // TODO: Store and retrieve original edge IDs during collapse
        // For now, this information is not preserved
        Ok(None)
    }
    
    fn expand(&self) -> GraphResult<Option<Vec<EdgeId>>> {
        // TODO: Implement meta-edge expansion
        // This would need to recreate the original edges that were aggregated
        Ok(None)
    }
    
    fn meta_properties(&self) -> GraphResult<HashMap<String, AttrValue>> {
        let mut properties = HashMap::new();
        
        // Add edge count if available
        if let Some(count) = self.edge_count() {
            properties.insert("edge_count".to_string(), AttrValue::Int(count));
        }
        
        // Add entity type
        properties.insert("entity_type".to_string(), AttrValue::Text("meta".to_string()));
        
        // Add is_meta_edge flag
        properties.insert("is_meta_edge".to_string(), AttrValue::Text("true".to_string()));
        
        // Add explicit source and target (these should be set on meta-edges)
        let graph = self.graph.borrow();
        if let Ok(Some(source_attr)) = graph.get_edge_attr(self.id, &"source".into()) {
            properties.insert("explicit_source".to_string(), source_attr);
        }
        if let Ok(Some(target_attr)) = graph.get_edge_attr(self.id, &"target".into()) {
            properties.insert("explicit_target".to_string(), target_attr);
        }
        
        Ok(properties)
    }
}