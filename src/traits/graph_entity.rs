//! GraphEntity - Universal trait for all entities in the graph universe
//!
//! This trait provides the foundation for infinite composability by establishing
//! a common interface that works seamlessly with our existing optimized storage
//! infrastructure (GraphPool, GraphSpace, HistoryForest).

use crate::api::graph::Graph;
use crate::errors::GraphResult;
use crate::types::{AttrName, AttrValue, EntityId};
use std::cell::RefCell;
use std::rc::Rc;

/// Universal trait that every entity in the graph universe implements
///
/// All entities store their data in GraphPool, track state in GraphSpace, and
/// version in HistoryForest. This trait provides a unified interface while
/// leveraging our existing high-performance storage systems.
///
/// # Design Principles
/// - **Storage Integration**: All attribute access goes through our optimized GraphPool
/// - **Zero Duplication**: Traits are pure interfaces - no parallel storage systems
/// - **Performance First**: Most operations are direct references to existing efficient data structures
/// - **Composability**: Every entity can be filtered, queried, and transformed consistently
pub trait GraphEntity: std::fmt::Debug {
    /// Universal identifier for this entity
    ///
    /// # Returns
    /// EntityId variant corresponding to the entity type and its storage ID
    fn entity_id(&self) -> EntityId;

    /// Type identifier for this entity
    ///
    /// # Returns
    /// Static string identifying the entity type (e.g., "node", "subgraph", "neighborhood")
    fn entity_type(&self) -> &'static str;

    /// Reference to our shared storage systems (GraphPool/Space/History)
    ///
    /// # Returns
    /// Rc<RefCell<Graph>> providing access to all storage infrastructure
    fn graph_ref(&self) -> Rc<RefCell<Graph>>;

    /// Get attribute value from GraphPool (no copying - direct reference)
    ///
    /// # Arguments
    /// * `name` - Attribute name to retrieve
    ///
    /// # Returns
    /// Optional reference to the attribute value stored in GraphPool
    ///
    /// # Performance
    /// O(1) - Direct lookup in our optimized columnar storage
    fn get_attribute(&self, name: &AttrName) -> GraphResult<Option<AttrValue>> {
        let binding = self.graph_ref();
        let graph = binding.borrow();
        match self.entity_id() {
            EntityId::Node(id) | EntityId::MetaNode(id) => {
                graph.pool().get_node_attribute(id, name)
            }
            EntityId::Edge(id) => graph.pool().get_edge_attribute(id, name),
            EntityId::Subgraph(id)
            | EntityId::Neighborhood(id)
            | EntityId::Component(id)
            | EntityId::Path(id)
            | EntityId::Filter(id) => graph.pool().get_subgraph_attribute(id, name),
        }
    }

    /// Set attribute value in GraphPool using our existing efficient storage
    ///
    /// # Arguments
    /// * `name` - Attribute name to set
    /// * `value` - Attribute value to store
    ///
    /// # Performance
    /// Uses our existing optimized AttributeColumn storage with memory pooling
    fn set_attribute(&self, name: AttrName, value: AttrValue) -> GraphResult<()> {
        let binding = self.graph_ref();
        let graph = binding.borrow_mut();
        match self.entity_id() {
            EntityId::Node(id) | EntityId::MetaNode(id) => {
                graph.pool_mut().set_node_attribute(id, name, value)
            }
            EntityId::Edge(id) => graph.pool_mut().set_edge_attribute(id, name, value),
            EntityId::Subgraph(id)
            | EntityId::Neighborhood(id)
            | EntityId::Component(id)
            | EntityId::Path(id)
            | EntityId::Filter(id) => graph.pool_mut().set_subgraph_attribute(id, name, value),
        }
    }

    /// Check if entity is currently active in GraphSpace
    ///
    /// # Returns
    /// true if entity is in the active set, false otherwise
    ///
    /// # Performance  
    /// O(1) - Direct HashSet lookup in GraphSpace
    fn is_active(&self) -> bool {
        let binding = self.graph_ref();
        let graph = binding.borrow();
        match self.entity_id() {
            EntityId::Node(id) | EntityId::MetaNode(id) => graph.space().is_node_active(id),
            EntityId::Edge(id) => graph.space().is_edge_active(id),
            EntityId::Subgraph(id)
            | EntityId::Neighborhood(id)
            | EntityId::Component(id)
            | EntityId::Path(id)
            | EntityId::Filter(id) => graph.space().is_subgraph_active(id),
        }
    }

    /// Get related entities using our efficient lookups
    ///
    /// For nodes: returns neighbors as EntityNode wrappers
    /// For edges: returns endpoint nodes  
    /// For subgraphs: returns contained nodes
    ///
    /// # Returns
    /// Vector of related entities as GraphEntity trait objects
    ///
    /// # Performance
    /// Uses existing efficient neighbor/containment algorithms
    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>>;

    /// Summary information using our display system
    ///
    /// # Returns
    /// Human-readable summary string for this entity
    fn summary(&self) -> String;

    /// Check if this entity contains or relates to another entity
    ///
    /// # Arguments
    /// * `other` - Entity to check for containment/relation
    ///
    /// # Returns
    /// true if this entity contains or is related to the other entity
    fn contains_entity(&self, other: &dyn GraphEntity) -> bool {
        // Default implementation checks if other is in related_entities
        if let Ok(related) = self.related_entities() {
            related
                .iter()
                .any(|entity| entity.entity_id() == other.entity_id())
        } else {
            false
        }
    }

    /// Count of related entities
    ///
    /// # Returns
    /// Number of entities related to this one
    ///
    /// # Performance
    /// Should be O(1) for most entity types by caching or direct field access
    fn relation_count(&self) -> usize {
        self.related_entities()
            .map(|entities| entities.len())
            .unwrap_or(0)
    }
}

/// Maximum number of relations to prevent memory issues
pub const MAX_RELATIONS: usize = 1_000_000;

/// Validation trait for entity safety (Worf's domain)
pub trait SafeGraphEntity: GraphEntity {
    /// Validate entity invariants
    ///
    /// # Returns
    /// Ok(()) if entity is valid, GraphError if invariants are violated
    fn validate(&self) -> GraphResult<()> {
        // Check entity ID validity
        if !self.entity_id().is_valid() {
            return Err(crate::errors::GraphError::InvalidInput(
                "Entity ID is invalid".to_string(),
            ));
        }

        // Check relation count limits
        if self.relation_count() > MAX_RELATIONS {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Entity has too many relations: {}",
                self.relation_count()
            )));
        }

        Ok(())
    }

    /// Check for reference cycles (important for hierarchical entities)
    ///
    /// # Returns
    /// Ok(true) if cycles detected, Ok(false) if no cycles, Err on validation failure
    fn check_cycles(&self) -> GraphResult<bool> {
        // TODO: Implement cycle detection for hierarchical structures
        Ok(false)
    }
}
