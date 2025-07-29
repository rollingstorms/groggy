//! Groggy - A Modular Graph Library with Git-like Version Control
//!
//! LIBRARY OVERVIEW:
//! This library provides a comprehensive system for managing graphs with full 
//! history tracking, branching, and versioning capabilities. Think "Git for graphs" 
//! with high-performance columnar storage and advanced query capabilities.
//!
//! CORE ARCHITECTURE:
//! ```text
//!     ┌─────────────────────────────────────────────────────────┐
//!     │                    Graph (Main API)                    │
//!     │              Smart Coordinator & Facade                │
//!     └─────────────────────┬───────────────────────────────────┘
//!                           │
//!           ┌───────────────┼───────────────┐
//!           │               │               │
//!     ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
//!     │GraphStore │  │ History   │  │  Query    │
//!     │(Storage)  │  │ System    │  │ Engine    │
//!     │           │  │(Git-like) │  │(Analysis) │
//!     └───────────┘  └───────────┘  └───────────┘
//! ```
//!
//! KEY DESIGN PRINCIPLES:
//! - **Immutable History**: All changes are preserved, enabling time-travel
//! - **Columnar Storage**: Efficient bulk operations for ML/analytics workloads
//! - **Content Addressing**: Automatic deduplication of identical changes
//! - **Smart Coordination**: Graph facade manages component interactions intelligently
//! - **Zero-Copy Views**: Read historical states without materializing full copies
//!
//! PERFORMANCE CHARACTERISTICS:
//! - Node/Edge Operations: O(1) amortized
//! - Attribute Access: O(1) with columnar storage
//! - History Operations: O(log n) with content addressing
//! - Query Operations: O(n) with optimizations for common patterns
//! - Memory Usage: Delta-compressed with configurable limits
//!
//! USAGE EXAMPLES:
//!
//! ```rust,no_run
//! use groggy::{Graph, AttrValue, GraphConfig};
//!
//! // Create a graph with custom configuration
//! let config = GraphConfig::performance_optimized();
//! let mut graph = Graph::with_config(config);
//!
//! // Build the graph structure
//! let alice = graph.add_node();
//! let bob = graph.add_node();
//! let friendship = graph.add_edge(alice, bob).unwrap();
//!
//! // Set rich attributes
//! graph.set_node_attr(alice, "name", AttrValue::Text("Alice".into())).unwrap();
//! graph.set_node_attr(alice, "age", AttrValue::Int(28)).unwrap();
//! graph.set_edge_attr(friendship, "strength", AttrValue::Float(0.9)).unwrap();
//!
//! // Commit to history with metadata
//! let commit1 = graph.commit("Initial social network", "data_team").unwrap();
//!
//! // Work with branches like Git
//! graph.create_branch("experiment").unwrap();
//! graph.checkout_branch("experiment").unwrap();
//!
//! // Make experimental changes
//! let charlie = graph.add_node();
//! graph.set_node_attr(charlie, "name", AttrValue::Text("Charlie".into())).unwrap();
//!
//! // Advanced querying
//! let adults = graph.find_nodes(|
//!     NodeFilter::Attribute("age", AttributeFilter::GreaterThan(AttrValue::Int(18)))
//! ).unwrap();
//!
//! // Time travel - view the graph at any point in history
//! let historical_view = graph.view_at_commit(commit1).unwrap();
//! assert_eq!(historical_view.node_count(), 2); // Before Charlie was added
//!
//! // Merge branches with conflict resolution
//! graph.checkout_branch("main").unwrap();
//! graph.merge_branch("experiment", "data_team").unwrap();
//! ```
//!
//! ADVANCED FEATURES:
//!
//! ```rust,no_run
//! use groggy::{Graph, GraphQuery, AggregationType};
//!
//! let mut graph = Graph::new();
//! // ... build graph ...
//!
//! // Complex analytical queries
//! let query = GraphQuery::builder()
//!     .find_nodes(NodeFilter::Attribute("type", AttributeFilter::Equals("person")))
//!     .aggregate("age", AggregationType::Average)
//!     .group_by("department")
//!     .limit(100)
//!     .build();
//!
//! let results = graph.execute_query(query).unwrap();
//!
//! // Bulk operations for performance
//! let node_ids = graph.add_nodes(1000); // Add 1000 nodes efficiently
//! graph.set_node_attrs_bulk("initialized", 
//!     node_ids.iter().map(|&id| (id, AttrValue::Bool(true))).collect()
//! ).unwrap();
//!
//! // Graph analytics
//! let degree_dist = graph.degree_distribution().unwrap();
//! let communities = graph.detect_communities().unwrap();
//! let centrality = graph.compute_centrality(CentralityType::Betweenness).unwrap();
//! ```

// Core type definitions
pub mod types;
pub mod config;
pub mod errors;
pub mod util;

// Core graph data structures
pub mod core {
    pub mod delta;
    pub mod change_tracker;
    pub mod pool;
    pub mod space;
}

// History management
pub mod history {
    pub mod state;
    pub mod forest;
}

// Read-only views and queries
pub mod view {
    pub mod view;
    pub mod query;
}

// Branch and reference management
pub mod refs {
    pub mod ref_manager;
}

// Public API
pub mod api {
    pub mod graph;
}

// Re-export commonly used types and the main API
pub use types::{NodeIndex, EdgeIndex, AttrName, AttrValue, StateId, BranchName};
pub use config::GraphConfig;
pub use errors::{GraphError, GraphResult, MergeError};
pub use api::graph::{Graph, GraphStats};
pub use view::view::{StateView, ViewSummary};
pub use view::query::{
    AttributeFilter, NumericComparison, StringComparison, MultiCriteria, Criterion,
    filter_nodes_by_attributes, filter_edges_by_attributes,
    filter_by_numeric_comparison, filter_by_string_comparison, filter_by_multi_criteria,
};
pub use refs::ref_manager::{RefManager, Branch, BranchInfo, TagInfo};
pub use history::state::{StateObject, StateMetadata};
pub use history::forest::HistoryForest;

// Re-export core types for advanced usage
pub use core::delta::{ColumnDelta, DeltaObject};
pub use core::change_tracker::ChangeTracker;
pub use core::pool::GraphPool;
pub use core::space::GraphSpace;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Get library information
pub fn info() -> LibraryInfo {
    LibraryInfo {
        name: NAME.to_string(),
        version: VERSION.to_string(),
        description: "A modular graph and history management library".to_string(),
    }
}

/// Library information
#[derive(Debug, Clone)]
pub struct LibraryInfo {
    pub name: String,
    pub version: String,
    pub description: String,
}

impl LibraryInfo {
    /// Get a formatted string representation
    pub fn banner(&self) -> String {
        format!("{} v{} - {}", self.name, self.version, self.description)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_info() {
        let info = info();
        assert_eq!(info.name, NAME);
        assert_eq!(info.version, VERSION);
        assert!(!info.banner().is_empty());
    }

    #[test]
    fn test_basic_integration() {
        // Test that all the main components work together
        let mut graph = Graph::new();
        
        // Basic graph operations
        let node1 = graph.add_node();
        let node2 = graph.add_node();
        let edge = graph.add_edge(node1, node2).unwrap();
        
        // Attributes
        graph.set_node_attribute(node1, "name".to_string(), AttrValue::Text("Alice".to_string())).unwrap();
        graph.set_edge_attribute(edge, "weight".to_string(), AttrValue::Float(1.5)).unwrap();
        
        // History
        let state1 = graph.commit("Initial commit".to_string(), "test_user".to_string()).unwrap();
        
        // Branching
        graph.create_branch("test_branch".to_string(), Some("Test branch".to_string())).unwrap();
        graph.checkout_branch(&"test_branch".to_string()).unwrap();
        
        // More changes
        let node3 = graph.add_node();
        graph.set_node_attribute(node3, "value".to_string(), AttrValue::Int(42)).unwrap();
        let state2 = graph.commit("Add node3".to_string(), "test_user".to_string()).unwrap();
        
        // Views
        let view = graph.view_at_state(state1).unwrap();
        assert_eq!(view.state_id(), state1);
        
        // Stats
        let stats = graph.stats();
        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.edge_count, 1);
        assert_eq!(stats.current_branch, "test_branch".to_string());
        
        // Filtering
        use std::collections::HashMap;
        let mut filters = HashMap::new();
        filters.insert("name".to_string(), AttributeFilter::Equals(AttrValue::Text("Alice".to_string())));
        let filtered = graph.filter_nodes(&filters).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0], node1);
        
        // Tags
        graph.create_tag("v1.0".to_string(), Some(state2)).unwrap();
        let tags = graph.list_tags();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name, "v1.0");
        assert_eq!(tags[0].state_id, state2);
    }

    #[test]
    fn test_error_handling() {
        let mut graph = Graph::new();
        
        // Test invalid operations
        let result = graph.add_edge(999, 1000);
        assert!(result.is_err());
        
        let result = graph.get_node_attribute(999, &"nonexistent".to_string());
        assert!(result.is_err());
        
        let result = graph.checkout_branch(&"nonexistent_branch".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_configuration() {
        let config = GraphConfig::default();
        let graph = Graph::with_config(config.clone());
        
        assert_eq!(graph.config().max_states, config.max_states);
        assert_eq!(graph.config().enable_gc, config.enable_gc);
    }
}
