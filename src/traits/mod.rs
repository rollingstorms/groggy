//! Shared trait system for GraphEntity composability
//!
//! This module provides the universal trait interfaces that enable every entity
//! in the graph universe to be composable, queryable, and optimizable while
//! leveraging our existing optimized storage infrastructure (GraphPool, GraphSpace, HistoryForest).

pub mod component_operations;
pub mod edge_operations;
pub mod filter_operations;
pub mod graph_entity;
pub mod neighborhood_operations;
pub mod node_operations;
pub mod subgraph_operations;

// Re-export core traits for easy importing
pub use component_operations::ComponentOperations;
pub use edge_operations::EdgeOperations;
pub use filter_operations::{FilterCriteria, FilterOperations, FilterStats};
pub use graph_entity::GraphEntity;
pub use neighborhood_operations::{NeighborhoodOperations, NeighborhoodStats};
pub use node_operations::NodeOperations;
pub use subgraph_operations::SubgraphOperations;
