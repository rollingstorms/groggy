//! Shared trait system for GraphEntity composability
//!
//! This module provides the universal trait interfaces that enable every entity
//! in the graph universe to be composable, queryable, and optimizable while
//! leveraging our existing optimized storage infrastructure (GraphPool, GraphSpace, HistoryForest).

pub mod graph_entity;
pub mod subgraph_operations;
pub mod node_operations;
pub mod edge_operations;

// Re-export core traits for easy importing
pub use graph_entity::GraphEntity;
pub use subgraph_operations::SubgraphOperations;
pub use node_operations::NodeOperations;
pub use edge_operations::EdgeOperations;