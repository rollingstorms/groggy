pub mod types;
pub mod core;
pub mod operations;
pub mod algorithms;
pub mod views;

pub use core::FastGraph;
pub use types::{NodeData, EdgeData, GraphType, LegacyNodeData, LegacyEdgeData};
pub use views::{GraphView, NodeView, AttributeView, ViewManager};
