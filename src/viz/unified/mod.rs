//! 🎯 Unified Visualization API - GraphDataProvider Trait
//!
//! This module provides the GraphDataProvider trait for graph data extraction.

use crate::errors::GraphResult;
use crate::viz::streaming::data_source::{GraphNode as VizNode, GraphEdge as VizEdge};

/// Trait for extracting graph data (implemented by Graph)
pub trait GraphDataProvider: Send + Sync {
    fn get_viz_nodes(&self) -> GraphResult<Vec<VizNode>>;
    fn get_viz_edges(&self) -> GraphResult<Vec<VizEdge>>;
    fn get_node_count(&self) -> usize;
    fn get_edge_count(&self) -> usize;
}