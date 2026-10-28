//! Owner-facing explicit capture/build adapters share the immutable query engine.
use super::GraphQueryIndex;
use crate::{subgraphs::Subgraph, AttrName, DurableGraph, Graph, GraphResult};
impl Graph {
    /// Capture current values and explicitly build transient query indexes.
    pub fn build_index(
        &self,
        node_properties: &[AttrName],
        edge_properties: &[AttrName],
        topology: bool,
    ) -> GraphResult<GraphQueryIndex> {
        self.read_view()?
            .build_index(node_properties, edge_properties, topology)
    }
}
impl Subgraph {
    /// Freeze the current selected values before explicit index construction.
    /// Stale selections reject rather than silently dropping inactive identities.
    pub fn build_index(
        &self,
        node_properties: &[AttrName],
        edge_properties: &[AttrName],
        topology: bool,
    ) -> GraphResult<GraphQueryIndex> {
        self.pin_view()?
            .build_index(node_properties, edge_properties, topology)
    }
}
impl DurableGraph {
    /// Explicit read-index construction publishes/appends no durable state.
    pub fn build_index(
        &self,
        node_properties: &[AttrName],
        edge_properties: &[AttrName],
        topology: bool,
    ) -> GraphResult<GraphQueryIndex> {
        self.read_view()?
            .build_index(node_properties, edge_properties, topology)
    }
}
