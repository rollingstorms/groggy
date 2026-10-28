//! Owner-facing explicit capture/build adapters share the immutable query engine.
use super::GraphQueryIndex;
use crate::{subgraphs::Subgraph, AttrName, DurableGraph, Graph, GraphResult};
impl Graph {
    /// Capture current selected references and their retained source membership.
    pub fn build_group_index(&self) -> GraphResult<super::GroupMembershipIndex> {
        self.read_view()?.build_group_index()
    }
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
    /// Freeze this connected selection for composable immutable queries.
    pub fn query(&self) -> GraphResult<crate::GraphView> {
        self.pin_view()
    }
    /// Capture the current connected selection before building a frozen index.
    pub fn build_group_index(&self) -> GraphResult<super::GroupMembershipIndex> {
        self.pin_view()?.build_group_index()
    }
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
    /// Capture the accepted durable state without keeping the writer lock alive.
    pub fn query(&self) -> GraphResult<crate::GraphView> {
        self.read_view()
    }
    /// Explicit group capture/build without appending or retaining a storage lock.
    pub fn build_group_index(&self) -> GraphResult<super::GroupMembershipIndex> {
        self.read_view()?.build_group_index()
    }
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

impl Graph {
    /// Explicitly capture/index the current branch's effective-time event lineage.
    pub fn build_effective_index(&self) -> GraphResult<super::EffectiveTimeIndex> {
        self.prepare_read_at(0, None)?.build_index()
    }
}
impl DurableGraph {
    /// Capture/index retained events without appending or retaining a storage lock.
    pub fn build_effective_index(&self) -> GraphResult<super::EffectiveTimeIndex> {
        self.prepare_read_at(0, None)?.build_index()
    }
}
