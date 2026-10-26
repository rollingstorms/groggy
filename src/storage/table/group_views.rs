//! Immutable, ID-preserving graph views grouped by an explicit node column.
use super::{structural, visualization, GraphTable, Table};
use crate::{
    state::{GraphSnapshot, PinnedReadState},
    types::GraphType,
    AttrValue, GraphError, GraphResult, GraphView,
};

impl GraphTable {
    /// Group node rows by one column and retain only internal edges. All results
    /// share one frozen capture of table values, without a writable graph owner.
    /// Standalone tables have no direction metadata; callers supply it explicitly.
    pub fn to_subgraphs(&self, by: &str, graph_type: GraphType) -> GraphResult<Vec<GraphView>> {
        let groups = self.induced_groups(self.nodes().group_by(&[by.to_owned()])?)?;
        let mut snapshot = GraphSnapshot::empty_with_type(0, graph_type);
        snapshot.active_nodes = structural::ids(self.nodes().base(), "node_id")?;
        let node_values = visualization::attributes(self.nodes().base(), &["node_id"])?;
        let edge_values =
            visualization::attributes(self.edges().base(), &["edge_id", "source", "target"])?;
        if node_values
            .iter()
            .chain(&edge_values)
            .flat_map(|row| row.values())
            .any(|value| matches!(value, AttrValue::SubgraphRef(_)))
        {
            return Err(GraphError::InvalidInput("Grouped table views require portable values; pooled subgraph references need their source pool".into()));
        }
        snapshot.node_attributes = snapshot
            .active_nodes
            .iter()
            .copied()
            .zip(node_values)
            .collect();
        let ids = structural::ids(self.edges().base(), "edge_id")?;
        let sources = structural::ids(self.edges().base(), "source")?;
        let targets = structural::ids(self.edges().base(), "target")?;
        snapshot.edges = ids
            .iter()
            .copied()
            .zip(sources.into_iter().zip(targets))
            .collect();
        snapshot.edge_attributes = ids.into_iter().zip(edge_values).collect();
        let state = PinnedReadState::from_snapshot(snapshot)?;
        groups
            .into_iter()
            .map(|group| {
                GraphView::from_membership(
                    state.clone(),
                    group.nodes().node_ids()?.into_iter().collect(),
                    group.edges().edge_ids()?.into_iter().collect(),
                )
            })
            .collect()
    }
}
