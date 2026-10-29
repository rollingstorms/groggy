//! Explicit independent graph construction from selected immutable values.
use super::*;
use crate::GraphView;

impl Graph {
    /// Materialize exactly a frozen selection into a new publishing graph.
    /// IDs, direction and selected properties are preserved. Source history and
    /// unrelated/inactive entities are not copied; use an owned snapshot for recovery.
    /// Typed local references must resolve within the selection. Referenced frozen
    /// group sources retain their own independent source-qualified membership.
    pub fn from_view(view: &GraphView) -> GraphResult<Self> {
        let mut snapshot = view.to_snapshot();
        snapshot.state_id = 0;
        for attrs in snapshot
            .node_attributes
            .values()
            .chain(snapshot.edge_attributes.values())
        {
            for value in attrs.values() {
                let valid = match value {
                    AttrValue::NodeArray(ids) => ids.iter().all(|id| view.contains_node(*id)),
                    AttrValue::EdgeArray(ids) => ids.iter().all(|id| view.contains_edge(*id)),
                    _ => true,
                };
                if !valid {
                    return Err(GraphError::InvalidInput(
                        "Graph copy has a typed entity reference outside its selection".into(),
                    ));
                }
            }
        }
        let mut groups = HashMap::new();
        for id in crate::storage::retained_group::references(&snapshot) {
            let group = view.read_state().groups().get(&id).ok_or_else(|| {
                GraphError::InvalidInput(format!("Graph copy lacks retained group source {id}"))
            })?;
            groups.insert(id, group.clone());
        }
        let mut graph = Graph::new_with_type_manual(snapshot.graph_type);
        graph
            .pool
            .borrow_mut()
            .install_snapshot_groups(Arc::new(groups));
        graph.reset_to_snapshot(snapshot)?;
        graph.seed_selected_import()?;
        graph.into_publishing()
    }
}
impl GraphView {
    /// Copy selected immutable values into an independent publishing graph.
    /// This preserves selected IDs/direction, not the source's branches or history.
    pub fn to_graph(&self) -> GraphResult<Graph> {
        Graph::from_view(self)
    }
}
