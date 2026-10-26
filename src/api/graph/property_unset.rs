//! Explicit property absence over the same prepared mutation boundary as assignments.
use super::*;

impl Graph {
    /// Remove a node property. Historical values and immutable views are retained.
    pub fn unset_node_attr(&mut self, node: NodeId, name: AttrName) -> GraphResult<()> {
        self.unset_node_attrs(HashMap::from([(name, vec![node])]))
    }

    /// Remove an edge property without assigning an explicit Null.
    pub fn unset_edge_attr(&mut self, edge: EdgeId, name: AttrName) -> GraphResult<()> {
        self.unset_edge_attrs(HashMap::from([(name, vec![edge])]))
    }

    /// Remove node property columns atomically; empty input is a no-op.
    pub fn unset_node_attrs(&mut self, columns: HashMap<AttrName, Vec<NodeId>>) -> GraphResult<()> {
        self.unset_properties(GraphBatch {
            unset_node_attrs: columns
                .into_iter()
                .map(|(name, ids)| {
                    (
                        name,
                        ids.into_iter().map(crate::BatchNodeRef::Existing).collect(),
                    )
                })
                .collect(),
            ..Default::default()
        })
    }

    /// Remove edge property columns atomically; empty input is a no-op.
    pub fn unset_edge_attrs(&mut self, columns: HashMap<AttrName, Vec<EdgeId>>) -> GraphResult<()> {
        self.unset_properties(GraphBatch {
            unset_edge_attrs: columns
                .into_iter()
                .map(|(name, ids)| {
                    (
                        name,
                        ids.into_iter().map(crate::BatchEdgeRef::Existing).collect(),
                    )
                })
                .collect(),
            ..Default::default()
        })
    }

    fn unset_properties(&mut self, batch: GraphBatch) -> GraphResult<()> {
        if !batch.has_operations() {
            return Ok(());
        }
        if self.automatic_publication {
            return self
                .apply_batch(
                    batch,
                    self.current_state_id(),
                    "unset_properties".into(),
                    "groggy".into(),
                )
                .map(|_| ());
        }
        let counts = self.pool.borrow().statistics();
        let prepared = batch.prepare(
            |id| self.has_node(id),
            |id| self.has_edge(id),
            |id| self.pool.borrow().get_edge_endpoints(id),
            counts.node_count,
            counts.edge_count,
        )?;
        self.validate_unsets(&prepared)?;
        self.execute_prepared_mutation(prepared).map(|_| ())
    }

    pub(super) fn validate_unsets(
        &self,
        batch: &crate::api::batch::PreparedBatch,
    ) -> GraphResult<()> {
        // A meta-node must keep its source reference. Ordinary user properties,
        // labels and relationship types may be absent, subject to a caller schema.
        if let Some(ids) = batch.unset_node_attrs.get("contains_subgraph") {
            for &id in ids {
                if self.has_node(id)
                    && self
                        .get_node_attr(id, &"entity_type".into())?
                        .is_some_and(|value| value.get_text().as_deref() == Some("meta"))
                {
                    return Err(GraphError::InvalidInput(format!(
                        "unset_node_attrs.contains_subgraph: meta node {id} requires its source reference")));
                }
            }
        }
        Ok(())
    }

    pub(super) fn execute_unsets(&mut self, columns: HashMap<AttrName, Vec<usize>>, is_node: bool) {
        for (name, ids) in columns {
            for id in ids {
                let old = self.space.unset_attr_index(id, &name, is_node);
                // Even an already absent key may need a backdated absence event.
                self.change_tracker
                    .record_attr_unset(id, name.clone(), old, is_node);
            }
        }
    }
}
