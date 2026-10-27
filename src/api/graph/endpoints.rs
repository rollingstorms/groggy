//! Checked endpoint revisions retain edge identity and all properties.
use super::*;
use crate::BatchNodeRef;
impl Graph {
    /// Revise one edge through this graph's publication policy.
    pub fn set_edge_endpoints(
        &mut self,
        edge: EdgeId,
        source: NodeId,
        target: NodeId,
    ) -> GraphResult<StateId> {
        self.set_edge_endpoints_bulk(vec![(edge, source, target)])
    }
    /// Revise all pairs atomically; manual graphs retain pending changes until acceptance.
    pub fn set_edge_endpoints_bulk(
        &mut self,
        rows: Vec<(EdgeId, NodeId, NodeId)>,
    ) -> GraphResult<StateId> {
        if rows.is_empty() {
            return Ok(self.current_state_id());
        }
        if self.automatic_publication {
            return self.writer("groggy").set_edge_endpoints_bulk(rows);
        }
        let allocated = self.pool.borrow().statistics();
        let batch = GraphBatch {
            edge_endpoints: rows
                .into_iter()
                .map(|(id, u, v)| (id, BatchNodeRef::Existing(u), BatchNodeRef::Existing(v)))
                .collect(),
            ..Default::default()
        };
        let prepared = batch.prepare(
            |id| self.has_node(id),
            |id| self.has_edge(id),
            |id| self.pool.borrow().get_edge_endpoints(id),
            allocated.node_count,
            allocated.edge_count,
        )?;
        self.validate_endpoint_revisions(&prepared)?;
        self.execute_prepared_mutation(prepared)?;
        Ok(self.current_state_id())
    }
    pub(super) fn validate_endpoint_revisions(
        &self,
        prepared: &crate::api::batch::PreparedBatch,
    ) -> GraphResult<()> {
        if !prepared.edge_endpoint_changes.is_empty() {
            if !self.change_tracker.supports_endpoint_changes() {
                return Err(GraphError::InvalidInput(
                    "Temporal strategy does not support endpoint changes".into(),
                ));
            }
            let changed = prepared
                .edge_endpoint_changes
                .iter()
                .map(|c| c.edge_id)
                .collect();
            if self.pool.borrow().legacy_groups_own_edges(&changed) {
                return Err(GraphError::InvalidInput("Endpoint changes require frozen group sources; edge belongs to a legacy membership-only group".into()));
            }
        }
        Ok(())
    }
    /// Prepare ordered endpoint reversal without changing IDs or graph direction.
    pub fn edge_reversal_batch(&self, edges: &[EdgeId]) -> GraphResult<GraphBatch> {
        let mut rows = Vec::with_capacity(edges.len());
        for &id in edges {
            if !self.has_edge(id) {
                return Err(GraphError::InvalidInput(format!("Edge {id} is not active")));
            }
            let (u, v) = self.edge_endpoints(id)?;
            rows.push((id, BatchNodeRef::Existing(v), BatchNodeRef::Existing(u)));
        }
        Ok(GraphBatch {
            edge_endpoints: rows,
            ..Default::default()
        })
    }
    /// Reverse selected edges in one checked publication, preserving all edge data.
    pub fn reverse_edges(&mut self, edges: &[EdgeId]) -> GraphResult<StateId> {
        if edges.is_empty() {
            return Ok(self.current_state_id());
        }
        let mut rows = Vec::with_capacity(edges.len());
        for &id in edges {
            let (u, v) = self.edge_endpoints(id)?;
            rows.push((id, v, u));
        }
        self.set_edge_endpoints_bulk(rows)
    }
}
