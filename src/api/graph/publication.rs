//! Build logical change times before any batch mutation can allocate or write.
use super::*;
use crate::api::batch::PreparedBatch;
use crate::state::publication::{now_micros, ChangeKey, PublicationMetadata};
use std::collections::BTreeMap;
impl Graph {
    pub(super) fn batch_publication(
        &self,
        batch: &PreparedBatch,
        first_node: NodeId,
        first_edge: EdgeId,
    ) -> GraphResult<PublicationMetadata> {
        self.batch_publication_at(batch, first_node, first_edge, now_micros()?)
    }

    pub(super) fn batch_publication_at(
        &self,
        batch: &PreparedBatch,
        first_node: NodeId,
        first_edge: EdgeId,
        recorded_at: crate::TimestampMicros,
    ) -> GraphResult<PublicationMetadata> {
        let effective_at = batch.effective_at.unwrap_or(recorded_at);
        let time = |key: &ChangeKey| batch.change_times.get(key).copied().unwrap_or(effective_at);
        let mut change_times = BTreeMap::new();
        for id in first_node..first_node + batch.add_nodes {
            let key = ChangeKey::NodeAdded(id);
            let at = time(&key);
            change_times.insert(key, at);
            change_times.insert(ChangeKey::NodeAttribute(id, "entity_type".into()), at);
        }
        let mut removal_times = HashMap::new();
        for &id in &batch.remove_nodes {
            let key = ChangeKey::NodeRemoved(id);
            let at = time(&key);
            change_times.insert(key, at);
            removal_times.insert(id, at);
        }
        let removed_nodes = removal_times.keys().copied().collect();
        for (edge, source, target) in self.incident_edges_for_nodes(&removed_nodes)? {
            // A shared implicit cascade takes the earliest removed endpoint time.
            // Explicit edge removals below retain their own override precedence.
            let at = [source, target]
                .iter()
                .filter_map(|node| removal_times.get(node))
                .min()
                .copied()
                .ok_or_else(|| GraphError::InvalidInput("Missing cascade removal time".into()))?;
            change_times.insert(ChangeKey::EdgeRemoved(edge), at);
        }
        for id in first_edge..first_edge + batch.add_edges.len() {
            let key = ChangeKey::EdgeAdded(id);
            change_times.insert(key.clone(), time(&key));
        }
        for &id in &batch.remove_edges {
            // An explicitly staged edge removal controls its own effective time.
            let key = ChangeKey::EdgeRemoved(id);
            change_times.insert(key.clone(), time(&key));
        }
        for (is_node, name, id) in batch
            .node_attrs
            .iter()
            .flat_map(|(name, rows)| rows.iter().map(move |(id, _)| (true, name, *id)))
            .chain(
                batch
                    .edge_attrs
                    .iter()
                    .flat_map(|(name, rows)| rows.iter().map(move |(id, _)| (false, name, *id))),
            )
            .chain(
                batch
                    .unset_node_attrs
                    .iter()
                    .flat_map(|(name, ids)| ids.iter().map(move |id| (true, name, *id))),
            )
            .chain(
                batch
                    .unset_edge_attrs
                    .iter()
                    .flat_map(|(name, ids)| ids.iter().map(move |id| (false, name, *id))),
            )
        {
            let (key, added) = if is_node {
                (
                    ChangeKey::NodeAttribute(id, name.clone()),
                    ChangeKey::NodeAdded(id),
                )
            } else {
                (
                    ChangeKey::EdgeAttribute(id, name.clone()),
                    ChangeKey::EdgeAdded(id),
                )
            };
            let fallback = change_times.get(&added).copied().unwrap_or(effective_at);
            change_times.insert(
                key.clone(),
                batch.change_times.get(&key).copied().unwrap_or(fallback),
            );
        }
        Ok(PublicationMetadata {
            recorded_at,
            effective_at,
            change_times,
        })
    }
}

impl Graph {
    pub(crate) fn restore_publication_clock(
        &self,
        prepared: &mut crate::api::batch::PreparedPublication,
        metadata: PublicationMetadata,
    ) -> GraphResult<()> {
        let allocated = self.pool.borrow().statistics();
        let expected = self.batch_publication_at(
            &prepared.batch,
            allocated.node_count,
            allocated.edge_count,
            metadata.recorded_at,
        )?;
        if expected != metadata {
            return Err(GraphError::InvalidInput(
                "Journal publication metadata does not match staged changes".into(),
            ));
        }
        self.validate_publication_topology(&prepared.batch, &metadata, allocated.edge_count)?;
        prepared.metadata = metadata;
        Ok(())
    }
}

impl Graph {
    pub(super) fn validate_publication_topology(
        &self,
        batch: &PreparedBatch,
        metadata: &PublicationMetadata,
        first_edge: EdgeId,
    ) -> GraphResult<()> {
        // Property-only writes do not inspect ancestry or lifecycle indexes.
        if batch.add_nodes == 0
            && batch.add_edges.is_empty()
            && batch.remove_nodes.is_empty()
            && batch.remove_edges.is_empty()
        {
            return Ok(());
        }
        let additions = batch
            .add_edges
            .iter()
            .enumerate()
            .map(|(offset, &endpoints)| (first_edge + offset, endpoints))
            .collect();
        let patch = crate::state::history::TopologyPatch::from_metadata(
            metadata,
            self.history.next_state_id(),
            &additions,
        )?;
        self.history.validate_effective_topology(
            self.current_commit,
            &patch,
            batch.effective_at.is_some()
                || batch.change_times.keys().any(|key| {
                    matches!(
                        key,
                        ChangeKey::NodeAdded(_)
                            | ChangeKey::NodeRemoved(_)
                            | ChangeKey::EdgeAdded(_)
                            | ChangeKey::EdgeRemoved(_)
                    )
                }),
        )
    }
}
