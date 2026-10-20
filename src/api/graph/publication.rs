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
        let recorded_at = now_micros()?;
        let effective_at = batch.effective_at.unwrap_or(recorded_at);
        let time = |key: &ChangeKey| batch.change_times.get(key).copied().unwrap_or(effective_at);
        let mut change_times = BTreeMap::new();
        for id in first_node..first_node + batch.add_nodes {
            let key = ChangeKey::NodeAdded(id);
            let at = time(&key);
            change_times.insert(key, at);
            change_times.insert(ChangeKey::NodeAttribute(id, "entity_type".into()), at);
        }
        for &id in &batch.remove_nodes {
            let key = ChangeKey::NodeRemoved(id);
            let at = time(&key);
            change_times.insert(key, at);
            if self.has_node(id) {
                for edge in self.incident_edges(id)? {
                    // An implicit edge removal takes the earliest removed endpoint time.
                    change_times
                        .entry(ChangeKey::EdgeRemoved(edge))
                        .and_modify(|old| *old = (*old).min(at))
                        .or_insert(at);
                }
            }
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
        for (name, assignments) in &batch.node_attrs {
            for (id, _) in assignments {
                let key = ChangeKey::NodeAttribute(*id, name.clone());
                let fallback = change_times
                    .get(&ChangeKey::NodeAdded(*id))
                    .copied()
                    .unwrap_or(effective_at);
                change_times.insert(
                    key.clone(),
                    batch.change_times.get(&key).copied().unwrap_or(fallback),
                );
            }
        }
        for (name, assignments) in &batch.edge_attrs {
            for (id, _) in assignments {
                let key = ChangeKey::EdgeAttribute(*id, name.clone());
                let fallback = change_times
                    .get(&ChangeKey::EdgeAdded(*id))
                    .copied()
                    .unwrap_or(effective_at);
                change_times.insert(
                    key.clone(),
                    batch.change_times.get(&key).copied().unwrap_or(fallback),
                );
            }
        }
        Ok(PublicationMetadata {
            recorded_at,
            effective_at,
            change_times,
        })
    }
}
