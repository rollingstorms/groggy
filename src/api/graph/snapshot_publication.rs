//! Shared exact snapshot publication for merges and tracked restoration.
use super::*;
use crate::state::{ChangeSet, GraphSnapshot};

/// Effective-time options for one exact snapshot publication. Recorded time is engine-owned.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphPublicationOptions {
    /// Batch default effective time; None uses the engine acceptance clock.
    pub effective_at: Option<crate::TimestampMicros>,
    /// Overrides for actual first-parent changes only.
    pub change_times: std::collections::BTreeMap<crate::ChangeKey, crate::TimestampMicros>,
}
/// Compatibility name for merge publication timing options.
pub type GraphMergeOptions = GraphPublicationOptions;

pub(crate) struct PreparedSnapshotPublication {
    pub metadata: crate::PublicationMetadata,
    snapshot: GraphSnapshot,
    changes: ChangeSet,
    branch: String,
    parents: Vec<StateId>,
    bindings: Vec<(String, ExternalEntityRef)>,
}
impl Graph {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_snapshot_publication(
        &self,
        snapshot: GraphSnapshot,
        changes: ChangeSet,
        left: StateId,
        branch: String,
        parents: Vec<StateId>,
        options: &GraphPublicationOptions,
        recorded_at: Option<i64>,
    ) -> GraphResult<PreparedSnapshotPublication> {
        let bindings = self.snapshot_identity_bindings(&snapshot)?;
        if !changes.edge_endpoint_changes.is_empty()
            && !self.change_tracker.supports_endpoint_changes()
        {
            return Err(GraphError::InvalidInput(
                "Temporal strategy does not support endpoint changes".into(),
            ));
        }
        let pool = self.pool.borrow();
        let rewired: HashSet<_> = snapshot
            .edges
            .iter()
            .filter_map(|(&id, &pair)| (pool.get_edge_endpoints(id) != Some(pair)).then_some(id))
            .collect();
        if pool.legacy_groups_own_edges(&rewired) {
            return Err(GraphError::InvalidInput("State publication endpoint changes require frozen group sources; edge belongs to a legacy membership-only group".into()));
        }
        drop(pool);
        let recorded_at = match recorded_at {
            Some(time) => time,
            None => crate::state::publication::now_micros()?,
        };
        let effective_at = options.effective_at.unwrap_or(recorded_at);
        let mut metadata = crate::PublicationMetadata {
            recorded_at,
            effective_at,
            change_times: Default::default(),
        };
        use crate::ChangeKey as K;
        let keys = changes
            .nodes_added
            .iter()
            .map(|&id| K::NodeAdded(id))
            .chain(changes.nodes_removed.iter().map(|&id| K::NodeRemoved(id)))
            .chain(
                changes
                    .edges_added
                    .iter()
                    .map(|&(id, _, _)| K::EdgeAdded(id)),
            )
            .chain(changes.edges_removed.iter().map(|&id| K::EdgeRemoved(id)))
            .chain(
                changes
                    .edge_endpoint_changes
                    .iter()
                    .map(|c| K::EdgeEndpoints(c.edge_id)),
            )
            .chain(
                changes
                    .node_attr_changes
                    .iter()
                    .map(|(id, name, _, _)| K::NodeAttribute(*id, name.clone())),
            )
            .chain(
                changes
                    .edge_attr_changes
                    .iter()
                    .map(|(id, name, _, _)| K::EdgeAttribute(*id, name.clone())),
            );
        for key in keys {
            let fallback = match &key {
                K::NodeAttribute(id, _) => metadata
                    .change_times
                    .get(&K::NodeAdded(*id))
                    .copied()
                    .unwrap_or(effective_at),
                K::EdgeAttribute(id, _) => metadata
                    .change_times
                    .get(&K::EdgeAdded(*id))
                    .copied()
                    .unwrap_or(effective_at),
                K::EdgeRemoved(id) => {
                    let (u, v) = self.edge_endpoints(*id)?;
                    [u, v]
                        .iter()
                        .filter_map(|id| metadata.change_times.get(&K::NodeRemoved(*id)))
                        .min()
                        .copied()
                        .unwrap_or(effective_at)
                }
                _ => effective_at,
            };
            metadata.change_times.insert(
                key.clone(),
                options.change_times.get(&key).copied().unwrap_or(fallback),
            );
        }
        if options
            .change_times
            .keys()
            .any(|key| !metadata.change_times.contains_key(key))
        {
            return Err(GraphError::InvalidInput(
                "State publication timestamp override does not target an actual resolution change"
                    .into(),
            ));
        }
        let additions = changes
            .edges_added
            .iter()
            .map(|&(id, u, v)| (id, (u, v)))
            .chain(
                changes
                    .edge_endpoint_changes
                    .iter()
                    .map(|c| (c.edge_id, c.new_endpoints)),
            )
            .collect();
        let patch = crate::state::history::TopologyPatch::from_metadata(
            &metadata,
            self.history.next_state_id(),
            &additions,
        )?;
        self.history.validate_effective_topology(
            left,
            &patch,
            options.effective_at.is_some()
                || options
                    .change_times
                    .keys()
                    .any(|key| !matches!(key, K::NodeAttribute(..) | K::EdgeAttribute(..))),
        )?;
        Ok(PreparedSnapshotPublication {
            metadata,
            snapshot,
            changes,
            branch,
            parents,
            bindings,
        })
    }
    pub(crate) fn execute_snapshot_publication(
        &mut self,
        prepared: PreparedSnapshotPublication,
        message: String,
        author: String,
    ) -> GraphResult<StateId> {
        let PreparedSnapshotPublication {
            metadata,
            mut snapshot,
            changes,
            branch,
            parents,
            bindings,
        } = prepared;
        let id = if parents.len() == 1 {
            self.history.create_commit_with_publication(
                changes,
                message,
                author,
                Some(parents[0]),
                Some(metadata),
            )?
        } else {
            self.history.create_merge_commit_with_publication(
                changes,
                message,
                author,
                parents,
                Some(metadata),
            )?
        };
        snapshot.state_id = id;
        self.install_snapshot_payload(snapshot, bindings)?;
        self.history.update_branch_head(&branch, id)?;
        self.current_commit = id;
        Ok(id)
    }
}

pub(super) fn exact_snapshot_changes(
    before: &GraphSnapshot,
    after: &GraphSnapshot,
) -> GraphResult<crate::state::ChangeSet> {
    use std::collections::BTreeSet;
    let mut changes = crate::state::ChangeSet::new();
    let old_nodes: HashSet<_> = before.active_nodes.iter().copied().collect();
    let new_nodes: HashSet<_> = after.active_nodes.iter().copied().collect();
    changes.nodes_added = new_nodes.difference(&old_nodes).copied().collect();
    changes.nodes_removed = old_nodes.difference(&new_nodes).copied().collect();
    changes.nodes_added.sort_unstable();
    changes.nodes_removed.sort_unstable();
    for (&id, &(u, v)) in &after.edges {
        match before.edges.get(&id) {
            None => changes.edges_added.push((id, u, v)),
            Some(&old) if old != (u, v) => {
                changes
                    .edge_endpoint_changes
                    .push(crate::EdgeEndpointChange {
                        edge_id: id,
                        old_endpoints: old,
                        new_endpoints: (u, v),
                    })
            }
            _ => {}
        }
    }
    changes.edges_added.sort_unstable();
    changes.edges_removed = before
        .edges
        .keys()
        .filter(|id| !after.edges.contains_key(id))
        .copied()
        .collect();
    changes.edges_removed.sort_unstable();
    changes
        .edge_endpoint_changes
        .sort_unstable_by_key(|c| c.edge_id);
    for (nodes, old_rows, new_rows, output) in [
        (
            true,
            &before.node_attributes,
            &after.node_attributes,
            &mut changes.node_attr_changes,
        ),
        (
            false,
            &before.edge_attributes,
            &after.edge_attributes,
            &mut changes.edge_attr_changes,
        ),
    ] {
        let ids: BTreeSet<_> = if nodes {
            new_nodes.iter().copied().collect()
        } else {
            after.edges.keys().copied().collect()
        };
        for id in ids {
            let was_active = if nodes {
                old_nodes.contains(&id)
            } else {
                before.edges.contains_key(&id)
            };
            let old = was_active.then(|| old_rows.get(&id)).flatten();
            let new = new_rows.get(&id);
            let names: BTreeSet<_> = old
                .into_iter()
                .flat_map(|row| row.keys())
                .chain(new.into_iter().flat_map(|row| row.keys()))
                .collect();
            for name in names {
                let a = old.and_then(|r| r.get(name));
                let b = new.and_then(|r| r.get(name));
                let equal = match (a, b) {
                    (None, None) => true,
                    (Some(a), Some(b)) => archive::same_value(a, b)?,
                    _ => false,
                };
                if !equal {
                    output.push((id, name.clone(), a.cloned(), b.cloned()));
                }
            }
        }
    }
    Ok(changes)
}
