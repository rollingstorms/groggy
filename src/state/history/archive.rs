//! Versioned archive representation; never serialize derived indexes or owner pins.
use super::*;
use crate::storage::{pool::GraphPool, retained_group};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

type StoredChange = (usize, String, Option<Value>, Option<Value>);
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredCommit {
    id: StateId,
    parents: Vec<StateId>,
    message: String,
    author: String,
    timestamp: u64,
    publication: Option<crate::PublicationMetadata>,
    content_hash: [u8; 32],
    delta_hash: [u8; 32],
    nodes_added: Vec<NodeId>,
    nodes_removed: Vec<NodeId>,
    edges_added: Vec<(EdgeId, NodeId, NodeId)>,
    edges_removed: Vec<EdgeId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    edge_endpoint_changes: Vec<crate::EdgeEndpointChange>,
    node_attr_changes: Vec<StoredChange>,
    edge_attr_changes: Vec<StoredChange>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HistoryDocument {
    commits: Vec<StoredCommit>,
    branches: BTreeMap<BranchName, StateId>,
    tags: BTreeMap<String, StateId>,
    next_state_id: StateId,
    checkpoints: Vec<crate::Checkpoint>,
}
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(format!("Invalid graph archive history: {message}"))
}
fn encode_changes(
    changes: &[(usize, String, Option<AttrValue>, Option<AttrValue>)],
) -> GraphResult<Vec<StoredChange>> {
    changes
        .iter()
        .map(|(id, name, old, new)| {
            Ok((
                *id,
                name.clone(),
                old.as_ref().map(retained_group::encode).transpose()?,
                new.as_ref().map(retained_group::encode).transpose()?,
            ))
        })
        .collect()
}
fn decode_changes(
    changes: Vec<StoredChange>,
    is_node: bool,
    pool: &GraphPool,
) -> GraphResult<Vec<(usize, String, Option<AttrValue>, Option<AttrValue>)>> {
    let mut seen = HashSet::new();
    changes
        .into_iter()
        .map(|(id, name, old, new)| {
            if !(if is_node {
                pool.is_node_allocated(id)
            } else {
                pool.is_edge_allocated(id)
            }) || !seen.insert((id, name.clone()))
            {
                return Err(invalid("duplicate or unallocated property target"));
            }
            let old = old.map(retained_group::decode).transpose()?;
            let new = new.map(retained_group::decode).transpose()?;
            for value in old.iter().chain(new.iter()) {
                pool.validate_archive_value(value)?;
            }
            Ok((id, name, old, new))
        })
        .collect()
}
impl HistoryForest {
    pub(crate) fn archive_document(&self) -> GraphResult<HistoryDocument> {
        let mut commits: Vec<_> = self.commits.values().collect();
        commits.sort_unstable_by_key(|c| c.id);
        let commits = commits
            .into_iter()
            .map(|c| {
                let d = &c.delta;
                Ok(StoredCommit {
                    id: c.id,
                    parents: c.parents.clone(),
                    message: c.message.clone(),
                    author: c.author.clone(),
                    timestamp: c.timestamp,
                    publication: c.publication.clone(),
                    content_hash: c.content_hash,
                    delta_hash: d.content_hash,
                    nodes_added: d.nodes_added.clone(),
                    nodes_removed: d.nodes_removed.clone(),
                    edges_added: d.edges_added.clone(),
                    edges_removed: d.edges_removed.clone(),
                    edge_endpoint_changes: d.edge_endpoint_changes.clone(),
                    node_attr_changes: encode_changes(&d.node_attr_changes)?,
                    edge_attr_changes: encode_changes(&d.edge_attr_changes)?,
                })
            })
            .collect::<GraphResult<_>>()?;
        Ok(HistoryDocument {
            commits,
            branches: self.branches.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            tags: self.tags.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            next_state_id: self.next_state_id,
            checkpoints: self.checkpoints.clone(),
        })
    }
}
impl HistoryDocument {
    pub(crate) fn restore(self, pool: &GraphPool) -> GraphResult<HistoryForest> {
        let mut history = HistoryForest::new_with_type(pool.graph_type());
        let mut previous = 0;
        for c in self.commits {
            if c.id == 0 || c.id <= previous || c.id >= self.next_state_id {
                return Err(invalid(
                    "versions are unsorted, duplicated or exceed allocation",
                ));
            }
            let mut parents = HashSet::new();
            for &parent in &c.parents {
                if parent >= c.id
                    || (parent != 0 && !history.commits.contains_key(&parent))
                    || !parents.insert(parent)
                {
                    return Err(invalid("missing, cyclic or duplicate parent"));
                }
            }
            let unique_ids = |ids: &[usize], is_node: bool| {
                let mut seen = HashSet::new();
                ids.iter().all(|id| {
                    (if is_node {
                        pool.is_node_allocated(*id)
                    } else {
                        pool.is_edge_allocated(*id)
                    }) && seen.insert(*id)
                })
            };
            if !unique_ids(&c.nodes_added, true)
                || !unique_ids(&c.nodes_removed, true)
                || !unique_ids(&c.edges_removed, false)
                || !unique_ids(
                    &c.edges_added.iter().map(|e| e.0).collect::<Vec<_>>(),
                    false,
                )
                || c.edges_added
                    .iter()
                    .any(|(_, u, v)| !pool.is_node_allocated(*u) || !pool.is_node_allocated(*v))
            {
                return Err(invalid("duplicate or unallocated topology target"));
            }
            let changes = ChangeSet {
                nodes_added: c.nodes_added,
                nodes_removed: c.nodes_removed,
                edges_added: c.edges_added,
                edges_removed: c.edges_removed,
                edge_endpoint_changes: c.edge_endpoint_changes,
                node_attr_changes: decode_changes(c.node_attr_changes, true, pool)?,
                edge_attr_changes: decode_changes(c.edge_attr_changes, false, pool)?,
            };
            let mut endpoint_ids = HashSet::new();
            for change in &changes.edge_endpoint_changes {
                if !endpoint_ids.insert(change.edge_id)
                    || !pool.is_edge_allocated(change.edge_id)
                    || changes.edges_added.iter().any(|e| e.0 == change.edge_id)
                    || changes.edges_removed.contains(&change.edge_id)
                    || [
                        change.old_endpoints.0,
                        change.old_endpoints.1,
                        change.new_endpoints.0,
                        change.new_endpoints.1,
                    ]
                    .iter()
                    .any(|&id| !pool.is_node_allocated(id))
                {
                    return Err(invalid("invalid endpoint revision referents"));
                }
            }
            let delta = Delta::from_changes(changes);
            if delta.content_hash != c.delta_hash {
                return Err(invalid("delta content hash mismatch"));
            }
            if let Some(metadata) = &c.publication {
                use crate::ChangeKey;
                let keys: std::collections::BTreeSet<_> =
                    delta
                        .nodes_added
                        .iter()
                        .copied()
                        .map(ChangeKey::NodeAdded)
                        .chain(
                            delta
                                .nodes_removed
                                .iter()
                                .copied()
                                .map(ChangeKey::NodeRemoved),
                        )
                        .chain(
                            delta
                                .edges_added
                                .iter()
                                .map(|(id, _, _)| ChangeKey::EdgeAdded(*id)),
                        )
                        .chain(
                            delta
                                .edges_removed
                                .iter()
                                .copied()
                                .map(ChangeKey::EdgeRemoved),
                        )
                        .chain(
                            delta.node_attr_changes.iter().map(|(id, name, _, _)| {
                                ChangeKey::NodeAttribute(*id, name.clone())
                            }),
                        )
                        .chain(
                            delta.edge_attr_changes.iter().map(|(id, name, _, _)| {
                                ChangeKey::EdgeAttribute(*id, name.clone())
                            }),
                        )
                        .chain(
                            delta
                                .edge_endpoint_changes
                                .iter()
                                .map(|c| ChangeKey::EdgeEndpoints(c.edge_id)),
                        )
                        .collect();
                if keys != metadata.change_times.keys().cloned().collect()
                    || c.timestamp != (metadata.recorded_at.max(0) as u64) / 1_000_000
                {
                    return Err(invalid("publication clocks do not match logical changes"));
                }
            }
            let delta = history.intern_delta(delta);
            let commit = Commit {
                id: c.id,
                parents: c.parents,
                delta,
                message: c.message,
                author: c.author,
                timestamp: c.timestamp,
                publication: c.publication,
                content_hash: c.content_hash,
            };
            // Validate the stored hash without inventing a new engine acceptance clock.
            use std::hash::{Hash, Hasher};
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            commit.id.hash(&mut hash);
            commit.parents.hash(&mut hash);
            commit.delta.content_hash.hash(&mut hash);
            commit.message.hash(&mut hash);
            commit.author.hash(&mut hash);
            commit.timestamp.hash(&mut hash);
            commit.publication.hash(&mut hash);
            let expected = hash.finish().to_le_bytes().repeat(4);
            if expected.as_slice() != commit.content_hash {
                return Err(invalid("version content hash mismatch"));
            }
            for &parent in &commit.parents {
                history.children.entry(parent).or_default().push(commit.id);
            }
            if commit.parents.is_empty() {
                history.roots.insert(commit.id);
            }
            previous = commit.id;
            history.commits.insert(commit.id, Arc::new(commit));
        }
        if self.next_state_id == 0
            || self.next_state_id == StateId::MAX
            || !self.branches.contains_key("main")
        {
            return Err(invalid("invalid version allocation or missing main branch"));
        }
        for (name, &head) in self.branches.iter().chain(&self.tags) {
            if name.is_empty() || (head != 0 && !history.commits.contains_key(&head)) {
                return Err(invalid("invalid branch/tag reference"));
            }
        }
        for checkpoint in &self.checkpoints {
            if checkpoint.state_id != 0 && !history.commits.contains_key(&checkpoint.state_id) {
                return Err(invalid("checkpoint refers to missing version"));
            }
        }
        history.next_state_id = self.next_state_id;
        history.branches = self.branches.into_iter().collect();
        history.tags = self.tags.into_iter().collect();
        history.checkpoints = self.checkpoints;
        let heads: HashSet<_> = history
            .branches
            .values()
            .chain(history.tags.values())
            .copied()
            .collect();
        for head in heads {
            crate::state::PinnedReadState::from_snapshot(history.reconstruct_state_at(head)?)?;
        }
        Ok(history)
    }
}
