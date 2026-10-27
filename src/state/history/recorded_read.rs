//! Owned recorded-state reconstruction and diff inputs, independent of an owner.
use super::*;

/// Immutable logical reconstruction input. It owns retained delta records, not
/// a mutable graph, history lookup authority, or frozen group-source registry.
/// Preparation walks ancestry; execution resolves property payloads off-owner.
#[derive(Debug, Clone)]
pub struct RecordedReadPlan {
    graph_type: GraphType,
    commits: Arc<[Arc<Commit>]>,
}
impl RecordedReadPlan {
    /// Reconstruct the captured first-parent lineage. Later owner changes,
    /// history collection and owner destruction do not alter the input.
    pub fn execute(&self) -> GraphResult<GraphSnapshot> {
        let base = GraphSnapshot::empty_with_type(0, self.graph_type);
        GraphSnapshot::reconstruct_from_delta_refs(
            Some(&base),
            self.commits
                .iter()
                .map(|commit| (commit.delta.as_ref(), commit.id)),
        )
    }
}

/// Two immutable logical lineages for detached recorded-state comparison.
#[derive(Debug, Clone)]
pub struct CommitDiffPlan {
    from: StateId,
    to: StateId,
    before: RecordedReadPlan,
    after: RecordedReadPlan,
}
impl CommitDiffPlan {
    /// Resolve both states and compare typed values/topology without an owner.
    pub fn execute(&self) -> GraphResult<CommitDiff> {
        if self.from == self.to {
            return Ok(CommitDiff {
                from_commit: self.from,
                to_commit: self.to,
                nodes_added: Vec::new(),
                nodes_removed: Vec::new(),
                edges_added: Vec::new(),
                edges_removed: Vec::new(),
                edge_endpoint_changes: Vec::new(),
                attr_changes: Vec::new(),
            });
        }
        let diff = self
            .before
            .execute()?
            .try_diff_with(&self.after.execute()?)?;
        Ok(CommitDiff {
            from_commit: self.from,
            to_commit: self.to,
            nodes_added: diff.nodes_added,
            nodes_removed: diff.nodes_removed,
            edges_added: diff.edges_added,
            edges_removed: diff.edges_removed,
            edge_endpoint_changes: diff.edge_endpoint_changes,
            attr_changes: diff.attribute_changes,
        })
    }
}
impl HistoryForest {
    /// Capture a first-parent lineage through the shared immutable read cache. No graph
    /// state or raw property payload is reconstructed/copied during preparation.
    pub fn prepare_recorded_read(&self, head: StateId) -> GraphResult<RecordedReadPlan> {
        let commits = if head == 0 {
            // Do not evict a useful cached lineage for the universal empty root.
            Arc::from([])
        } else {
            self.read_context(head)?.recorded_commits()
        };
        Ok(RecordedReadPlan {
            graph_type: self.graph_type,
            commits,
        })
    }
    /// Capture both selections before releasing the owner/runtime lock.
    pub fn prepare_commit_diff(&self, from: StateId, to: StateId) -> GraphResult<CommitDiffPlan> {
        let before = self.prepare_recorded_read(from)?;
        let after = if from == to {
            before.clone()
        } else {
            self.prepare_recorded_read(to)?
        };
        Ok(CommitDiffPlan {
            from,
            to,
            before,
            after,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_recorded_ancestry_rejects_cycles_and_missing_parents() {
        let mut history = HistoryForest::new();
        let id = history
            .create_commit(
                ChangeSet {
                    nodes_added: vec![0],
                    ..ChangeSet::new()
                },
                "root".into(),
                "test".into(),
                None,
            )
            .unwrap();
        Arc::make_mut(history.commits.get_mut(&id).unwrap()).parents = vec![id];
        assert!(history
            .prepare_recorded_read(id)
            .unwrap_err()
            .to_string()
            .contains("Cycle"));
        Arc::make_mut(history.commits.get_mut(&id).unwrap()).parents = vec![999];
        assert!(history.prepare_recorded_read(id).is_err());
    }
    #[test]
    fn identical_validated_states_share_lineage_and_skip_reconstruction() {
        let mut history = HistoryForest::new();
        let id = history
            .create_commit(
                ChangeSet {
                    nodes_added: vec![0],
                    ..ChangeSet::new()
                },
                "root".into(),
                "test".into(),
                None,
            )
            .unwrap();
        let plan = history.prepare_commit_diff(id, id).unwrap();
        assert!(Arc::ptr_eq(&plan.before.commits, &plan.after.commits));
        let cloned = plan.clone();
        assert!(Arc::ptr_eq(&plan.before.commits, &cloned.before.commits));
        drop(history);
        let diff = plan.execute().unwrap();
        assert!(diff.nodes_added.is_empty() && diff.attr_changes.is_empty());
        assert_eq!((diff.from_commit, diff.to_commit), (id, id));
        assert!(HistoryForest::new().prepare_commit_diff(999, 999).is_err());
    }
    #[test]
    fn recorded_effective_and_zero_reads_share_the_existing_lineage_cache() {
        let mut history = HistoryForest::new();
        let id = history
            .create_commit(
                ChangeSet {
                    nodes_added: vec![0],
                    ..ChangeSet::new()
                },
                "root".into(),
                "test".into(),
                None,
            )
            .unwrap();
        let context = history.read_context(id).unwrap();
        let recorded = history.prepare_recorded_read(id).unwrap();
        assert!(Arc::ptr_eq(&context.recorded_commits(), &recorded.commits));
        history.prepare_recorded_read(0).unwrap();
        assert!(Arc::ptr_eq(&context, &history.read_context(id).unwrap()));
    }
}
