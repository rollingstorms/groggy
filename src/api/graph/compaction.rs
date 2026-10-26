//! Explicit exact-value compaction with guarded detached execution.
use super::*;
use crate::storage::pool::compaction::{self, Columns, CompactedColumns};

type Mappings = HashMap<AttrName, HashMap<usize, usize>>;

/// Raw property slot counts before and after explicit deduplication. These are
/// not byte/RSS counts. Distinct values remain unless reclamation is explicit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolCompactionReport {
    /// Accepted graph state, unchanged by maintenance.
    pub selected_state: StateId,
    /// Raw node slots before deduplication.
    pub previous_node_values: usize,
    /// Raw node slots after deduplication.
    pub current_node_values: usize,
    /// Raw edge slots before deduplication.
    pub previous_edge_values: usize,
    /// Raw edge slots after deduplication.
    pub current_edge_values: usize,
    /// Unreferenced raw node slots discarded only by explicit reclamation.
    pub reclaimed_unreferenced_node_values: usize,
    /// Unreferenced raw edge slots discarded only by explicit reclamation.
    pub reclaimed_unreferenced_edge_values: usize,
}
/// Owned raw-column capture for detached compaction. Capture copies values on
/// the owner thread. Finish rejects a stale or foreign owner without mutation.
pub struct PoolCompactionPlan {
    guard: Guard,
    nodes: Columns,
    edges: Columns,
    retained: Option<Vec<Arc<crate::state::history::Commit>>>,
}
/// Detached result; install only on the unchanged source owner.
pub struct PoolCompactionOutcome {
    guard: Guard,
    nodes: CompactedColumns,
    edges: CompactedColumns,
}
struct Guard {
    identity: Arc<()>,
    state: StateId,
    nodes: Mappings,
    edges: Mappings,
    node_lengths: HashMap<AttrName, usize>,
    edge_lengths: HashMap<AttrName, usize>,
    history_extent: Option<(StateId, usize)>,
}
impl PoolCompactionPlan {
    /// Deduplicate exact values outside the owner/runtime lock. No graph changes.
    pub fn execute(self) -> GraphResult<PoolCompactionOutcome> {
        let roots = |is_node| {
            self.retained
                .as_ref()
                .map(|commits| compaction::RetainedRoots {
                    mappings: if is_node {
                        &self.guard.nodes
                    } else {
                        &self.guard.edges
                    },
                    values: commits
                        .iter()
                        .flat_map(|commit| {
                            let changes = if is_node {
                                &commit.delta.node_attr_changes
                            } else {
                                &commit.delta.edge_attr_changes
                            };
                            changes.iter().flat_map(|(_, name, old, new)| {
                                old.iter().chain(new.iter()).map(move |value| (name, value))
                            })
                        })
                        .collect(),
                })
        };
        let nodes = compaction::compact(self.nodes, roots(true))?;
        let edges = compaction::compact(self.edges, roots(false))?;
        Ok(PoolCompactionOutcome {
            guard: self.guard,
            nodes,
            edges,
        })
    }
}
impl PoolCompactionOutcome {
    /// Slot counts for this detached result. Installation still requires source validation.
    pub fn report(&self) -> PoolCompactionReport {
        PoolCompactionReport {
            selected_state: self.guard.state,
            previous_node_values: self.nodes.previous,
            current_node_values: self.nodes.current,
            previous_edge_values: self.edges.previous,
            current_edge_values: self.edges.current,
            reclaimed_unreferenced_node_values: self.nodes.reclaimed,
            reclaimed_unreferenced_edge_values: self.edges.reclaimed,
        }
    }
    pub(crate) fn rewrite_archive(
        &self,
        archive: &mut super::archive::GraphArchive,
    ) -> GraphResult<()> {
        archive.replace_compacted_columns(&self.nodes, &self.edges)
    }
}
impl Graph {
    /// Remove duplicate raw property slots while preserving all distinct values,
    /// states, branches, inactive IDs, groups and identity reservations. Requires
    /// a clean head. Existing raw offsets are invalidated; read pins are independent.
    pub fn compact_pool(&mut self) -> GraphResult<PoolCompactionReport> {
        self.compact_pool_with_reclamation(false)
    }
    /// Explicitly discard raw slots not referenced by any working mapping or any
    /// retained delta before/after value when reclaim_unreferenced is true. This
    /// never prunes history or releases entities/groups/identity reservations.
    pub fn compact_pool_with_reclamation(
        &mut self,
        reclaim_unreferenced: bool,
    ) -> GraphResult<PoolCompactionReport> {
        let outcome = self
            .prepare_pool_compaction_with_reclamation(reclaim_unreferenced)?
            .execute()?;
        self.finish_pool_compaction(outcome)
    }
    /// Capture retained columns for detached compaction. This explicit operation
    /// can temporarily duplicate the raw property payload in memory.
    pub fn prepare_pool_compaction(&self) -> GraphResult<PoolCompactionPlan> {
        self.prepare_pool_compaction_with_reclamation(false)
    }
    /// Capture all retained history roots for explicit raw-value reclamation.
    /// Old and new delta values protect historical/effective-time reconstruction;
    /// frozen source pins own their independent values and are not modified.
    pub fn prepare_pool_compaction_with_reclamation(
        &self,
        reclaim_unreferenced: bool,
    ) -> GraphResult<PoolCompactionPlan> {
        self.validate_write_head(self.current_state_id())?;
        let pool = self.pool.borrow();
        Ok(PoolCompactionPlan {
            guard: Guard {
                identity: Arc::clone(self.history.identity()),
                state: self.current_state_id(),
                nodes: self.space.retained_attr_indices(true).clone(),
                edges: self.space.retained_attr_indices(false).clone(),
                node_lengths: pool.compaction_lengths(true),
                edge_lengths: pool.compaction_lengths(false),
                history_extent: reclaim_unreferenced.then(|| {
                    (
                        self.history.next_state_id(),
                        self.history.retained_version_count(),
                    )
                }),
            },
            nodes: pool.compaction_columns(true),
            edges: pool.compaction_columns(false),
            retained: reclaim_unreferenced.then(|| self.history.get_commit_history()),
        })
    }
    /// Atomically install a successful result only if the source state, columns
    /// and mappings are unchanged. No fallible work remains after validation.
    pub fn finish_pool_compaction(
        &mut self,
        outcome: PoolCompactionOutcome,
    ) -> GraphResult<PoolCompactionReport> {
        self.validate_write_head(outcome.guard.state)?;
        let pool = self.pool.borrow();
        if !Arc::ptr_eq(self.history.identity(), &outcome.guard.identity)
            || outcome.guard.history_extent.is_some_and(|extent| {
                extent
                    != (
                        self.history.next_state_id(),
                        self.history.retained_version_count(),
                    )
            })
            || self.space.retained_attr_indices(true) != &outcome.guard.nodes
            || self.space.retained_attr_indices(false) != &outcome.guard.edges
            || pool.compaction_lengths(true) != outcome.guard.node_lengths
            || pool.compaction_lengths(false) != outcome.guard.edge_lengths
        {
            return Err(GraphError::InvalidInput(
                "Stale or foreign pool compaction result".into(),
            ));
        }
        drop(pool);
        let remap = |mappings: Mappings, columns: &CompactedColumns| -> GraphResult<Mappings> {
            mappings
                .into_iter()
                .map(|(name, rows)| {
                    let rows = rows
                        .into_iter()
                        .map(|(id, index)| Ok((id, columns.offset(&name, index)?)))
                        .collect::<GraphResult<_>>()?;
                    Ok((name, rows))
                })
                .collect()
        };
        let report = outcome.report();
        let nodes = remap(outcome.guard.nodes, &outcome.nodes)?;
        let edges = remap(outcome.guard.edges, &outcome.edges)?;
        let mut pool = self.pool.borrow_mut();
        pool.install_compacted_columns(true, outcome.nodes.columns);
        pool.install_compacted_columns(false, outcome.edges.columns);
        self.space.replace_attr_indices(true, nodes);
        self.space.replace_attr_indices(false, edges);
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_edge_mapping_cannot_install_already_compacted_node_columns() {
        let mut graph = Graph::new();
        graph.add_nodes(2).unwrap();
        graph.add_edge(0, 1).unwrap();
        for _ in 0..3 {
            graph
                .set_node_attr(0, "v".into(), AttrValue::Int(1))
                .unwrap();
            graph
                .set_edge_attr(0, "v".into(), AttrValue::Int(2))
                .unwrap();
        }
        let node_lengths = graph.pool.borrow().compaction_lengths(true);
        let edge_lengths = graph.pool.borrow().compaction_lengths(false);
        graph.space.set_edge_attr_index(0, "v".into(), usize::MAX);
        let outcome = graph
            .prepare_pool_compaction_with_reclamation(true)
            .unwrap()
            .execute()
            .unwrap();
        assert!(graph.finish_pool_compaction(outcome).is_err());
        assert_eq!(graph.pool.borrow().compaction_lengths(true), node_lengths);
        assert_eq!(graph.pool.borrow().compaction_lengths(false), edge_lengths);
        assert_eq!(
            graph.space.get_edge_attr_index(0, &"v".into()),
            Some(usize::MAX)
        );
    }
    #[test]
    fn malformed_unmapped_raw_values_reject_reclamation_without_changing_owner() {
        let mut graph = Graph::new();
        graph.add_node().unwrap();
        graph
            .pool
            .borrow_mut()
            .set_attr("bad".into(), AttrValue::Json("{".into()), true);
        let before = graph.pool.borrow().compaction_lengths(true);
        let plan = graph
            .prepare_pool_compaction_with_reclamation(true)
            .unwrap();
        let error = plan
            .execute()
            .err()
            .expect("corruption must reject")
            .to_string();
        assert!(
            error.contains("bad") && error.contains("value 0"),
            "{error}"
        );
        assert_eq!(graph.pool.borrow().compaction_lengths(true), before);
    }
}
