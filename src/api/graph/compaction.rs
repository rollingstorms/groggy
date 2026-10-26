//! Explicit exact-value compaction with guarded detached execution.
use super::*;
use crate::storage::pool::compaction::{self, Columns, CompactedColumns};

type Mappings = HashMap<AttrName, HashMap<usize, usize>>;

/// Raw property slot counts before and after explicit deduplication. These are
/// not byte/RSS counts; every distinct retained value remains available.
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
}
/// Owned raw-column capture for detached compaction. Capture copies values on
/// the owner thread. Finish rejects a stale or foreign owner without mutation.
pub struct PoolCompactionPlan {
    guard: Guard,
    nodes: Columns,
    edges: Columns,
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
}
impl PoolCompactionPlan {
    /// Deduplicate exact values outside the owner/runtime lock. No graph changes.
    pub fn execute(self) -> GraphResult<PoolCompactionOutcome> {
        Ok(PoolCompactionOutcome {
            guard: self.guard,
            nodes: compaction::compact(self.nodes)?,
            edges: compaction::compact(self.edges)?,
        })
    }
}
impl Graph {
    /// Remove duplicate raw property slots while preserving all distinct values,
    /// states, branches, inactive IDs, groups and identity reservations. Requires
    /// a clean head. Existing raw offsets are invalidated; read pins are independent.
    pub fn compact_pool(&mut self) -> GraphResult<PoolCompactionReport> {
        let outcome = self.prepare_pool_compaction()?.execute()?;
        self.finish_pool_compaction(outcome)
    }
    /// Capture retained columns for detached compaction. This explicit operation
    /// can temporarily duplicate the raw property payload in memory.
    pub fn prepare_pool_compaction(&self) -> GraphResult<PoolCompactionPlan> {
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
            },
            nodes: pool.compaction_columns(true),
            edges: pool.compaction_columns(false),
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
                    let offsets = columns.offsets.get(&name).ok_or_else(|| {
                        GraphError::InvalidInput("Missing compaction column".into())
                    })?;
                    let rows = rows
                        .into_iter()
                        .map(|(id, index)| {
                            Ok((
                                id,
                                *offsets.get(index).ok_or_else(|| {
                                    GraphError::InvalidInput(
                                        "Compaction source index out of bounds".into(),
                                    )
                                })?,
                            ))
                        })
                        .collect::<GraphResult<_>>()?;
                    Ok((name, rows))
                })
                .collect()
        };
        let nodes = remap(outcome.guard.nodes, &outcome.nodes)?;
        let edges = remap(outcome.guard.edges, &outcome.edges)?;
        let report = PoolCompactionReport {
            selected_state: outcome.guard.state,
            previous_node_values: outcome.nodes.previous,
            current_node_values: outcome.nodes.current,
            previous_edge_values: outcome.edges.previous,
            current_edge_values: outcome.edges.current,
        };
        let mut pool = self.pool.borrow_mut();
        pool.install_compacted_columns(true, outcome.nodes.columns);
        pool.install_compacted_columns(false, outcome.edges.columns);
        self.space.replace_attr_indices(true, nodes);
        self.space.replace_attr_indices(false, edges);
        Ok(report)
    }
}
