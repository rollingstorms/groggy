//! Explicit complete owner-domain reclamation after history reachability pruning.
use super::*;

/// A deterministic preview/receipt. Half-open ID ranges avoid enumerating sparse
/// allocation extents; counts describe new removals, not already purged IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PurgeReport {
    /// Selected accepted state; purge does not manufacture a mutation/version.
    pub selected_state: StateId,
    /// Newly purged node ID intervals [start,end).
    pub node_ranges: Vec<(NodeId, NodeId)>,
    /// Newly purged edge ID intervals [start,end).
    pub edge_ranges: Vec<(EdgeId, EdgeId)>,
    /// Newly purged node IDs, including orphan sparse reservations.
    pub purged_nodes: usize,
    /// Newly purged edge IDs, including orphan sparse reservations.
    pub purged_edges: usize,
    /// Removed owner group registrations/aliases; independent nested payloads survive.
    pub removed_groups: Vec<crate::types::SubgraphId>,
    /// Released names with no current or retained-history identity reference.
    pub released_external_ids: Vec<String>,
    /// Raw orphan node property slots reclaimed by the same maintenance operation.
    pub reclaimed_node_values: usize,
    /// Raw orphan edge property slots reclaimed by the same maintenance operation.
    pub reclaimed_edge_values: usize,
}
/// Owned portable capture suitable for detached preview/execution. Capture and
/// exact stale validation copy/encode all retained data; this is explicit maintenance.
pub struct PurgePlan {
    source: Vec<u8>,
    identity: Arc<()>,
}
/// Complete candidate, with no source mutations before successful installation.
pub struct PurgeOutcome {
    source: Vec<u8>,
    identity: Arc<()>,
    report: PurgeReport,
    pub(super) pool: GraphPool,
    pub(super) identities: ExternalIdentityIndex,
    pub(super) nodes: HashMap<AttrName, HashMap<usize, usize>>,
    pub(super) edges: HashMap<AttrName, HashMap<usize, usize>>,
}
impl PurgePlan {
    /// Check all retained states, collect orphan values/entities/group registrations
    /// and names, and verify the candidate. Frozen source domains are independent.
    pub fn execute(self) -> GraphResult<PurgeOutcome> {
        let mut graph = archive::GraphArchive::decode_document(&self.source)?.restore()?;
        graph.check_integrity()?;
        let compact = graph.compact_pool_with_reclamation(true)?;
        let mut nodes: HashSet<_> = graph.node_ids().into_iter().collect();
        let mut edges: HashSet<_> = graph.edge_ids().into_iter().collect();
        let mut groups = graph.pool.borrow().property_group_references();
        let mut names = HashMap::new();
        for is_node in [true, false] {
            let pool = graph.pool.borrow();
            for (id, name, index) in graph.space.active_attr_indices(is_node) {
                if name != EXTERNAL_ID {
                    continue;
                }
                let value = pool
                    .get_attr_by_index(name, index, is_node)
                    .and_then(AttrValue::get_text)
                    .ok_or_else(|| {
                        GraphError::InvalidInput("Invalid active external identity root".into())
                    })?;
                names.insert(
                    value,
                    if is_node {
                        ExternalEntityRef::Node(id)
                    } else {
                        ExternalEntityRef::Edge(id)
                    },
                );
            }
        }

        for version in graph.history_records() {
            let d = &version.delta;
            nodes.extend(&d.nodes_added);
            nodes.extend(&d.nodes_removed);
            edges.extend(&d.edges_removed);
            for &(id, u, v) in &d.edges_added {
                edges.insert(id);
                nodes.insert(u);
                nodes.insert(v);
            }
            for change in &d.edge_endpoint_changes {
                edges.insert(change.edge_id);
                nodes.extend([
                    change.old_endpoints.0,
                    change.old_endpoints.1,
                    change.new_endpoints.0,
                    change.new_endpoints.1,
                ]);
            }
            for (is_node, changes) in [(true, &d.node_attr_changes), (false, &d.edge_attr_changes)]
            {
                for (id, name, old, new) in changes {
                    if is_node {
                        nodes.insert(*id);
                    } else {
                        edges.insert(*id);
                    }
                    for value in old.iter().chain(new.iter()) {
                        if name == EXTERNAL_ID {
                            let external = value.get_text().ok_or_else(|| {
                                GraphError::InvalidInput(
                                    "Invalid retained external identity root".into(),
                                )
                            })?;
                            names.insert(
                                external,
                                if is_node {
                                    ExternalEntityRef::Node(*id)
                                } else {
                                    ExternalEntityRef::Edge(*id)
                                },
                            );
                        }
                        if let AttrValue::SubgraphRef(id) = value {
                            groups.insert(*id);
                        }
                    }
                }
            }
        }
        let removed = graph
            .pool
            .borrow_mut()
            .reclaim_storage(nodes, &mut edges, &groups)?;
        let mut released = Vec::new();
        let pool = graph.pool.borrow();
        graph.identities.bindings.retain(|name, entity| {
            let allocated = match entity {
                ExternalEntityRef::Node(id) => pool.is_node_allocated(*id),
                ExternalEntityRef::Edge(id) => pool.is_edge_allocated(*id),
            };
            let keep = allocated && names.get(name) == Some(entity);
            if !keep {
                released.push(name.clone());
            }
            keep
        });
        drop(pool);
        released.sort();
        graph.check_integrity()?;
        let report = PurgeReport {
            selected_state: graph.current_state_id(),
            purged_nodes: removed.nodes.iter().map(|&(a, b)| b - a).sum(),
            purged_edges: removed.edges.iter().map(|&(a, b)| b - a).sum(),
            node_ranges: removed.nodes,
            edge_ranges: removed.edges,
            removed_groups: removed.groups,
            released_external_ids: released,
            reclaimed_node_values: compact.reclaimed_unreferenced_node_values,
            reclaimed_edge_values: compact.reclaimed_unreferenced_edge_values,
        };
        let empty = GraphPool::new_with_type(graph.graph_type());
        let pool = std::mem::replace(&mut *graph.pool.borrow_mut(), empty);
        let nodes = graph.space.retained_attr_indices(true).clone();
        let edges = graph.space.retained_attr_indices(false).clone();
        Ok(PurgeOutcome {
            source: self.source,
            identity: self.identity,
            report,
            pool,
            identities: graph.identities,
            nodes,
            edges,
        })
    }
}
impl PurgeOutcome {
    /// Inspect the computed candidate before choosing to install it.
    pub fn report(&self) -> &PurgeReport {
        &self.report
    }
    pub(crate) fn rewrite_archive(&self, archive: &mut archive::GraphArchive) -> GraphResult<()> {
        archive.replace_purged_storage(self)
    }
}
impl Graph {
    /// Report the same complete candidate as purge_unreferenced, without mutation.
    pub fn preview_purge(&self) -> GraphResult<PurgeReport> {
        Ok(self.prepare_purge()?.execute()?.report)
    }
    /// Explicitly purge data with no retained-history/current owner-domain root.
    /// No branch/history is implicitly pruned, no internal IDs are reused, and
    /// independent open graph/table/group/owned snapshots remain readable.
    pub fn purge_unreferenced(&mut self) -> GraphResult<PurgeReport> {
        let outcome = self.prepare_purge()?.execute()?;
        self.finish_purge(outcome)
    }
    /// Capture all retained input for detached execution or preview.
    pub fn prepare_purge(&self) -> GraphResult<PurgePlan> {
        Ok(PurgePlan {
            source: self.archive_document()?.encode_document()?,
            identity: Arc::clone(self.history.identity()),
        })
    }
    /// Install only if the complete portable source and owner identity still match.
    pub fn finish_purge(&mut self, outcome: PurgeOutcome) -> GraphResult<PurgeReport> {
        self.validate_write_head(outcome.report.selected_state)?;
        if !Arc::ptr_eq(self.history.identity(), &outcome.identity)
            || self.archive_document()?.encode_document()? != outcome.source
        {
            return Err(GraphError::InvalidInput(
                "Stale or foreign purge result".into(),
            ));
        }
        *self.pool.borrow_mut() = outcome.pool;
        self.identities = outcome.identities;
        self.space.replace_attr_indices(true, outcome.nodes);
        self.space.replace_attr_indices(false, outcome.edges);
        Ok(outcome.report)
    }
    /// Reserved owner ID, independent of active membership, until explicit purge.
    pub fn is_node_allocated(&self, id: NodeId) -> bool {
        self.pool.borrow().is_node_allocated(id)
    }
    /// Reserved owner edge ID, independent of active membership, until explicit purge.
    pub fn is_edge_allocated(&self, id: EdgeId) -> bool {
        self.pool.borrow().is_edge_allocated(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn huge_sparse_allocation_extents_use_constant_size_purge_intervals() {
        let mut graph = Graph::new();
        graph
            .pool
            .borrow_mut()
            .preserve_allocation(1_000_000_000_000, 2_000_000_000_000);
        let report = graph.purge_unreferenced().unwrap();
        assert_eq!(report.node_ranges, vec![(0, 1_000_000_000_000)]);
        assert_eq!(report.edge_ranges, vec![(0, 2_000_000_000_000)]);
        assert_eq!(graph.pool.borrow().purged_node_count(), 1_000_000_000_000);
        assert_eq!(graph.add_node().unwrap(), 1_000_000_000_000);
        assert!(!graph.is_node_allocated(999_999_999_999));
        assert_eq!(graph.add_node().unwrap(), 1_000_000_000_001);
        assert_eq!(
            graph
                .add_edge(1_000_000_000_000, 1_000_000_000_001)
                .unwrap(),
            2_000_000_000_000
        );

        let copy = graph.prepare_archive().unwrap().restore().unwrap();
        assert!(!copy.is_edge_allocated(1_999_999_999_999));
        assert_eq!(
            copy.pool().get_edge_endpoints(2_000_000_000_000),
            Some((1_000_000_000_000, 1_000_000_000_001))
        );
        copy.check_integrity().unwrap();
    }
    #[test]
    fn legacy_membership_roots_retain_owner_nodes_edges_and_endpoint_dependencies() {
        let mut graph = Graph::new();
        graph.add_node().unwrap();
        graph.create_branch("side".into()).unwrap();
        graph.checkout_branch("side".into()).unwrap();
        graph.add_nodes(2).unwrap();
        let edge = graph.add_edge(1, 2).unwrap();
        let group = graph
            .pool
            .borrow_mut()
            .store_subgraph(
                HashSet::from([1, 2]),
                HashSet::from([edge]),
                "legacy".into(),
            )
            .unwrap();
        graph.checkout_branch("main".into()).unwrap();
        graph.create_meta_node(group).unwrap();
        graph.prune_branch("side").unwrap();
        assert_eq!(graph.purge_unreferenced().unwrap().purged_edges, 0);
        assert!(graph.is_node_allocated(1));
        assert!(graph.is_node_allocated(2));
        assert!(graph.is_edge_allocated(edge));
        assert!(graph.pool.borrow().get_subgraph(group).is_ok());
        graph.check_integrity().unwrap();
    }
    #[test]
    fn raw_restoration_helpers_reject_purged_ids_before_mutating_storage() {
        let mut graph = Graph::new();
        graph.pool.borrow_mut().preserve_allocation(3, 4);
        graph.purge_unreferenced().unwrap();
        let mut pool = graph.pool.borrow().clone();
        assert!(pool.ensure_node_id_exists(1).is_err());
        assert!(pool.add_edge_with_id(1, 0, 2).is_err());
        assert_eq!(pool.statistics().node_count, 3);
        assert_eq!(pool.statistics().edge_count, 4);
    }
}
