//! Detached verification of an eligible current topology cache; no rebuilding.
use super::*;
use crate::types::GraphType;

/// Immutable cached arrays plus authoritative membership/topology at capture.
/// No GraphSpace, pool borrow or mutable graph crosses the owner boundary.
pub(crate) struct CachedTopologyCheck {
    edge_ids: Arc<Vec<EdgeId>>,
    sources: Arc<Vec<NodeId>>,
    targets: Arc<Vec<NodeId>>,
    node_ids: Arc<Vec<NodeId>>,
    offsets: Arc<Vec<usize>>,
    neighbor_nodes: Arc<Vec<NodeId>>,
    neighbor_edges: Arc<Vec<EdgeId>>,
    neighbors: Arc<HashMap<NodeId, Vec<(NodeId, EdgeId)>>>,
    expected_nodes: Vec<NodeId>,
    expected_edges: Vec<(EdgeId, NodeId, NodeId)>,
    graph_type: GraphType,
}
fn invalid(message: impl std::fmt::Display) -> GraphError {
    GraphError::InvalidInput(format!("Current topology cache: {message}"))
}
impl GraphSpace {
    /// Capture only a cache eligible for serving this current topology version.
    /// Cold/stale caches are allowed and remain untouched. Poison is an error.
    pub(crate) fn prepare_cached_topology_check(
        &self,
        pool: &GraphPool,
    ) -> GraphResult<Option<CachedTopologyCheck>> {
        let cache = self.cache.read().map_err(|_| {
            invalid("poisoned cache lock; reconstruct an owned snapshot or reopen the owner")
        })?;
        if cache.built_version != self.version {
            return Ok(None);
        }
        let snapshot = cache
            .snapshot
            .as_ref()
            .ok_or_else(|| invalid("current version has no snapshot"))?;
        if snapshot.version != self.version {
            return Err(invalid(
                "snapshot version disagrees with the current cache version",
            ));
        }
        let expected_nodes: Vec<_> = self.active_nodes.iter().copied().collect();
        let mut expected_edges = Vec::with_capacity(self.active_edges.len());
        for &edge in &self.active_edges {
            let (u, v) = pool
                .get_edge_endpoints(edge)
                .ok_or_else(|| invalid(format!("active edge {edge} has no retained endpoints")))?;
            if !self.active_nodes.contains(&u) || !self.active_nodes.contains(&v) {
                return Err(invalid(format!(
                    "active edge {edge} has inactive endpoints"
                )));
            }
            expected_edges.push((edge, u, v));
        }
        Ok(Some(CachedTopologyCheck {
            edge_ids: Arc::clone(&snapshot.edge_ids),
            sources: Arc::clone(&snapshot.sources),
            targets: Arc::clone(&snapshot.targets),
            node_ids: Arc::clone(&snapshot.node_ids),
            offsets: Arc::clone(&snapshot.neighbor_offsets),
            neighbor_nodes: Arc::clone(&snapshot.neighbor_nodes),
            neighbor_edges: Arc::clone(&snapshot.neighbor_edges),
            neighbors: Arc::clone(&snapshot.neighbors),
            expected_nodes,
            expected_edges,
            graph_type: pool.graph_type(),
        }))
    }
}
impl CachedTopologyCheck {
    pub(crate) fn verify(mut self) -> GraphResult<()> {
        self.expected_nodes.sort_unstable();
        self.expected_edges.sort_unstable();
        if self.edge_ids.len() != self.sources.len() || self.edge_ids.len() != self.targets.len() {
            return Err(invalid("edge/source/target column lengths differ"));
        }
        let mut triples: Vec<_> = self
            .edge_ids
            .iter()
            .zip(self.sources.iter())
            .zip(self.targets.iter())
            .map(|((&e, &u), &v)| (e, u, v))
            .collect();
        triples.sort_unstable();
        if triples != self.expected_edges {
            return Err(invalid(
                "cached edge IDs/endpoints disagree with active retained topology",
            ));
        }
        if *self.node_ids != self.expected_nodes {
            return Err(invalid(
                "CSR node IDs disagree with sorted active membership (including isolates)",
            ));
        }
        let length = self
            .node_ids
            .len()
            .checked_add(1)
            .ok_or_else(|| invalid("CSR node extent overflow"))?;
        if self.offsets.len() != length
            || self.offsets.first() != Some(&0)
            || self.offsets.last() != Some(&self.neighbor_nodes.len())
            || self.neighbor_nodes.len() != self.neighbor_edges.len()
            || self.offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(invalid("malformed CSR offsets or neighbor column lengths"));
        }
        let mut expected =
            incidence_neighbors(self.expected_edges.iter().copied(), self.graph_type);
        for neighbors in expected.values_mut() {
            neighbors.sort_unstable();
        }
        if self.neighbors.len() != expected.len() {
            return Err(invalid(
                "neighbor-map membership differs from active topology",
            ));
        }
        for (&node, want) in &expected {
            let mut got = self
                .neighbors
                .get(&node)
                .ok_or_else(|| invalid(format!("missing neighbor-map row for node {node}")))?
                .clone();
            got.sort_unstable();
            if &got != want {
                return Err(invalid(format!("neighbor-map row for node {node} has wrong endpoints, edge IDs or multiplicity")));
            }
        }
        for (row, &node) in self.node_ids.iter().enumerate() {
            let start = self.offsets[row];
            let end = self.offsets[row + 1];
            // Monotone offsets starting at zero and ending at the column length
            // were validated above, so all row slices are in range.
            let mut got: Vec<_> = self.neighbor_nodes[start..end]
                .iter()
                .copied()
                .zip(self.neighbor_edges[start..end].iter().copied())
                .collect();
            got.sort_unstable();
            let want = expected.get(&node).map(Vec::as_slice).unwrap_or(&[]);
            if got != want {
                return Err(invalid(format!(
                    "CSR row for node {node} has wrong neighbors, edge IDs or multiplicity"
                )));
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, GraphConfig};
    fn graph(kind: GraphType) -> Graph {
        let mut graph = Graph::with_config(GraphConfig {
            graph_type: kind,
            ..Default::default()
        })
        .unwrap();
        graph.add_nodes(3).unwrap();
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 1).unwrap();
        graph.neighbors(0).unwrap();
        graph
    }
    #[test]
    fn current_cache_verifies_directed_undirected_parallel_loops_and_isolates_without_rebuilding() {
        let empty = Graph::new();
        assert!(empty.edge_sources().is_empty());
        empty.check_integrity().unwrap();
        for kind in [GraphType::Directed, GraphType::Undirected] {
            let mut g = graph(kind);
            let version = g.space().cache.read().unwrap().built_version;
            g.check_integrity().unwrap();
            assert_eq!(g.space().cache.read().unwrap().built_version, version);
            g.set_node_attr(0, "value".into(), crate::AttrValue::Int(7))
                .unwrap();
            g.check_integrity().unwrap();
            g.set_edge_endpoints(0, 2, 1).unwrap();
            g.neighbors(2).unwrap();
            g.check_integrity().unwrap();
        }
    }
    #[test]
    fn corrupted_current_cache_shapes_contents_and_version_are_rejected_explicitly() {
        for variant in 0..7 {
            let g = graph(GraphType::Undirected);
            {
                let mut cache = g.space().cache.write().unwrap();
                let snapshot = cache.snapshot.as_mut().unwrap();
                match variant {
                    0 => snapshot.sources = Arc::new(vec![]),
                    1 => snapshot.node_ids = Arc::new(vec![0, 1]),
                    2 => {
                        snapshot.neighbor_offsets =
                            Arc::new(vec![0, usize::MAX, 0, snapshot.neighbor_nodes.len()])
                    }
                    3 => {
                        snapshot.neighbor_edges = Arc::new(vec![999; snapshot.neighbor_nodes.len()])
                    }
                    4 => snapshot.neighbors = Arc::new(HashMap::new()),
                    5 => snapshot.targets = Arc::new(vec![2; snapshot.edge_ids.len()]),
                    6 => snapshot.version += 1,
                    _ => unreachable!(),
                }
            }
            let error = g.check_integrity().unwrap_err();
            assert!(
                error.to_string().contains("Current topology cache"),
                "variant {variant}: {error}"
            );
        }
        let g = graph(GraphType::Directed);
        g.space().cache.write().unwrap().snapshot = None;
        assert!(g
            .check_integrity()
            .unwrap_err()
            .to_string()
            .contains("current version has no snapshot"));
    }
    #[test]
    fn detached_checks_keep_captured_arrays_and_cold_stale_caches_remain_untouched() {
        let mut g = Graph::new();
        g.add_nodes(2).unwrap();
        assert!(g.space().cache.read().unwrap().snapshot.is_none());
        g.check_integrity().unwrap();
        assert!(g.space().cache.read().unwrap().snapshot.is_none());
        g.add_edge(0, 1).unwrap();
        g.neighbors(0).unwrap();
        let captured = g.prepare_integrity_check().unwrap();
        g.space()
            .cache
            .write()
            .unwrap()
            .snapshot
            .as_mut()
            .unwrap()
            .neighbor_edges = Arc::new(vec![999, 999]);
        captured.execute().unwrap();
        assert!(g.check_integrity().is_err());
        // Topology mutation invalidates the old cache; diagnostic capture must not rebuild it.
        g.add_node().unwrap();
        let built = g.space().cache.read().unwrap().built_version;
        g.check_integrity().unwrap();
        assert_eq!(g.space().cache.read().unwrap().built_version, built);
    }
    #[test]
    fn poisoned_cache_lock_reports_an_error_without_panicking_or_repairing() {
        let g = graph(GraphType::Directed);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = g.space().cache.write().unwrap();
            panic!("injected cache poison");
        }));
        assert!(g
            .check_integrity()
            .unwrap_err()
            .to_string()
            .contains("poisoned cache lock"));
        assert!(g.space().cache.read().is_err());
    }
}
