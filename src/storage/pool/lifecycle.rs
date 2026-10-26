//! Compact permanent purge ledgers. Allocation counters never move backwards.
use super::*;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub(crate) struct PurgedIds(pub(crate) Vec<(usize, usize)>);
impl PurgedIds {
    pub(crate) fn contains(&self, id: usize) -> bool {
        let index = self.0.partition_point(|&(start, _)| start <= id);
        index > 0 && id < self.0[index - 1].1
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub(crate) fn count(&self) -> usize {
        self.0.iter().map(|&(a, b)| b - a).sum()
    }
    pub(crate) fn validate(&self, bound: usize) -> GraphResult<()> {
        let mut previous = None;
        for &(start, end) in &self.0 {
            if start >= end || end > bound || previous.is_some_and(|p| start <= p) {
                return Err(GraphError::InvalidInput(
                    "Invalid permanent purge intervals".into(),
                ));
            }
            previous = Some(end);
        }
        Ok(())
    }
    pub(crate) fn complement(bound: usize, retained: &std::collections::HashSet<usize>) -> Self {
        let mut ids: Vec<_> = retained.iter().copied().filter(|&id| id < bound).collect();
        ids.sort_unstable();
        let mut ranges = Vec::new();
        let mut start = 0;
        for id in ids {
            if start < id {
                ranges.push((start, id));
            }
            start = id + 1;
        }
        if start < bound {
            ranges.push((start, bound));
        }
        Self(ranges)
    }
}
impl GraphPool {
    /// Whether an ID is reserved and has not been explicitly purged. This is
    /// independent of current GraphSpace liveness; sparse imported reservations count.
    pub fn is_node_allocated(&self, id: NodeId) -> bool {
        id < self.next_node_id && !self.purged_nodes.contains(id)
    }
    /// Whether an edge ID remains reserved, including unmaterialized imported gaps.
    pub fn is_edge_allocated(&self, id: EdgeId) -> bool {
        id < self.next_edge_id && !self.purged_edges.contains(id)
    }
    /// Number of explicitly purged node IDs within the lifetime allocation extent.
    pub fn purged_node_count(&self) -> usize {
        self.purged_nodes.count()
    }
    /// Number of explicitly purged edge IDs within the lifetime allocation extent.
    pub fn purged_edge_count(&self) -> usize {
        self.purged_edges.count()
    }
    pub(crate) fn archive_version(&self) -> u32 {
        if self.sparse_edges.is_some() {
            3
        } else if self.purged_nodes.is_empty() && self.purged_edges.is_empty() {
            1
        } else {
            2
        }
    }
}

pub(crate) struct ReclaimedStorage {
    pub(crate) nodes: Vec<(usize, usize)>,
    pub(crate) edges: Vec<(usize, usize)>,
    pub(crate) groups: Vec<SubgraphId>,
}
impl PurgedIds {
    fn difference(&self, old: &Self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        let mut prior = 0;
        for &(a, b) in &self.0 {
            let mut start = a;
            while prior < old.0.len() && old.0[prior].1 <= start {
                prior += 1;
            }
            let mut index = prior;
            while index < old.0.len() && old.0[index].0 < b {
                let (u, v) = old.0[index];
                if start < u {
                    result.push((start, u));
                }
                start = start.max(v);
                index += 1;
            }
            if start < b {
                result.push((start, b));
            }
        }
        result
    }
}
impl GraphPool {
    pub(crate) fn property_group_references(&self) -> std::collections::HashSet<SubgraphId> {
        self.node_attributes
            .values()
            .chain(self.edge_attributes.values())
            .flat_map(|c| c.as_slice())
            .filter_map(|v| {
                if let AttrValue::SubgraphRef(id) = v {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }
    /// Coordinate supplied owner-domain roots with legacy memberships. Frozen
    /// groups have independent ID domains; their own dependency Arcs remain intact.
    pub(crate) fn reclaim_storage(
        &mut self,
        mut nodes: std::collections::HashSet<NodeId>,
        edges: &mut std::collections::HashSet<EdgeId>,
        groups: &std::collections::HashSet<SubgraphId>,
    ) -> GraphResult<ReclaimedStorage> {
        for id in groups {
            self.validate_stored_subgraph(*id)?;
            if !self.retained_groups.contains_key(id) {
                let (members, connections, _) = &self.stored_subgraphs[id];
                nodes.extend(members);
                edges.extend(connections);
            }
        }
        for &edge in edges.iter() {
            if let Some((u, v)) = self.get_edge_endpoints(edge) {
                nodes.insert(u);
                nodes.insert(v);
            }
        }
        if nodes.iter().any(|&id| !self.is_node_allocated(id))
            || edges.iter().any(|&id| !self.is_edge_allocated(id))
        {
            return Err(GraphError::InvalidInput(
                "Retained root refers to an unallocated or purged entity".into(),
            ));
        }
        let next_nodes = PurgedIds::complement(self.next_node_id, &nodes);
        let next_edges = PurgedIds::complement(self.next_edge_id, edges);
        let mut removed_groups: Vec<_> = self
            .stored_subgraphs
            .keys()
            .filter(|id| !groups.contains(id))
            .copied()
            .collect();
        removed_groups.sort_unstable();
        let report = ReclaimedStorage {
            nodes: next_nodes.difference(&self.purged_nodes),
            edges: next_edges.difference(&self.purged_edges),
            groups: removed_groups,
        };
        if !next_edges.is_empty() || self.sparse_edges.is_some() {
            let retained = edges
                .iter()
                .filter_map(|&id| self.get_edge_endpoints(id).map(|pair| (id, pair)))
                .collect();
            self.sparse_edges = Some(retained);
            self.edge_alive = Vec::new();
            self.edge_sources = Vec::new();
            self.edge_targets = Vec::new();
        }
        self.purged_nodes = next_nodes;
        self.purged_edges = next_edges;
        self.stored_subgraphs.retain(|id, _| groups.contains(id));
        let retained = self
            .retained_groups
            .iter()
            .filter(|(id, _)| groups.contains(id))
            .map(|(&id, g)| (id, g.clone()))
            .collect();
        self.retained_groups = std::sync::Arc::new(retained);
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn purge_intervals_reject_malformed_and_resolve_boundaries_and_new_differences() {
        for ranges in [
            vec![(2, 2)],
            vec![(3, 2)],
            vec![(0, 2), (1, 3)],
            vec![(0, 2), (2, 3)],
            vec![(0, 11)],
        ] {
            assert!(PurgedIds(ranges).validate(10).is_err());
        }
        let old = PurgedIds(vec![(1, 3), (5, 6)]);
        old.validate(10).unwrap();
        for id in 0..10 {
            assert_eq!(old.contains(id), [1, 2, 5].contains(&id));
        }
        let new = PurgedIds::complement(10, &std::collections::HashSet::from([0, 7, 9]));
        assert_eq!(new.0, vec![(1, 7), (8, 9)]);
        assert_eq!(new.difference(&old), vec![(3, 5), (6, 7), (8, 9)]);
    }
    #[test]
    fn purge_reclaims_interior_endpoint_arrays_and_sparse_insertions_keep_numeric_order() {
        let mut pool = GraphPool::new();
        pool.ensure_node_id_exists(2).unwrap();
        pool.add_edges(&[(0, 1), (1, 2), (0, 2), (2, 2)]);
        let mut edges = std::collections::HashSet::from([1, 3]);
        pool.reclaim_storage(
            std::collections::HashSet::from([0, 1, 2]),
            &mut edges,
            &Default::default(),
        )
        .unwrap();
        assert_eq!(pool.sparse_edges.as_ref().unwrap().len(), 2);
        assert_eq!(pool.edge_sources.capacity(), 0);
        assert_eq!(pool.edge_targets.capacity(), 0);
        assert_eq!(pool.edge_alive.capacity(), 0);
        assert_eq!(pool.get_edge_endpoints(0), None);
        assert_eq!(pool.get_incident_edges(2).unwrap(), vec![1, 3]);
        assert!(pool.has_edge_between(2, 1));
        assert!(!pool.has_edge_between(0, 1));
        assert_eq!(pool.add_edges(&[(0, 2), (1, 2)]), vec![4, 5]);
        assert_eq!(pool.get_incident_edges(2).unwrap(), vec![1, 3, 4, 5]);
        assert!(pool.edge_sources.is_empty());
        let restored = pool
            .archive_document()
            .unwrap()
            .restore(pool.graph_type)
            .unwrap();
        assert_eq!(restored.get_edge_endpoints(5), Some((1, 2)));
        assert_eq!(restored.get_incident_edges(2).unwrap(), vec![1, 3, 4, 5]);
    }
}
