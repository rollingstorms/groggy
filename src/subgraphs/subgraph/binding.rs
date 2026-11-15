//! Optional immutable read pin paired with the existing explicit write owner.
use super::*;
impl Subgraph {
    /// Validate the selected read source, without confusing retained/provisional
    /// entities in a frozen binding with current owner liveness.
    pub(crate) fn validate_selected_read(&self) -> GraphResult<()> {
        if self.frozen.is_none() {
            self.validate_query_scope(&self.graph.borrow())?;
        }
        Ok(())
    }

    /// Direction belongs to the selected source when reads are immutable.
    pub(crate) fn read_is_directed(&self) -> bool {
        self.frozen.as_ref().map_or_else(
            || self.graph.borrow().is_directed(),
            crate::GraphView::is_directed,
        )
    }

    /// Bind current selected reads to one immutable source without changing the owner.
    /// Default-facing adapters use this; low-level execution scopes remain explicit.
    pub fn freeze(mut self) -> GraphResult<Self> {
        if self.frozen.is_none() {
            let root = self.graph.borrow().view()?;
            self.frozen = Some(crate::GraphView::from_shared_membership(
                root.read_state().clone(),
                self.nodes.clone(),
                self.edges.clone(),
            )?);
        }
        Ok(self)
    }
    /// Frozen source for bound reads, or None for an explicit current execution scope.
    pub fn frozen_view(&self) -> Option<&crate::GraphView> {
        self.frozen.as_ref()
    }
    /// Attach an explicit write owner to an immutable read without recapturing it.
    pub fn from_frozen_view(
        graph: Arc<OwnerCell<Graph>>,
        view: crate::GraphView,
        name: String,
    ) -> Self {
        let (nodes, edges) = view.shared_membership();
        Self {
            graph,
            frozen: Some(view),
            nodes,
            edges,
            subgraph_type: name,
            subgraph_id: Self::generate_simple_id(),
            component_cache: Arc::new(OwnerCell::new(ComponentCacheStore::default())),
            ordered_nodes_cache: Arc::new(OnceLock::new()),
            ordered_edges_cache: Arc::new(OnceLock::new()),
            topology_cache: OwnerCell::new(HashMap::new()),
            validated_scope: OwnerCell::new(None),
        }
    }
    pub(crate) fn bound_selection(&self, view: crate::GraphView, name: String) -> Self {
        Self::from_frozen_view(self.graph.clone(), view, name)
    }
    /// Resolve an explicit execution scope only when its source still matches the owner.
    /// Readonly/transient execution permits a matching unpublished source; writes
    /// require the accepted-state contract shared with apply_view_batch.
    pub fn execution_scope(&self, require_accepted: bool) -> GraphResult<Self> {
        if let Some(view) = &self.frozen {
            self.graph
                .borrow()
                .validate_bound_source(view, require_accepted)?;
        }
        let mut result = self.clone();
        result.frozen = None;
        // Source validation proves these version-keyed execution caches remain
        // eligible for reuse. Owner topology revisions still invalidate them.
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selected() -> (Arc<OwnerCell<Graph>>, Subgraph, NodeId, NodeId, EdgeId) {
        let owner = Arc::new(OwnerCell::new(Graph::new()));
        let a = owner.borrow_mut().add_node().unwrap();
        let b = owner.borrow_mut().add_node().unwrap();
        let edge = owner.borrow_mut().add_edge(a, b).unwrap();
        owner
            .borrow_mut()
            .set_node_attr(a, "value".into(), AttrValue::Int(7))
            .unwrap();
        let scope = Subgraph::new(
            owner.clone(),
            HashSet::from([a, b]),
            HashSet::from([edge]),
            "bound".into(),
        )
        .freeze()
        .unwrap();
        (owner, scope, a, b, edge)
    }

    #[test]
    fn bound_reads_and_derivations_survive_owner_deletion() {
        let (owner, scope, a, b, edge) = selected();
        owner.borrow_mut().remove_node(b).unwrap();
        owner
            .borrow_mut()
            .set_node_attr(a, "value".into(), AttrValue::Int(99))
            .unwrap();
        assert_eq!(
            scope.get_node_attribute(a, &"value".into()).unwrap(),
            Some(AttrValue::Int(7))
        );
        {
            let _writer = owner.borrow_mut();
            assert_eq!(
                scope
                    .aggregate_attribute_with_defaults("value", "sum", &HashMap::new())
                    .unwrap(),
                AttrValue::Int(7)
            );
        }
        assert_eq!(scope.edge_endpoints(edge).unwrap(), (a, b));
        assert_eq!(scope.neighbors(a).unwrap(), vec![b]);
        assert_eq!(
            scope
                .degree_read(crate::subgraphs::DegreeMode::Incident, false)
                .unwrap()
                .counts(&[a, b, a])
                .unwrap(),
            vec![1, 1, 1]
        );
        let child = scope.bfs_selection(a, None).unwrap();
        assert!(child.has_node(b));
        assert_eq!(
            child.get_node_attribute(a, &"value".into()).unwrap(),
            Some(AttrValue::Int(7))
        );
        assert_eq!(scope.component_selections().unwrap().len(), 1);
        assert_eq!(scope.sample(2, Some(2)).unwrap().edge_count(), 1);
        assert_eq!(scope.to_networkx().unwrap().edges.len(), 1);
        assert!(scope.adjacency_matrix(true, None).is_ok());
        assert!(scope.execution_scope(true).is_err());
        assert!(scope
            .set_node_attribute_bulk(&"value".into(), AttrValue::Int(11))
            .is_err());
        assert_eq!(
            owner.borrow().get_node_attr(a, &"value".into()).unwrap(),
            Some(AttrValue::Int(99))
        );
    }

    #[test]
    fn explicit_write_keeps_original_bound_values_and_rejects_reuse() {
        let (owner, scope, a, _, _) = selected();
        assert!(scope.execution_scope(true).is_ok());
        scope
            .set_node_attribute_bulk(&"value".into(), AttrValue::Int(11))
            .unwrap();
        assert_eq!(
            scope.get_node_attribute(a, &"value".into()).unwrap(),
            Some(AttrValue::Int(7))
        );
        assert_eq!(
            owner.borrow().get_node_attr(a, &"value".into()).unwrap(),
            Some(AttrValue::Int(11))
        );
        assert!(scope
            .set_node_attribute_bulk(&"value".into(), AttrValue::Int(12))
            .is_err());
    }
}

#[cfg(test)]
mod specialized_tests {
    use super::*;
    use crate::subgraphs::{ComponentSubgraph, FilteredSubgraph, NeighborhoodSubgraph};
    use crate::traits::{
        ComponentOperations, FilterCriteria, FilterOperations, NeighborhoodOperations,
    };

    #[test]
    fn specialized_frozen_derivations_keep_sources_and_metadata() {
        let owner = Arc::new(OwnerCell::new(Graph::new()));
        for _ in 0..3 {
            owner.borrow_mut().add_node().unwrap();
        }
        owner.borrow_mut().add_edge(0, 1).unwrap();
        owner.borrow_mut().add_edge(1, 2).unwrap();
        owner
            .borrow_mut()
            .set_node_attr(0, "v".into(), AttrValue::Int(7))
            .unwrap();
        let criteria = FilterCriteria::NodeAttributeEquals {
            name: "v".into(),
            value: AttrValue::Int(7),
        };
        let filtered = FilteredSubgraph::from_criteria(owner.clone(), criteria.clone())
            .unwrap()
            .freeze()
            .unwrap();
        let other = FilteredSubgraph::from_criteria(owner.clone(), criteria)
            .unwrap()
            .freeze()
            .unwrap();
        let neighborhood = NeighborhoodSubgraph::new(
            owner.clone(),
            vec![0],
            0,
            HashSet::from([0]),
            HashSet::new(),
        )
        .freeze()
        .unwrap();
        let component = ComponentSubgraph::new(
            owner.clone(),
            HashSet::from([0, 1, 2]),
            HashSet::from([0, 1]),
            4,
            true,
            5,
        )
        .freeze()
        .unwrap();
        owner
            .borrow_mut()
            .set_node_attr(0, "v".into(), AttrValue::Int(9))
            .unwrap();
        owner.borrow_mut().remove_node(1).unwrap();
        assert!(filtered.matches_node_filter(0).unwrap());
        assert_eq!(filtered.filter_stats().total_nodes, 3);
        assert_eq!(
            filtered
                .and_filter(&other)
                .unwrap()
                .get_node_attribute(0, &"v".into())
                .unwrap(),
            Some(AttrValue::Int(7))
        );
        assert_eq!(filtered.or_filter(&other).unwrap().node_count(), 1);
        let complement = filtered.not_filter().unwrap();
        assert!(complement.contains_node(1));
        assert!(complement.contains_node(2));
        assert_eq!(complement.edge_count(), 1);
        let narrowed = filtered
            .add_criteria(FilterCriteria::NodeDegreeRange { min: 1, max: 1 })
            .unwrap();
        assert_eq!(narrowed.node_count(), 1);
        let refreshed = filtered.reapply_filter().unwrap();
        assert_eq!(refreshed.node_count(), 0);
        assert!(filtered.matches_node_filter(0).unwrap());
        let expanded = neighborhood.expand_by(2).unwrap();
        assert_eq!(expanded.node_count(), 3);
        assert_eq!(expanded.hops(), 2);
        assert_eq!(expanded.central_nodes(), &[0]);
        assert_eq!(expanded.nodes_at_hop(0).unwrap(), vec![0]);
        assert_eq!(expanded.nodes_at_hop(1).unwrap(), vec![1]);
        assert_eq!(expanded.nodes_at_hop(2).unwrap(), vec![2]);
        assert!(neighborhood
            .expand_by(usize::MAX)
            .unwrap()
            .expand_by(1)
            .is_err());
        assert_eq!(component.component_id(), 4);
        assert_eq!(component.total_components(), 5);
        assert_eq!(component.neighbors(0).unwrap(), vec![1]);
        assert!(component
            .induced_subgraph(&[0, 1])
            .unwrap()
            .frozen_view()
            .is_some());
    }
}
