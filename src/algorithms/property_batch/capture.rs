//! Selected pending read materialization from an already checked immutable base.
//! Live-owner lookup/coordinate validation belongs to capture_scoped; this kernel
//! cannot borrow or mutate a Graph. It preserves pending capture authority and
//! accepted-history separation without creating a writable scratch owner.
use super::*;

impl PropertyBatch {
    /// The adapter must supply this batch's checked accepted root. Membership is
    /// explicit so fixed children and dynamic roots reuse the identical kernel.
    pub(crate) fn capture_membership(
        &mut self,
        base: &crate::GraphView,
        selected_nodes: &HashSet<NodeId>,
        selected_edges: &HashSet<EdgeId>,
        fixed: bool,
    ) -> GraphResult<crate::GraphView> {
        let topology = self.pending_topology(base)?;
        let (nodes, edges) = if fixed {
            for id in selected_nodes {
                if !topology.nodes.contains(id) {
                    return Err(crate::GraphError::InvalidInput(format!(
                        "Execution frame node {id} is outside the pending selection"
                    )));
                }
            }
            for id in selected_edges {
                let &(source, target) = topology.edges.get(id).ok_or_else(|| {
                    crate::GraphError::InvalidInput(format!(
                        "Execution frame edge {id} is outside the pending selection"
                    ))
                })?;
                if !selected_nodes.contains(&source) || !selected_nodes.contains(&target) {
                    return Err(crate::GraphError::InvalidInput(format!(
                        "Execution frame edge {id} endpoint is outside its node selection"
                    )));
                }
            }
            (selected_nodes.clone(), selected_edges.clone())
        } else {
            let (added_nodes, added_edges) = self.added_ids();
            if selected_nodes
                .iter()
                .any(|id| !base.nodes().contains(id) && !self.is_added_node(*id))
                || selected_edges
                    .iter()
                    .any(|id| !base.edges().contains(id) && !self.is_added_edge(*id))
            {
                return Err(crate::GraphError::InvalidInput(
                    "Execution selection expands outside its original batch root".into(),
                ));
            }
            let nodes: HashSet<_> = selected_nodes
                .iter()
                .copied()
                .chain(added_nodes)
                .filter(|id| topology.nodes.contains(id))
                .collect();
            let edges = selected_edges
                .iter()
                .copied()
                .chain(added_edges)
                .filter(|id| {
                    topology.edges.get(id).is_some_and(|(source, target)| {
                        nodes.contains(source) && nodes.contains(target)
                    })
                })
                .collect();
            (nodes, edges)
        };
        let original = base.read_state().snapshot();
        let mut snapshot = crate::state::state::GraphSnapshot::empty_with_type(
            original.state_id,
            original.graph_type,
        );
        snapshot.active_nodes = nodes.into_iter().collect();
        snapshot.active_nodes.sort_unstable();
        for &id in &snapshot.active_nodes {
            if let Some(attrs) = original.node_attributes.get(&id) {
                snapshot
                    .node_attributes
                    .insert(id, copy_unstaged_attributes(id, attrs, &self.nodes));
            }
            if self.is_added_node(id) {
                snapshot
                    .node_attributes
                    .entry(id)
                    .or_default()
                    .insert("entity_type".into(), AttrValue::Text("base".into()));
            }
        }
        for id in edges {
            let endpoints = topology.edges.get(&id).ok_or_else(|| {
                crate::GraphError::InvalidInput("Missing prepared frame endpoints".into())
            })?;
            snapshot.edges.insert(id, *endpoints);
            if let Some(attrs) = original.edge_attributes.get(&id) {
                snapshot
                    .edge_attributes
                    .insert(id, copy_unstaged_attributes(id, attrs, &self.edges));
            }
        }
        // Column-first staged storage remains authoritative. Read only selected
        // rows, rather than cloning every sibling's accumulated values.
        for (name, column) in &self.nodes {
            for &id in &snapshot.active_nodes {
                if let Some(value) = column.get(&id) {
                    snapshot
                        .node_attributes
                        .entry(id)
                        .or_default()
                        .insert(name.clone(), value.clone());
                }
            }
        }
        for (name, column) in &self.edges {
            for &id in snapshot.edges.keys() {
                if let Some(value) = column.get(&id) {
                    snapshot
                        .edge_attributes
                        .entry(id)
                        .or_default()
                        .insert(name.clone(), value.clone());
                }
            }
        }
        let state = crate::state::read_state::PinnedReadState::from_snapshot(snapshot)?
            .with_groups(base.read_state().groups().clone());
        self.register_capture(&state);
        Ok(crate::GraphView::all(state))
    }

    // Structural plans append only. Their lengths identify geometry revisions;
    // property-only assignments do not invalidate prepared membership/endpoints.
    fn topology_stamp(&self) -> TopologyStamp {
        (
            self.add_nodes,
            self.add_edges.len(),
            self.remove_nodes.len(),
            self.remove_edges.len(),
            self.edge_endpoints.len(),
        )
    }
    fn pending_topology(&mut self, base: &crate::GraphView) -> GraphResult<Arc<PendingTopology>> {
        let stamp = self.topology_stamp();
        let cached = self
            .root
            .is_some()
            .then(|| {
                self.topology_cache
                    .as_ref()
                    .filter(|(old, _)| *old == stamp)
                    .map(|(_, topology)| topology.clone())
            })
            .flatten();
        let topology = if let Some(topology) = cached {
            topology
        } else {
            let mut revisions = HashMap::with_capacity(self.edge_endpoints.len());
            for &(edge, source, target) in &self.edge_endpoints {
                let (BatchNodeRef::Existing(source), BatchNodeRef::Existing(target)) =
                    (source, target)
                else {
                    return Err(crate::GraphError::InvalidInput(
                        "Unsupported staged added endpoint".into(),
                    ));
                };
                if revisions.insert(edge, (source, target)).is_some() {
                    return Err(crate::GraphError::InvalidInput(format!(
                        "Duplicate staged endpoint revision for edge {edge}"
                    )));
                }
            }
            let mut removed_nodes = HashSet::with_capacity(self.remove_nodes.len());
            for reference in &self.remove_nodes {
                let BatchNodeRef::Existing(id) = reference else {
                    return Err(crate::GraphError::InvalidInput(
                        "Unsupported staged added node removal".into(),
                    ));
                };
                removed_nodes.insert(*id);
            }
            let mut removed_edges = HashSet::with_capacity(self.remove_edges.len());
            for reference in &self.remove_edges {
                let BatchEdgeRef::Existing(id) = reference else {
                    return Err(crate::GraphError::InvalidInput(
                        "Unsupported staged added edge removal".into(),
                    ));
                };
                removed_edges.insert(*id);
            }
            let (added_nodes, added_edges) = self.added_ids();
            let nodes: HashSet<_> = base
                .nodes()
                .iter()
                .copied()
                .chain(added_nodes.iter().copied())
                .filter(|id| !removed_nodes.contains(id))
                .collect();
            let mut edges = HashMap::new();
            for (&id, &(source, target)) in added_edges.iter().zip(&self.add_edges) {
                let resolve = |reference| match reference {
                    BatchNodeRef::Existing(id) => id,
                    BatchNodeRef::Added(position) => added_nodes[position],
                };
                let (source, target) = (resolve(source), resolve(target));
                if !removed_edges.contains(&id)
                    && nodes.contains(&source)
                    && nodes.contains(&target)
                {
                    edges.insert(id, (source, target));
                }
            }
            for &id in base.edges() {
                let (source, target) = revisions
                    .get(&id)
                    .copied()
                    .map(Ok)
                    .unwrap_or_else(|| base.edge_endpoints(id))?;
                if !removed_edges.contains(&id)
                    && nodes.contains(&source)
                    && nodes.contains(&target)
                {
                    edges.insert(id, (source, target));
                }
            }
            Arc::new(PendingTopology { nodes, edges })
        };
        let validate = |node, id| {
            if if node {
                topology.nodes.contains(&id)
            } else {
                topology.edges.contains_key(&id)
            } {
                Ok(())
            } else {
                Err(crate::GraphError::InvalidInput(format!(
                    "Staged {} property target {id} is outside the resulting selection",
                    if node { "node" } else { "edge" }
                )))
            }
        };
        if self.root.is_some() && self.validated_topology == Some(stamp) {
            for &(node, id) in &self.dirty_targets {
                validate(node, id)?;
            }
        } else {
            for column in self.nodes.values() {
                for &id in column.keys() {
                    validate(true, id)?;
                }
            }
            for column in self.edges.values() {
                for &id in column.keys() {
                    validate(false, id)?;
                }
            }
        }
        if self.root.is_some() {
            self.topology_cache = Some((stamp, topology.clone()));
            self.validated_topology = Some(stamp);
            self.dirty_targets.clear();
        }
        Ok(topology)
    }

    /// Membership derivatives retain the same issued capture identity.
    pub(crate) fn owns_capture(&self, source: &crate::GraphView) -> bool {
        self.captures
            .get(&source.read_state().capture_key())
            .and_then(|(_, capture)| capture.upgrade())
            .is_some_and(|issued| issued.shares_capture(source.read_state()))
    }

    fn register_capture(&mut self, state: &crate::state::read_state::PinnedReadState) {
        if self.captures.len() >= self.capture_sweep_at {
            self.captures
                .retain(|_, (_, capture)| capture.upgrade().is_some());
            let retained = self.captures.len().saturating_mul(2).max(64);
            if self.captures.capacity() > retained.saturating_mul(2) {
                self.captures.shrink_to(retained);
            }
            self.capture_sweep_at = retained;
        }
        self.captures.insert(
            state.capture_key(),
            (self.read_revision, state.downgrade_capture()),
        );
    }
}

/// Copy accepted values only when the pending column has no replacement for
/// this row. The overlay below copies the final assignment exactly once, including
/// explicit Null; unrelated rows of a staged column retain their accepted values.
fn copy_unstaged_attributes(
    id: usize,
    attrs: &HashMap<AttrName, AttrValue>,
    staged: &BTreeMap<AttrName, BTreeMap<usize, AttrValue>>,
) -> HashMap<AttrName, AttrValue> {
    if staged.is_empty() {
        return attrs.clone();
    }
    let mut result = HashMap::with_capacity(attrs.len());
    for (name, value) in attrs {
        if !staged
            .get(name)
            .is_some_and(|column| column.contains_key(&id))
        {
            result.insert(name.clone(), value.clone());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Graph {
        let mut graph = Graph::new_directed();
        graph
            .apply_batch(
                GraphBatch {
                    add_nodes: 3,
                    add_edges: vec![
                        (BatchNodeRef::Added(0), BatchNodeRef::Added(1)),
                        (BatchNodeRef::Added(1), BatchNodeRef::Added(2)),
                    ],
                    effective_at: Some(0),
                    ..Default::default()
                },
                0,
                "fixture".into(),
                "test".into(),
            )
            .unwrap();
        graph
    }

    #[test]
    fn immutable_pending_materialization_needs_no_live_or_writable_owner() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("read-source.ggj");
        let store = crate::DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
        let graph = store.graph().unwrap();
        let base = graph.read_view().unwrap();
        graph.validate_bound_source(&base, true).unwrap();
        let head = graph.current_state_id();
        let mut pending = PropertyBatch::new(graph.write_coordinate());
        pending
            .stage_nodes(graph, 1, &HashMap::new(), Some(10))
            .unwrap();
        pending.stage_endpoint_revisions(vec![(0, 0, 3)], Some(20));
        pending.stage_removals(true, vec![1], Some(30));
        pending
            .stage(
                true,
                HashMap::from([("score".into(), vec![(3, AttrValue::Float64(1.0000000001))])]),
            )
            .unwrap();
        pending
            .stage(
                false,
                HashMap::from([("weight".into(), vec![(0, AttrValue::Int(7))])]),
            )
            .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        drop(store);
        // The only input is a checked immutable pin. No Subgraph, Arc<Graph>,
        // mutable owner borrow or durable file handle survives into this call.
        let view = pending
            .capture_membership(&base, base.nodes(), base.edges(), false)
            .unwrap();
        assert_eq!(view.node_ids(), [0, 2, 3]);
        assert_eq!(view.edge_ids(), [0]);
        assert_eq!(view.edge_endpoints(0).unwrap(), (0, 3));
        assert_eq!(
            view.node_attr(3, &"score".into()).unwrap(),
            Some(AttrValue::Float64(1.0000000001))
        );
        assert_eq!(
            view.edge_attr(0, &"weight".into()).unwrap(),
            Some(AttrValue::Int(7))
        );
        assert_eq!(view.write_state_id(), None);
        assert_eq!(
            crate::algorithms::AlgorithmRead::new(view.clone())
                .unwrap()
                .bfs_distances(0, None)
                .unwrap(),
            [(0, 0), (3, 1)]
        );
        assert_eq!(base.edge_endpoints(0).unwrap(), (0, 1));
        let reopened = crate::DurableGraph::open(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(reopened.graph().unwrap().current_state_id(), head);
        assert_eq!(reopened.graph().unwrap().node_count(), 3);
        assert_eq!(reopened.graph().unwrap().edge_endpoints(0).unwrap(), (0, 1));
        // Reopening establishes a different owner. Matching numeric IDs/head do
        // not authorize publication of the old prepared operation.
        assert!(reopened
            .graph()
            .unwrap()
            .validate_write_coordinate(&pending.coordinate)
            .is_err());
    }

    #[test]
    fn immutable_source_frames_keep_exact_membership_and_endpoint_closure() {
        let graph = fixture();
        let base = graph.read_view().unwrap();
        graph.validate_bound_source(&base, true).unwrap();
        let mut pending = PropertyBatch::new(graph.write_coordinate());
        pending
            .stage_nodes(&graph, 1, &HashMap::new(), None)
            .unwrap();
        pending.stage_endpoint_revisions(vec![(0, 0, 3)], None);
        let nodes = HashSet::from([0, 3]);
        let edges = HashSet::from([0]);
        let child = pending
            .capture_membership(&base, &nodes, &edges, true)
            .unwrap();
        assert_eq!(child.node_ids(), [0, 3]);
        assert_eq!(child.edge_ids(), [0]);
        assert!(pending
            .capture_membership(&base, &HashSet::from([0]), &edges, true)
            .is_err());
        assert!(pending
            .capture_membership(&base, &HashSet::from([99]), &HashSet::new(), false)
            .is_err());
        pending.stage_removals(true, vec![0], None);
        assert!(pending
            .capture_membership(&base, &nodes, &edges, true)
            .is_err());
        assert_eq!(child.node_ids(), [0, 3]);
        assert_eq!(child.edge_endpoints(0).unwrap(), (0, 3));
        assert_eq!(graph.current_state_id(), 1);
        assert_eq!(graph.pool().statistics().node_count, 3);
    }

    #[test]
    fn immutable_frame_validation_cannot_hide_foreign_staged_property_targets() {
        let graph = fixture();
        let base = graph.read_view().unwrap();
        let mut pending = PropertyBatch::new(graph.write_coordinate());
        pending
            .stage(
                true,
                HashMap::from([("value".into(), vec![(2, AttrValue::Int(4))])]),
            )
            .unwrap();
        let child = HashSet::from([0]);
        assert!(pending
            .capture_membership(&base, &child, &HashSet::new(), true)
            .is_ok());
        pending.stage_removals(true, vec![2], None);
        let error = pending
            .capture_membership(&base, &child, &HashSet::new(), true)
            .unwrap_err();
        assert!(error.to_string().contains("target 2"));
        assert_eq!(graph.current_state_id(), 1);
        assert!(graph.has_node(2));
    }
}
