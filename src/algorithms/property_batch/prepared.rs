//! Owned execution effects, independent of the pending read/capture machinery.
//! Preparing a payload grants no mutation authority and publishes nothing. The
//! receiving coordinator must validate its complete owner/branch/head coordinate
//! immediately before acceptance. This is not a durable pipeline executor.
use super::*;
use std::ops::Range;

/// No graph owner, read pin, callback, cache or history payload survives here.
#[derive(Debug)]
pub(super) struct PreparedEffects {
    coordinate: WriteCoordinate,
    batch: Option<GraphBatch>,
    projection: SelectionProjection,
}

/// Ranges preserve global provisional positions without allocating ID vectors.
#[derive(Debug)]
pub(super) struct SelectionProjection {
    topology: bool,
    nodes: Range<NodeId>,
    edges: Range<EdgeId>,
}

impl PropertyBatch {
    /// Move the staged column values and explicit clocks into a canonical batch.
    /// Accepted IDs and the default engine clock are resolved only at publication.
    pub(super) fn prepare(self) -> PreparedEffects {
        let topology = self.has_topology();
        let (nodes, edges) = self.allocation_base.unwrap_or_default();
        let projection = SelectionProjection {
            topology,
            nodes: nodes..nodes + self.add_nodes,
            edges: edges..edges + self.add_edges.len(),
        };
        let empty = !topology && self.nodes.is_empty() && self.edges.is_empty();
        let batch =
            if empty {
                None
            } else {
                let mut change_times = self.change_times;
                change_times.extend(self.property_times.into_iter().map(
                    |((node, id, name), at)| {
                        let target = if node {
                            BatchChangeRef::NodeAttribute(BatchNodeRef::Existing(id), name)
                        } else {
                            BatchChangeRef::EdgeAttribute(BatchEdgeRef::Existing(id), name)
                        };
                        (target, at)
                    },
                ));
                Some(GraphBatch {
                    add_nodes: self.add_nodes,
                    add_edges: self.add_edges,
                    remove_nodes: self.remove_nodes,
                    remove_edges: self.remove_edges,
                    edge_endpoints: self.edge_endpoints,
                    change_times,
                    node_attrs: self
                        .nodes
                        .into_iter()
                        .map(|(name, rows)| {
                            (
                                name,
                                rows.into_iter()
                                    .map(|(id, value)| (BatchNodeRef::Existing(id), value))
                                    .collect(),
                            )
                        })
                        .collect(),
                    edge_attrs: self
                        .edges
                        .into_iter()
                        .map(|(name, rows)| {
                            (
                                name,
                                rows.into_iter()
                                    .map(|(id, value)| (BatchEdgeRef::Existing(id), value))
                                    .collect(),
                            )
                        })
                        .collect(),
                    ..Default::default()
                })
            };
        PreparedEffects {
            coordinate: self.coordinate,
            batch,
            projection,
        }
    }
}

impl PreparedEffects {
    /// Empty effects still validate the complete coordinate. All nonempty effects
    /// go through the existing publication coordinator exactly once.
    pub(super) fn accept(self, graph: &mut Graph) -> GraphResult<SelectionProjection> {
        graph.validate_write_coordinate(&self.coordinate)?;
        if let Some(batch) = self.batch {
            graph.apply_batch(
                batch,
                self.coordinate.state_id(),
                "atomic builder effects".into(),
                "groggy".into(),
            )?;
        }
        Ok(self.projection)
    }

    pub(super) fn accept_selection(self, subgraph: Subgraph) -> GraphResult<Subgraph> {
        let binding = subgraph.graph();
        let projection = self.accept(&mut binding.borrow_mut())?;
        if !projection.topology {
            return Ok(subgraph);
        }
        let graph = binding.borrow();
        let nodes = subgraph
            .nodes()
            .iter()
            .copied()
            .chain(projection.nodes)
            .filter(|id| graph.has_node(*id))
            .collect();
        let edges = subgraph
            .edges()
            .iter()
            .copied()
            .chain(projection.edges)
            .filter(|id| graph.has_edge(*id))
            .collect();
        drop(graph);
        Ok(Subgraph::new(
            binding,
            nodes,
            edges,
            subgraph.subgraph_type().into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::publication::ChangeKey;

    fn fixture() -> Graph {
        let mut graph = Graph::new_publishing();
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
    fn preparation_moves_values_and_drops_capture_authority_without_publishing() {
        let owner = fixture().into_shared();
        let selected = Subgraph::new(
            owner.clone(),
            [0, 1, 2].into(),
            [0, 1].into(),
            "root".into(),
        );
        let head = owner.borrow().current_state_id();
        let mut pending =
            PropertyBatch::for_selection(owner.borrow().write_coordinate(), &selected);
        let view = pending.capture(&selected).unwrap();
        let weak = view.read_state().downgrade_capture();
        drop(view);
        let text = "owned value".repeat(1024);
        let buffer = text.as_ptr();
        pending
            .stage(
                true,
                HashMap::from([("text".into(), vec![(0, AttrValue::Text(text))])]),
            )
            .unwrap();
        let prepared = pending.prepare();
        assert!(
            weak.upgrade().is_none(),
            "preparation retained a read payload"
        );
        assert_eq!(
            Arc::strong_count(&owner),
            2,
            "preparation retained an owner"
        );
        let AttrValue::Text(value) =
            &prepared.batch.as_ref().unwrap().node_attrs[&AttrName::from("text")][0].1
        else {
            panic!("text type changed");
        };
        assert_eq!(
            value.as_ptr(),
            buffer,
            "preparation cloned the value buffer"
        );
        assert_eq!(owner.borrow().current_state_id(), head);
        assert_eq!(
            owner.borrow().get_node_attr(0, &"text".into()).unwrap(),
            None
        );
        assert!(!owner.borrow().has_uncommitted_changes());
        prepared.accept(&mut owner.borrow_mut()).unwrap();
        assert_eq!(owner.borrow().current_state_id(), head + 1);
    }

    #[test]
    fn delayed_topology_acceptance_projects_once_and_preserves_typed_values_and_clocks() {
        let owner = fixture().into_shared();
        let selected = Subgraph::new(owner.clone(), [0, 1].into(), [0].into(), "selected".into());
        let before = owner.borrow().read_view().unwrap();
        let mut pending =
            PropertyBatch::for_selection(owner.borrow().write_coordinate(), &selected);
        assert_eq!(
            pending
                .stage_nodes(&owner.borrow(), 1, &HashMap::new(), Some(10))
                .unwrap(),
            [3]
        );
        assert_eq!(
            pending
                .stage_edges(
                    &owner.borrow(),
                    &selected,
                    &[(BatchNodeRef::Existing(0), BatchNodeRef::Added(0))],
                    &HashMap::new(),
                    Some(20)
                )
                .unwrap(),
            [2]
        );
        pending.stage_endpoint_revisions(vec![(0, 0, 3)], Some(20));
        pending.stage_removals(true, vec![1], Some(30));
        pending
            .stage_at(
                true,
                HashMap::from([(
                    "precise".into(),
                    vec![(3, AttrValue::Float64(1.0000000001))],
                )]),
                Some(40),
            )
            .unwrap();
        pending
            .stage_at(
                false,
                HashMap::from([("empty".into(), vec![(2, AttrValue::Null)])]),
                Some(50),
            )
            .unwrap();
        pending
            .stage(
                true,
                HashMap::from([("default_clock".into(), vec![(0, AttrValue::Bool(true))])]),
            )
            .unwrap();
        let prepared = pending.prepare();
        assert_eq!(owner.borrow().pool().statistics().node_count, 3);
        assert_eq!(owner.borrow().pool().statistics().edge_count, 2);
        assert_eq!(owner.borrow().current_state_id(), 1);
        assert!(prepared.batch.as_ref().unwrap().effective_at.is_none());
        let result = prepared.accept_selection(selected).unwrap();
        assert_eq!(result.nodes(), &HashSet::from([0, 3]));
        assert_eq!(result.edges(), &HashSet::from([0, 2]));
        assert_eq!(result.subgraph_type(), "selected");
        let graph = owner.borrow();
        assert_eq!(graph.current_state_id(), 2);
        assert_eq!(
            graph.get_node_attr(3, &"precise".into()).unwrap(),
            Some(AttrValue::Float64(1.0000000001))
        );
        assert_eq!(
            graph.get_edge_attr(2, &"empty".into()).unwrap(),
            Some(AttrValue::Null)
        );
        let metadata = graph.publication_metadata(2).unwrap().unwrap();
        assert_eq!(
            metadata.change_times[&ChangeKey::NodeAttribute(3, "precise".into())],
            40
        );
        assert_eq!(
            metadata.change_times[&ChangeKey::EdgeAttribute(2, "empty".into())],
            50
        );
        assert_eq!(
            metadata.change_times[&ChangeKey::NodeAttribute(0, "default_clock".into())],
            metadata.recorded_at
        );
        assert_eq!(before.edge_endpoints(0).unwrap(), (0, 1));
        assert_eq!(before.node_ids(), [0, 1, 2]);
        assert_eq!(
            graph
                .read_view_at(39, None)
                .unwrap()
                .node_attr(3, &"precise".into())
                .unwrap(),
            None
        );
        assert_eq!(
            graph
                .read_view_at(40, None)
                .unwrap()
                .node_attr(3, &"precise".into())
                .unwrap(),
            Some(AttrValue::Float64(1.0000000001))
        );
    }

    #[test]
    fn delayed_acceptance_rechecks_owner_branch_head_and_dirty_state_even_for_empty_effects() {
        for empty in [false, true] {
            for changed in ["owner", "branch", "head", "dirty"] {
                let mut graph = fixture();
                let mut pending = PropertyBatch::new(graph.write_coordinate());
                if !empty {
                    pending
                        .stage_nodes(&graph, 1, &HashMap::new(), None)
                        .unwrap();
                    pending
                        .stage(
                            true,
                            HashMap::from([("x".into(), vec![(0, AttrValue::Int(7))])]),
                        )
                        .unwrap();
                }
                let prepared = pending.prepare();
                match changed {
                    "owner" => graph = fixture(),
                    "branch" => {
                        graph.create_branch("other".into()).unwrap();
                        graph.checkout_branch("other".into()).unwrap();
                    }
                    "head" => {
                        graph
                            .set_node_attr(0, "other".into(), AttrValue::Int(1))
                            .unwrap();
                    }
                    "dirty" => {
                        graph.set_automatic_publication(false).unwrap();
                        graph
                            .set_node_attr(0, "other".into(), AttrValue::Int(1))
                            .unwrap();
                    }
                    _ => unreachable!(),
                }
                let head = graph.current_state_id();
                let dirty = graph.has_uncommitted_changes();
                assert!(
                    prepared.accept(&mut graph).is_err(),
                    "{changed}, empty={empty}"
                );
                assert_eq!(graph.current_state_id(), head);
                assert_eq!(graph.has_uncommitted_changes(), dirty);
                assert_eq!(graph.pool().statistics().node_count, 3);
                assert_eq!(graph.get_node_attr(0, &"x".into()).unwrap(), None);
            }
        }
    }

    #[test]
    fn discarded_preparation_and_late_coordinator_failure_consume_no_ids() {
        for discard in [false, true] {
            let mut graph = fixture();
            let before = graph.read_view().unwrap();
            let mut pending = PropertyBatch::new(graph.write_coordinate());
            pending
                .stage_nodes(&graph, 1, &HashMap::new(), None)
                .unwrap();
            // Staging permits assembly before the canonical target checks.
            pending
                .stage(
                    true,
                    HashMap::from([("x".into(), vec![(99, AttrValue::Int(7))])]),
                )
                .unwrap();
            let prepared = pending.prepare();
            if discard {
                drop(prepared);
            } else {
                assert!(prepared.accept(&mut graph).is_err());
            }
            assert_eq!(graph.current_state_id(), 1);
            assert!(!graph.has_uncommitted_changes());
            assert_eq!(graph.node_count(), 3);
            assert_eq!(before.node_ids(), [0, 1, 2]);
            assert_eq!(graph.add_node().unwrap(), 3);
        }
    }

    #[test]
    fn empty_preparation_returns_the_existing_selection_without_publication() {
        let owner = fixture().into_shared();
        let selected = Subgraph::new(owner.clone(), [0, 1].into(), [0].into(), "selected".into());
        let batch = PropertyBatch::for_selection(owner.borrow().write_coordinate(), &selected);
        let prepared = batch.prepare();
        assert!(prepared.batch.is_none());
        let result = prepared.accept_selection(selected).unwrap();
        assert!(Arc::ptr_eq(&owner, &result.graph()));
        assert_eq!(result.nodes(), &HashSet::from([0, 1]));
        assert_eq!(result.edges(), &HashSet::from([0]));
        assert_eq!(owner.borrow().current_state_id(), 1);
    }
}
