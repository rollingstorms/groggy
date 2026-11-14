use groggy::ownership::OwnerCell;
use groggy::{
    subgraphs::{composer::MetaNodePlan, Subgraph},
    traits::{subgraph_operations::NodeStrategy, SubgraphOperations},
    AttrValue as V, Graph,
};
use std::collections::HashSet;
use std::sync::Arc;

fn selection(graph: &Arc<OwnerCell<Graph>>, nodes: &[usize], edges: &[usize]) -> Subgraph {
    Subgraph::new(
        graph.clone(),
        nodes.iter().copied().collect(),
        edges.iter().copied().collect(),
        "component".into(),
    )
    .freeze()
    .unwrap()
}
fn plan(selection: &Subgraph, destructive: bool) -> MetaNodePlan {
    let mut plan = MetaNodePlan::new(selection.node_set().clone(), selection.edge_set().clone())
        .with_node_agg("total".into(), "sum".into(), Some("x".into()));
    if destructive {
        plan.node_strategy = NodeStrategy::Collapse;
    }
    plan
}

#[test]
fn collection_accepts_once_and_retains_each_original_source_in_both_policies() {
    for manual in [false, true] {
        for destructive in [false, true] {
            let graph = Arc::new(OwnerCell::new(if manual {
                Graph::new_manual()
            } else {
                Graph::new()
            }));
            {
                let mut graph = graph.borrow_mut();
                graph.add_nodes(4).unwrap();
                for (id, value) in [2, 4, 7, 9].into_iter().enumerate() {
                    graph.set_node_attr(id, "x".into(), V::Int(value)).unwrap();
                }
                graph.add_edges(&[(0, 1), (2, 3)]).unwrap();
                graph.commit("before".into(), "test".into()).unwrap();
            }
            let first = selection(&graph, &[0, 1], &[0]);
            let second = selection(&graph, &[2, 3], &[1]);
            let before = graph.borrow().current_state_id();
            let metas = MetaNodePlan::execute_many(
                &[
                    (plan(&first, destructive), first.clone()),
                    (plan(&second, destructive), second),
                ],
                false,
                None,
            )
            .unwrap();
            assert_eq!(metas.iter().map(|m| m.id()).collect::<Vec<_>>(), vec![4, 5]);
            assert_eq!(
                graph.borrow().current_state_id(),
                before + u64::from(!manual)
            );
            for (meta, nodes, total) in [
                (&metas[0], HashSet::from([0, 1]), 6),
                (&metas[1], HashSet::from([2, 3]), 16),
            ] {
                assert_eq!(
                    graph
                        .borrow()
                        .get_node_attr(meta.id(), &"total".into())
                        .unwrap(),
                    Some(V::Int(total))
                );
                assert_eq!(
                    meta.expand_to_subgraph().unwrap().unwrap().node_set(),
                    &nodes
                );
            }
            assert_eq!(
                first.get_node_attribute(0, &"x".into()).unwrap(),
                Some(V::Int(2))
            );
            assert_eq!(graph.borrow().node_count(), if destructive { 2 } else { 6 });
            if manual {
                graph
                    .borrow_mut()
                    .commit("after".into(), "test".into())
                    .unwrap();
            }
            assert!(!graph.borrow().has_uncommitted_changes());
            let snapshot = graph.borrow().snapshot_at_commit(before + 1).unwrap();
            assert_eq!(snapshot.node_attr(5, &"total".into()), Some(V::Int(16)));
        }
    }
}

#[test]
fn late_aggregation_failure_preserves_everything_and_allocation() {
    let graph = Arc::new(OwnerCell::new(Graph::new()));
    graph.borrow_mut().add_nodes(2).unwrap();
    graph
        .borrow_mut()
        .set_node_attr(0, "x".into(), V::Int(2))
        .unwrap();
    let first = selection(&graph, &[0], &[]);
    let second = selection(&graph, &[1], &[]);
    let before = graph.borrow().read_view().unwrap();
    let groups = graph.borrow().pool().subgraph_count();
    assert!(MetaNodePlan::execute_many(
        &[(plan(&first, true), first), (plan(&second, true), second)],
        false,
        None
    )
    .is_err());
    assert_eq!(
        graph.borrow().current_state_id(),
        before.write_state_id().unwrap()
    );
    assert_eq!(graph.borrow().node_count(), 2);
    assert_eq!(graph.borrow().pool().subgraph_count(), groups);
    assert!(!graph.borrow().has_uncommitted_changes());
    assert_eq!(graph.borrow_mut().add_node().unwrap(), 2);
}

#[test]
fn mismatched_owner_stale_source_and_destructive_overlap_reject_before_effects() {
    let graph = Arc::new(OwnerCell::new(Graph::new()));
    graph.borrow_mut().add_node().unwrap();
    let first = selection(&graph, &[0], &[]);
    let empty_plan = || MetaNodePlan::new(HashSet::from([0]), HashSet::new());
    assert!(MetaNodePlan::execute_many(
        &[
            (plan(&first, true), first.clone()),
            (plan(&first, true), first.clone())
        ],
        true,
        None
    )
    .is_err());
    let other = Arc::new(OwnerCell::new(Graph::new()));
    other.borrow_mut().add_node().unwrap();
    assert!(MetaNodePlan::execute_many(
        &[
            (empty_plan(), first.clone()),
            (empty_plan(), selection(&other, &[0], &[]))
        ],
        true,
        None
    )
    .is_err());
    let before = graph.borrow().current_state_id();
    graph
        .borrow_mut()
        .set_node_attr(0, "x".into(), V::Int(8))
        .unwrap();
    assert!(MetaNodePlan::execute_many(&[(empty_plan(), first)], true, None).is_err());
    assert_eq!(graph.borrow().current_state_id(), before + 1);
    assert_eq!(graph.borrow_mut().add_node().unwrap(), 1);
    assert!(MetaNodePlan::execute_many(&[], false, None)
        .unwrap()
        .is_empty());
}

#[test]
fn collection_remaps_each_new_meta_endpoint_and_publishes_one_clock() {
    let mut owner = Graph::new_directed();
    owner.add_nodes(3).unwrap();
    owner.add_edges(&[(0, 2), (2, 1)]).unwrap();
    let graph = Arc::new(OwnerCell::new(owner));
    let a = selection(&graph, &[0], &[]);
    let b = selection(&graph, &[1], &[]);
    let mut first = MetaNodePlan::new(HashSet::from([0]), HashSet::new());
    first.node_strategy = NodeStrategy::Collapse;
    let mut second = MetaNodePlan::new(HashSet::from([1]), HashSet::new());
    second.node_strategy = NodeStrategy::Collapse;
    let metas = MetaNodePlan::execute_many(&[(first, a), (second, b)], true, None).unwrap();
    assert_eq!(
        metas.iter().map(|meta| meta.id()).collect::<Vec<_>>(),
        vec![3, 4]
    );
    let graph = graph.borrow();
    let pairs: HashSet<_> = graph
        .edge_ids()
        .into_iter()
        .map(|edge| graph.edge_endpoints(edge).unwrap())
        .collect();
    assert_eq!(pairs, HashSet::from([(3, 2), (2, 4)]));
    for edge in graph.edge_ids() {
        let (source, target) = graph.edge_endpoints(edge).unwrap();
        assert_eq!(
            graph.get_edge_attr(edge, &"source".into()).unwrap(),
            Some(V::Int(source as i64))
        );
        assert_eq!(
            graph.get_edge_attr(edge, &"target".into()).unwrap(),
            Some(V::Int(target as i64))
        );
    }

    let metadata = graph
        .publication_metadata(graph.current_state_id())
        .unwrap()
        .unwrap();
    assert!(metadata
        .change_times
        .values()
        .all(|&time| time == metadata.recorded_at));
    let view = graph.read_view_at(metadata.recorded_at, None).unwrap();
    assert_eq!(view.node_count(), 3);
    assert_eq!(view.edge_count(), 2);
}

#[test]
fn cross_selection_destructive_boundary_rejects_without_installing_groups() {
    let graph = Arc::new(OwnerCell::new(Graph::new()));
    graph.borrow_mut().add_nodes(2).unwrap();
    graph.borrow_mut().add_edge(0, 1).unwrap();
    let a = selection(&graph, &[0], &[]);
    let b = selection(&graph, &[1], &[]);
    let mut first = MetaNodePlan::new(HashSet::from([0]), HashSet::new());
    first.node_strategy = NodeStrategy::Collapse;
    let second = MetaNodePlan::new(HashSet::from([1]), HashSet::new());
    let before = graph.borrow().current_state_id();
    let groups = graph.borrow().pool().subgraph_count();
    assert!(MetaNodePlan::execute_many(&[(first, a), (second, b)], true, None).is_err());
    assert_eq!(graph.borrow().current_state_id(), before);
    assert_eq!(graph.borrow().pool().subgraph_count(), groups);
    assert_eq!(graph.borrow_mut().add_node().unwrap(), 2);
    assert_eq!(graph.borrow_mut().add_edge(0, 1).unwrap(), 1);
}
