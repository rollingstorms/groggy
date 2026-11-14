//! Specialized wrappers must share canonical selected topology, not filter owner paths.
use groggy::ownership::OwnerCell;
use groggy::{
    subgraphs::{ComponentSubgraph, FilteredSubgraph, NeighborhoodSubgraph},
    traits::{filter_operations::FilterCriteria, SubgraphOperations},
    types::GraphType,
    Graph,
};
use std::collections::HashSet;
use std::sync::Arc;

fn variants(
    owner: Arc<OwnerCell<Graph>>,
    nodes: HashSet<usize>,
    edges: HashSet<usize>,
) -> Vec<Box<dyn SubgraphOperations>> {
    vec![
        Box::new(FilteredSubgraph::new(
            owner.clone(),
            nodes.clone(),
            edges.clone(),
            FilterCriteria::And(Vec::new()),
            5,
            4,
        )),
        Box::new(ComponentSubgraph::new(
            owner.clone(),
            nodes.clone(),
            edges.clone(),
            7,
            true,
            9,
        )),
        Box::new(NeighborhoodSubgraph::new(owner, vec![0], 2, nodes, edges)),
    ]
}

#[test]
fn specialized_traversal_cannot_leave_selected_edges_then_reenter() {
    for directed in [false, true] {
        let mut graph = Graph::new_with_type_manual(if directed {
            GraphType::Directed
        } else {
            GraphType::Undirected
        });
        graph.add_nodes(5).unwrap();
        for (s, t) in [(0, 1), (1, 2), (2, 4), (0, 2)] {
            graph.add_edge(s, t).unwrap();
        }
        let owner = Arc::new(OwnerCell::new(graph));
        for view in variants(owner.clone(), HashSet::from([0, 2, 4]), HashSet::from([2])) {
            assert_eq!(
                view.bfs(0, None).unwrap().node_set(),
                &HashSet::from([0]),
                "{}",
                view.entity_type()
            );
            assert_eq!(view.dfs(0, None).unwrap().node_set(), &HashSet::from([0]));
            assert!(view.shortest_path_subgraph(0, 4).unwrap().is_none());
            assert_eq!(view.neighbors(0).unwrap(), Vec::<usize>::new());
            assert_eq!(view.degree(0).unwrap(), 0);
            let components = view.connected_components().unwrap();
            assert_eq!(components.len(), 2);
            assert!(components
                .iter()
                .any(|c| c.node_set() == &HashSet::from([0])));
            assert!(components
                .iter()
                .any(|c| c.node_set() == &HashSet::from([2, 4])));
            for component in components {
                assert_eq!(component.entity_type(), view.entity_type());
                assert!(component.node_set().is_subset(view.node_set()));
                assert!(component.edge_set().is_subset(view.edge_set()));
            }
        }
    }
}

#[test]
fn selected_induction_preserves_parallel_instances_loops_and_empty_membership() {
    for directed in [false, true] {
        let mut graph = Graph::new_with_type_manual(if directed {
            GraphType::Directed
        } else {
            GraphType::Undirected
        });
        graph.add_nodes(5).unwrap();
        for (s, t) in [(0, 1), (0, 1), (1, 0), (1, 1), (1, 2)] {
            graph.add_edge(s, t).unwrap();
        }
        let owner = Arc::new(OwnerCell::new(graph));
        for view in variants(
            owner.clone(),
            HashSet::from([0, 1, 4]),
            HashSet::from([0, 2, 3]),
        ) {
            let induced = view.induced_subgraph(&[0, 1, 2]).unwrap();
            assert_eq!(induced.node_set(), &HashSet::from([0, 1]));
            assert_eq!(induced.edge_set(), &HashSet::from([0, 2, 3]));
            assert_eq!(induced.entity_type(), view.entity_type());
            let edges = view.subgraph_from_edges(&[0, 1, 3, 99]).unwrap();
            assert_eq!(edges.edge_set(), &HashSet::from([0, 3]));
            assert_eq!(edges.node_set(), &HashSet::from([0, 1]));
            assert_eq!(edges.entity_type(), view.entity_type());
            assert!(view.induced_subgraph(&[]).unwrap().node_set().is_empty());
            assert!(view.subgraph_from_edges(&[]).unwrap().edge_set().is_empty());
            assert!(view.bfs(99, None).is_err());
            assert!(view.dfs(99, None).is_err());
            assert!(view.neighbors(99).is_err());
            assert!(view.degree(99).is_err());
            assert!(view.shortest_path_subgraph(0, 99).unwrap().is_none());
            let zero = view.shortest_path_subgraph(0, 0).unwrap().unwrap();
            assert_eq!(zero.node_set(), &HashSet::from([0]));
            assert!(zero.edge_set().is_empty());
        }
    }
}

#[test]
fn specialized_clones_share_membership_and_frozen_reads_survive_owner_removal() {
    use groggy::{
        traits::{ComponentOperations, FilterOperations, NeighborhoodOperations},
        AttrValue,
    };
    let mut graph = Graph::new();
    graph.add_nodes(2).unwrap();
    graph.add_edge(0, 1).unwrap();
    graph
        .set_node_attr(0, "value".into(), AttrValue::Int(7))
        .unwrap();
    let owner = Arc::new(OwnerCell::new(graph));
    let filter = FilteredSubgraph::new(
        owner.clone(),
        HashSet::from([0, 1]),
        HashSet::from([0]),
        FilterCriteria::And(Vec::new()),
        2,
        1,
    );
    let component = ComponentSubgraph::new(
        owner.clone(),
        HashSet::from([0, 1]),
        HashSet::from([0]),
        7,
        true,
        9,
    );
    let neighborhood = NeighborhoodSubgraph::new(
        owner.clone(),
        vec![0],
        2,
        HashSet::from([0, 1]),
        HashSet::from([0]),
    );
    assert!(std::ptr::eq(filter.node_set(), filter.clone().node_set()));
    assert!(std::ptr::eq(
        component.node_set(),
        component.clone().node_set()
    ));
    assert!(std::ptr::eq(
        neighborhood.node_set(),
        neighborhood.clone().node_set()
    ));
    assert!(std::ptr::eq(filter.edge_set(), filter.clone().edge_set()));
    assert_eq!(filter.original_node_count(), 2);
    assert_eq!(filter.filter_criteria().complexity_score(), 1);
    assert_eq!(component.component_id(), 7);
    assert_eq!(component.total_components(), 9);
    assert!(component.is_largest_component());
    assert_eq!(neighborhood.central_nodes(), &[0]);
    assert_eq!(neighborhood.hops(), 2);
    let pinned = [
        filter.pin_view().unwrap(),
        component.pin_view().unwrap(),
        neighborhood.pin_view().unwrap(),
    ];
    for view in [
        &filter as &dyn SubgraphOperations,
        &component,
        &neighborhood,
    ] {
        assert_eq!(view.neighbors(0).unwrap(), vec![1]);
        assert_eq!(view.degree(1).unwrap(), 1);
    }
    owner.borrow_mut().remove_node(1).unwrap();
    let state = owner.borrow().current_state_id();
    for view in [
        &filter as &dyn SubgraphOperations,
        &component,
        &neighborhood,
    ] {
        assert!(view.neighbors(0).is_err());
        assert!(view.degree(1).is_err());
        assert!(view.bfs(0, None).is_err());
        assert!(view.dfs(0, None).is_err());
        assert!(view.connected_components().is_err());
        assert!(view.induced_subgraph(&[0]).is_err());
        assert!(view.shortest_path_subgraph(0, 1).is_err());
    }
    assert!(filter.pin_view().is_err());
    assert!(component.pin_view().is_err());
    assert!(neighborhood.pin_view().is_err());
    assert_eq!(owner.borrow().current_state_id(), state);
    drop((filter, component, neighborhood, owner));
    std::thread::spawn(move || {
        for view in pinned {
            assert_eq!(view.neighbors(0).unwrap(), vec![1]);
            assert_eq!(
                view.node_attr(0, &"value".into()).unwrap(),
                Some(AttrValue::Int(7))
            );
        }
    })
    .join()
    .unwrap();
}

#[test]
fn filtered_complement_never_retains_edges_with_excluded_endpoints() {
    use groggy::traits::FilterOperations;
    let mut graph = Graph::new();
    graph.add_nodes(4).unwrap();
    for (s, t) in [(0, 1), (1, 2), (2, 3), (3, 3)] {
        graph.add_edge(s, t).unwrap();
    }
    let owner = Arc::new(OwnerCell::new(graph));
    let filter = FilteredSubgraph::new(
        owner,
        HashSet::from([0, 1]),
        HashSet::from([0]),
        FilterCriteria::And(Vec::new()),
        4,
        4,
    );
    let inverse = filter.not_filter().unwrap();
    assert_eq!(inverse.node_set(), &HashSet::from([2, 3]));
    assert_eq!(inverse.edge_set(), &HashSet::from([2, 3]));
    assert_eq!(
        inverse.bfs(2, None).unwrap().node_set(),
        &HashSet::from([2, 3])
    );
}
