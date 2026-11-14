//! Traversal may never leave selected edges/nodes and later re-enter the result.
use groggy::ownership::OwnerCell;
use groggy::{subgraphs::Subgraph, traits::SubgraphOperations, types::GraphType, Graph, GraphView};
use std::collections::HashSet;
use std::sync::Arc;
fn fixture(directed: bool) -> (Graph, GraphView) {
    let mut graph = Graph::new_with_type_manual(if directed {
        GraphType::Directed
    } else {
        GraphType::Undirected
    });
    graph.add_nodes(6).unwrap();
    for (s, t) in [(0, 1), (1, 2), (0, 3), (3, 2), (0, 1), (1, 1), (2, 4)] {
        graph.add_edge(s, t).unwrap();
    }
    let root = graph.read_view().unwrap();
    (graph, root)
}
fn exact(root: &GraphView, nodes: &[usize], edges: &[usize]) -> GraphView {
    GraphView::from_membership(
        root.read_state().clone(),
        nodes.iter().copied().collect(),
        edges.iter().copied().collect(),
    )
    .unwrap()
}
#[test]
fn excluded_intermediate_nodes_and_instances_cannot_supply_paths() {
    for directed in [false, true] {
        let (_, root) = fixture(directed);
        let selected = exact(&root, &[0, 2, 4], &[6]);
        assert!(root.has_path(0, 2).unwrap());
        assert!(!selected.has_path(0, 2).unwrap());
        assert_eq!(selected.bfs(0, None).unwrap().node_ids(), vec![0]);
        assert_eq!(selected.dfs(0, None).unwrap().node_ids(), vec![0]);
        assert!(selected.shortest_path_subgraph(0, 2).unwrap().is_none());
        let route = exact(&root, &[0, 1, 2, 3, 4], &[2, 3, 6]);
        let path = route.shortest_path_subgraph(0, 4).unwrap().unwrap();
        assert_eq!(path.node_ids(), vec![0, 2, 3, 4]);
        assert_eq!(path.edge_ids(), vec![2, 3, 6]);
    }
}
#[test]
fn ascending_neighbor_and_edge_ties_bound_tree_edges_without_induction() {
    let (_, root) = fixture(true);
    assert_eq!(root.bfs(0, None).unwrap().edge_ids(), vec![0, 1, 2, 6]);
    assert_eq!(root.dfs(0, None).unwrap().edge_ids(), vec![0, 1, 2, 6]);
    for tree in [root.bfs(0, Some(0)).unwrap(), root.dfs(0, Some(0)).unwrap()] {
        assert_eq!(tree.node_ids(), vec![0]);
        assert!(tree.edge_ids().is_empty());
    }
    for tree in [root.bfs(0, Some(1)).unwrap(), root.dfs(0, Some(1)).unwrap()] {
        assert_eq!(tree.node_ids(), vec![0, 1, 3]);
        assert_eq!(tree.edge_ids(), vec![0, 2]);
    }
    assert!(root.bfs(99, None).is_err());
    assert!(root.dfs(99, None).is_err());
    assert!(!root.has_path(99, 99).unwrap());
    assert!(root.shortest_path_subgraph(99, 99).unwrap().is_none());
    assert!(root.shortest_path_subgraph(4, 0).unwrap().is_none());
    let same = root.shortest_path_subgraph(0, 0).unwrap().unwrap();
    assert_eq!(same.node_ids(), vec![0]);
    assert!(same.edge_ids().is_empty());
}
#[test]
fn all_three_node_directed_graphs_match_independent_reachability_and_hop_distances() {
    for mask in 0..512 {
        let mut graph = Graph::new_with_type_manual(GraphType::Directed);
        graph.add_nodes(3).unwrap();
        let mut distances = [[99usize; 3]; 3];
        for (s, row) in distances.iter_mut().enumerate() {
            row[s] = 0;
            for (t, cell) in row.iter_mut().enumerate() {
                if mask & (1 << (s * 3 + t)) != 0 {
                    graph.add_edge(s, t).unwrap();
                    *cell = (*cell).min(1);
                }
            }
        }
        for k in 0..3 {
            for s in 0..3 {
                for t in 0..3 {
                    distances[s][t] = distances[s][t].min(distances[s][k] + distances[k][t]);
                }
            }
        }
        let view = graph.read_view().unwrap();
        for (source, distances) in distances.iter().enumerate() {
            let expected: Vec<_> = (0..3).filter(|&node| distances[node] != 99).collect();
            for tree in [
                view.bfs(source, None).unwrap(),
                view.dfs(source, None).unwrap(),
            ] {
                assert_eq!(tree.node_ids(), expected, "mask {mask}, source {source}");
                assert_eq!(tree.edge_count() + 1, tree.node_count());
            }
            for (target, &distance) in distances.iter().enumerate() {
                assert_eq!(view.has_path(source, target).unwrap(), distance != 99);
                let path = view.shortest_path_subgraph(source, target).unwrap();
                assert_eq!(path.is_some(), distance != 99);
                if let Some(path) = path {
                    assert_eq!(path.edge_count(), distance);
                    assert_eq!(path.node_count(), distance + 1);
                }
            }
        }
    }
}
#[test]
fn captured_trees_outlive_mutation_and_drop_and_connected_adapters_share_scope() {
    let (graph, root) = fixture(true);
    let selected = exact(&root, &[0, 2, 4], &[6]);
    let graph_owner = Arc::new(OwnerCell::new(graph));
    let live = Subgraph::new(
        graph_owner.clone(),
        selected.nodes().clone(),
        selected.edges().clone(),
        "selected".into(),
    );
    assert_eq!(live.bfs(0, None).unwrap().node_count(), 1);
    assert_eq!(live.dfs(0, None).unwrap().node_count(), 1);
    assert!(!live.has_path(0, 2).unwrap());
    assert!(live.shortest_path_subgraph(0, 2).unwrap().is_none());
    // Inducing within a noninduced selection must never resurrect parallel edge 4.
    let exact = Subgraph::new(
        graph_owner.clone(),
        HashSet::from([0, 1]),
        HashSet::from([0]),
        "exact".into(),
    );
    assert_eq!(
        exact.induced_subgraph(&[0, 1]).unwrap().edge_set(),
        &HashSet::from([0])
    );
    graph_owner.borrow_mut().remove_node(1).unwrap();
    drop(live);
    drop(exact);
    drop(graph_owner);
    std::thread::spawn(move || assert_eq!(root.bfs(0, None).unwrap().edge_ids(), vec![0, 1, 2, 6]))
        .join()
        .unwrap();
}

#[test]
fn connected_component_fallback_and_cache_use_selected_edge_instances() {
    let (graph, root) = fixture(false);
    let owner = Arc::new(OwnerCell::new(graph));
    let selected = Subgraph::new(
        owner.clone(),
        HashSet::from([0, 1, 2]),
        HashSet::from([0]),
        "scoped".into(),
    );
    for _ in 0..2 {
        let components = selected.connected_components().unwrap();
        assert_eq!(components.len(), 2);
        assert_eq!(components[0].node_set(), &HashSet::from([0, 1]));
        assert_eq!(components[0].edge_set(), &HashSet::from([0]));
        assert_eq!(components[1].node_set(), &HashSet::from([2]));
    }
    owner
        .borrow_mut()
        .set_node_attr(0, "change".into(), groggy::AttrValue::Int(1))
        .unwrap();
    assert_eq!(selected.connected_components().unwrap().len(), 2);
    assert_eq!(
        root.select_nodes(&[0, 1, 2])
            .unwrap()
            .connected_components(groggy::algorithms::community::ComponentMode::Undirected)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn restoration_with_equal_numeric_space_versions_invalidates_connected_reader_caches() {
    let mut graph = Graph::new();
    graph.add_nodes(3).unwrap();
    graph.add_edge(0, 1).unwrap();
    let first = graph.current_state_id();
    graph.set_edge_endpoints(0, 0, 2).unwrap();
    let second = graph.current_state_id();
    graph
        .restore_state(first, second, groggy::StateRestoreMode::Switch)
        .unwrap();
    let first_version = graph.space().get_version();
    let owner = Arc::new(OwnerCell::new(graph));
    let selection = Subgraph::new(
        owner.clone(),
        HashSet::from([0, 1, 2]),
        HashSet::from([0]),
        "live".into(),
    );
    assert_eq!(selection.neighbors(0).unwrap(), vec![1]);
    assert_eq!(selection.degree(1).unwrap(), 1);
    assert_eq!(
        selection.connected_components().unwrap()[0].node_set(),
        &HashSet::from([0, 1])
    );
    assert!(!selection.has_path(0, 2).unwrap()); // primes validation too
    owner
        .borrow_mut()
        .restore_state(second, first, groggy::StateRestoreMode::Switch)
        .unwrap();
    assert_eq!(owner.borrow().space().get_version(), first_version);
    assert_eq!(selection.neighbors(0).unwrap(), vec![2]);
    assert_eq!(selection.degree(1).unwrap(), 0);
    assert_eq!(selection.degree(2).unwrap(), 1);
    assert_eq!(
        selection.connected_components().unwrap()[0].node_set(),
        &HashSet::from([0, 2])
    );
    assert!(selection.has_path(0, 2).unwrap());
    assert!(!selection.has_path(0, 1).unwrap());
}

#[test]
fn connected_csr_rejects_removed_entities_and_rewired_outside_endpoints() {
    for warm in [false, true] {
        for remove_node in [false, true] {
            let mut graph = Graph::new();
            graph.add_nodes(3).unwrap();
            graph.add_edge(0, 1).unwrap();
            let frozen = graph.read_view().unwrap();
            let owner = Arc::new(OwnerCell::new(graph));
            let selected = Subgraph::new(
                owner.clone(),
                HashSet::from([0, 1]),
                HashSet::from([0]),
                "selection".into(),
            );
            if warm {
                assert_eq!(selected.neighbors(0).unwrap(), vec![1]);
                assert_eq!(selected.degree(1).unwrap(), 1);
            }
            if remove_node {
                owner.borrow_mut().remove_node(1).unwrap();
            } else {
                owner.borrow_mut().remove_edge(0).unwrap();
            }
            let state = owner.borrow().current_state_id();
            for node in [0, 1] {
                assert!(selected.neighbors(node).is_err());
                assert!(selected.degree(node).is_err());
            }
            assert_eq!(owner.borrow().current_state_id(), state);
            assert_eq!(frozen.neighbors(0).unwrap(), vec![1]);
            assert_eq!(frozen.degree(1).unwrap(), 1);
        }
    }
    let mut graph = Graph::new();
    graph.add_nodes(3).unwrap();
    graph.add_edge(0, 1).unwrap();
    let owner = Arc::new(OwnerCell::new(graph));
    let selected = Subgraph::new(
        owner.clone(),
        HashSet::from([0, 1]),
        HashSet::from([0]),
        "selection".into(),
    );
    assert_eq!(selected.neighbors(0).unwrap(), vec![1]);
    owner.borrow_mut().set_edge_endpoints(0, 0, 2).unwrap();
    assert!(selected.neighbors(0).is_err());
    assert!(selected.degree(0).is_err());
    owner.borrow_mut().set_edge_endpoints(0, 0, 1).unwrap();
    assert_eq!(selected.neighbors(0).unwrap(), vec![1]);
}
