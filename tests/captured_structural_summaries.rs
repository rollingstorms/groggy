//! Structural summaries agree with independent simple undirected combinatorics.
use groggy::{subgraphs::Subgraph, traits::SubgraphOperations, types::GraphType, Graph, GraphView};
use std::{cell::RefCell, collections::HashSet, rc::Rc};
fn close(left: f64, right: f64) {
    assert!((left - right).abs() < 1e-12, "{left} != {right}");
}
#[test]
fn every_four_node_simple_graph_matches_independent_triangle_triple_counts() {
    for mask in 0..64 {
        let mut graph = Graph::new_manual();
        graph.add_nodes(4).unwrap();
        let mut adjacent = [[false; 4]; 4];
        for (bit, (s, t)) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            .into_iter()
            .enumerate()
        {
            if mask & (1 << bit) != 0 {
                graph.add_edge(s, t).unwrap();
                adjacent[s][t] = true;
                adjacent[t][s] = true;
            }
        }
        let mut expected = Vec::new();
        let mut all_closed = 0.;
        let mut all_triples = 0.;
        for node in 0..4 {
            let neighbors: Vec<_> = (0..4).filter(|&other| adjacent[node][other]).collect();
            let mut closed = 0.;
            let mut triples = 0.;
            for (position, &a) in neighbors.iter().enumerate() {
                for &b in &neighbors[position + 1..] {
                    triples += 1.;
                    if adjacent[a][b] {
                        closed += 1.;
                    }
                }
            }
            expected.push(if triples == 0. { 0. } else { closed / triples });
            all_closed += closed;
            all_triples += triples;
        }
        let view = graph.read_view().unwrap();
        let prepared = view.structural_read().unwrap();
        let rows = prepared.clustering_coefficients().unwrap();
        for (node, value) in rows {
            close(value, expected[node]);
            close(
                view.clustering_coefficient(Some(node)).unwrap(),
                expected[node],
            );
        }
        let transitivity = if all_triples == 0. {
            0.
        } else {
            all_closed / all_triples
        };
        close(prepared.transitivity().unwrap(), transitivity);
        close(view.transitivity().unwrap(), transitivity);
        let average = expected.iter().sum::<f64>() / 4.;
        close(view.clustering_coefficient(None).unwrap(), average);
        let selected = Subgraph::new(
            Rc::new(RefCell::new(graph)),
            view.nodes().clone(),
            view.edges().clone(),
            "all".into(),
        );
        close(selected.clustering_coefficient(None).unwrap(), average);
        close(selected.transitivity().unwrap(), transitivity);
        for (node, &expected) in expected.iter().enumerate() {
            close(
                selected.clustering_coefficient(Some(node)).unwrap(),
                expected,
            );
        }
    }
}
#[test]
fn excluded_edges_loops_parallel_mass_and_isolates_do_not_invent_triangles() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(4).unwrap();
    for (s, t) in [(0, 1), (1, 2), (0, 2), (0, 1), (0, 0)] {
        graph.add_edge(s, t).unwrap();
    }
    let root = graph.read_view().unwrap();
    close(root.clustering_coefficient(None).unwrap(), 0.75);
    close(root.transitivity().unwrap(), 1.);
    let view = GraphView::from_membership(
        root.read_state().clone(),
        root.nodes().clone(),
        HashSet::from([0, 1, 3, 4]),
    )
    .unwrap();
    close(view.clustering_coefficient(None).unwrap(), 0.);
    close(view.clustering_coefficient(Some(0)).unwrap(), 0.);
    close(view.transitivity().unwrap(), 0.);
    let owner = Rc::new(RefCell::new(graph));
    let selected = Subgraph::new(
        owner,
        view.nodes().clone(),
        view.edges().clone(),
        "wedge".into(),
    );
    close(selected.clustering_coefficient(None).unwrap(), 0.);
    close(selected.transitivity().unwrap(), 0.);
    close(root.density(), 5. / 6.); // density counts edge instances, unlike simple clustering
}
#[test]
fn prepared_topology_outlives_property_changes_rewiring_owner_and_source_view() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<groggy::algorithms::UndirectedStructure>();
    let mut graph = Graph::new_manual();
    graph.add_nodes(4).unwrap();
    for (s, t) in [(0, 1), (1, 2), (2, 0)] {
        graph.add_edge(s, t).unwrap();
    }
    let owner = Rc::new(RefCell::new(graph));
    let weak = Rc::downgrade(&owner);
    let selected = Subgraph::new(
        owner.clone(),
        HashSet::from([0, 1, 2, 3]),
        HashSet::from([0, 1, 2]),
        "triangle".into(),
    );
    let input = selected.structural_read().unwrap();
    owner.borrow_mut().set_edge_endpoints(2, 2, 3).unwrap();
    close(selected.clustering_coefficient(None).unwrap(), 0.);
    close(input.clustering_coefficient(None).unwrap(), 0.75);
    drop(selected);
    drop(owner);
    assert!(weak.upgrade().is_none());
    std::thread::spawn(move || close(input.transitivity().unwrap(), 1.))
        .join()
        .unwrap();
}
#[test]
fn empty_one_node_invalid_ids_and_directed_domain_are_explicit() {
    for count in [0, 1] {
        let mut graph = Graph::new_manual();
        graph.add_nodes(count).unwrap();
        let view = graph.read_view().unwrap();
        close(view.clustering_coefficient(None).unwrap(), 0.);
        close(view.transitivity().unwrap(), 0.);
        close(view.density(), 0.);
        assert_eq!(view.clustering_coefficients().unwrap().len(), count);
        assert!(view.clustering_coefficient(Some(99)).is_err());
    }
    let mut graph = Graph::new_with_type_manual(GraphType::Directed);
    graph.add_nodes(2).unwrap();
    graph.add_edge(0, 1).unwrap();
    let view = graph.read_view().unwrap();
    close(view.density(), 0.5);
    assert!(view.structural_read().is_err());
    assert!(view.clustering_coefficient(None).is_err());
    assert!(view.transitivity().is_err());
    let selected = Subgraph::new(
        Rc::new(RefCell::new(graph)),
        view.nodes().clone(),
        view.edges().clone(),
        "directed".into(),
    );
    close(selected.density(), 0.5);
    close(SubgraphOperations::density(&selected), 0.5);
    assert!(selected.structural_read().is_err());
}
