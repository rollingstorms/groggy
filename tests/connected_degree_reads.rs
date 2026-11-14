//! Connected degree preparation validates liveness, then computes without an owner.
use groggy::ownership::OwnerCell;
use groggy::{
    subgraphs::{DegreeMode, Subgraph},
    traits::SubgraphOperations,
    types::GraphType,
    Graph,
};
use std::collections::HashSet;
use std::sync::Arc;

#[test]
fn degree_rows_match_frozen_scope_direction_parallel_and_loop_semantics() {
    for directed in [false, true] {
        let mut graph = Graph::new_with_type_manual(if directed {
            GraphType::Directed
        } else {
            GraphType::Undirected
        });
        graph.add_nodes(5).unwrap();
        for (source, target) in [(0, 1), (1, 0), (0, 1), (1, 1), (2, 1), (1, 3)] {
            graph.add_edge(source, target).unwrap();
        }
        let frozen = graph.read_view().unwrap();
        let owner = Arc::new(OwnerCell::new(graph));
        let selected = Subgraph::new(
            owner.clone(),
            HashSet::from([0, 1, 4]),
            HashSet::from([0, 1, 3]),
            "noninduced".into(),
        );
        let selected_frozen = selected.pin_view().unwrap();
        let ids = [4, 1, 0, 1];
        for full in [false, true] {
            let reference = if full { &frozen } else { &selected_frozen };
            for mode in [
                DegreeMode::Incident,
                DegreeMode::Incoming,
                DegreeMode::Outgoing,
            ] {
                let expected: Vec<_> = ids
                    .iter()
                    .map(|&node| match mode {
                        DegreeMode::Incident => reference.degree(node).unwrap(),
                        DegreeMode::Incoming => reference.in_degree(node).unwrap(),
                        DegreeMode::Outgoing => reference.out_degree(node).unwrap(),
                    })
                    .collect();
                let read = selected.degree_read(mode, full).unwrap();
                assert_eq!(read.counts(&ids).unwrap(), expected);
                assert!(read.counts(&[]).unwrap().is_empty());
                // Full-owner edge counting never permits owner-only result IDs.
                assert!(read.counts(&[0, 2, 1]).is_err());
                assert!(read.counts(&[99]).is_err());
            }
        }
        for node in ids {
            assert_eq!(
                selected.degree(node).unwrap(),
                selected_frozen.degree(node).unwrap()
            );
        }
        let read = selected.degree_read(DegreeMode::Incoming, false).unwrap();
        let before = read.counts(&ids).unwrap();
        owner.borrow_mut().remove_node(1).unwrap();
        for full in [false, true] {
            for mode in [
                DegreeMode::Incident,
                DegreeMode::Incoming,
                DegreeMode::Outgoing,
            ] {
                assert!(selected.degree_read(mode, full).is_err());
            }
        }
        drop(selected);
        drop(owner);
        std::thread::spawn(move || assert_eq!(read.counts(&ids).unwrap(), before))
            .join()
            .unwrap();
    }
}

#[test]
fn empty_degree_preparation_is_valid_and_does_not_publish() {
    let owner = Arc::new(OwnerCell::new(Graph::new()));
    let selected = Subgraph::new(
        owner.clone(),
        HashSet::new(),
        HashSet::new(),
        "empty".into(),
    );
    let state = owner.borrow().current_state_id();
    for full in [false, true] {
        let read = selected.degree_read(DegreeMode::Incident, full).unwrap();
        assert!(read.counts(&[]).unwrap().is_empty());
        assert!(read.counts(&[0]).is_err());
    }
    assert_eq!(owner.borrow().current_state_id(), state);
}
