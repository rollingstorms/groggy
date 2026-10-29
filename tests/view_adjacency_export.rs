use groggy::{subgraphs::Subgraph, Graph};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

fn fixture(directed: bool) -> Subgraph {
    let mut graph = if directed {
        Graph::new_directed()
    } else {
        Graph::new()
    };
    graph.add_nodes(5).unwrap();
    for (s, t) in [(0, 2), (0, 1), (0, 2), (2, 2), (3, 0)] {
        graph.add_edge(s, t).unwrap();
    }
    Subgraph::new(
        Rc::new(RefCell::new(graph)),
        (0..5).collect(),
        (0..5).collect(),
        "selection".into(),
    )
}

#[test]
fn live_and_frozen_exports_keep_directedness_parallel_occurrences_loops_and_isolates() {
    for directed in [false, true] {
        let selection = fixture(directed);
        let frozen = selection.pin_view().unwrap();
        let expected = BTreeMap::from([
            (
                0,
                if directed {
                    vec![1, 2, 2]
                } else {
                    vec![1, 2, 2, 3]
                },
            ),
            (1, if directed { vec![] } else { vec![0] }),
            (2, if directed { vec![2] } else { vec![0, 0, 2] }),
            (3, vec![0]),
            (4, vec![]),
        ]);
        assert_eq!(selection.adjacency_list().unwrap(), expected);
        assert_eq!(frozen.adjacency_list().unwrap(), expected);
        // Warm and cloned connected inputs retain the same complete rows.
        assert_eq!(selection.clone().adjacency_list().unwrap(), expected);
        let empty = frozen.select_nodes(&[]).unwrap();
        assert!(empty.adjacency_list().unwrap().is_empty());
        assert_eq!(
            frozen.select_nodes(&[4]).unwrap().adjacency_list().unwrap(),
            BTreeMap::from([(4, vec![])])
        );
    }
}

#[test]
fn subsets_never_reintroduce_excluded_edges_and_reads_outlive_owner_mutation_and_drop() {
    let original = fixture(true);
    let selection = Subgraph::new(
        original.graph(),
        [0, 2, 4].into(),
        [0, 3].into(),
        "narrow".into(),
    );
    let frozen = selection.pin_view().unwrap();
    let expected = BTreeMap::from([(0, vec![2]), (2, vec![2]), (4, vec![])]);
    let detached = selection.adjacency_read().unwrap();
    assert_eq!(frozen.adjacency_list().unwrap(), expected);
    original
        .graph()
        .borrow_mut()
        .set_edge_endpoints(0, 3, 2)
        .unwrap();
    assert!(selection.adjacency_read().is_err());
    assert!(selection.adjacency_list().is_err());
    assert_eq!(detached.adjacency_list(), expected);
    drop(selection);
    drop(original);
    let rows = std::thread::spawn(move || {
        assert_eq!(frozen.adjacency_list().unwrap(), detached.adjacency_list());
        detached.adjacency_list()
    })
    .join()
    .unwrap();
    assert_eq!(rows, expected);
}

#[test]
fn removal_and_equal_version_restoration_do_not_reuse_wrong_topology() {
    let selection = fixture(true);
    let first = selection.graph().borrow().current_state_id();
    let original = selection.adjacency_list().unwrap();
    selection
        .graph()
        .borrow_mut()
        .set_edge_endpoints(0, 4, 2)
        .unwrap();
    let second = selection.graph().borrow().current_state_id();
    let changed = selection.adjacency_list().unwrap();
    assert_ne!(changed, original);
    selection
        .graph()
        .borrow_mut()
        .restore_state(first, second, groggy::StateRestoreMode::Switch)
        .unwrap();
    assert_eq!(selection.adjacency_list().unwrap(), original);
    selection
        .graph()
        .borrow_mut()
        .restore_state(second, first, groggy::StateRestoreMode::Switch)
        .unwrap();
    assert_eq!(selection.adjacency_list().unwrap(), changed);
    selection.graph().borrow_mut().remove_edge(1).unwrap();
    assert!(selection.adjacency_list().is_err());
}
