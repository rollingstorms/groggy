use groggy::{subgraphs::Subgraph, AttrValue, Graph};
use std::{cell::RefCell, collections::HashSet, rc::Rc};

fn fixture(directed: bool) -> Subgraph {
    let mut graph = if directed {
        Graph::new_directed()
    } else {
        Graph::new()
    };
    graph.add_nodes(8).unwrap();
    for node in 0..6 {
        graph.add_edge(node, node).unwrap();
    }
    for pair in [(0, 1), (0, 1), (1, 0), (1, 2), (2, 3), (6, 7)] {
        graph.add_edge(pair.0, pair.1).unwrap();
    }
    graph
        .set_node_attr(0, "value".into(), AttrValue::Int(42))
        .unwrap();
    Subgraph::new(
        Rc::new(RefCell::new(graph)),
        (0..6).collect(),
        (0..10).collect(),
        "selected".into(),
    )
}

#[test]
fn samples_share_seeded_membership_and_induce_only_selected_instances_in_both_directions() {
    for directed in [false, true] {
        let connected = fixture(directed);
        let frozen = connected.pin_view().unwrap();
        for count in [0, 1, 3, 6, usize::MAX] {
            for seed in 0..16 {
                let live = connected.sample(count, Some(seed)).unwrap();
                let sample = frozen.sample(count, Some(seed)).unwrap();
                assert_eq!(sample.node_count(), count.min(6));
                assert_eq!(live.nodes(), sample.nodes());
                assert_eq!(live.edges(), sample.edges());
                let expected: HashSet<_> = frozen
                    .edges()
                    .iter()
                    .copied()
                    .filter(|&edge| {
                        let (source, target) = frozen.edge_endpoints(edge).unwrap();
                        sample.nodes().contains(&source) && sample.nodes().contains(&target)
                    })
                    .collect();
                assert_eq!(sample.edges(), &expected);
                assert!(std::ptr::eq(
                    sample.read_state().snapshot(),
                    frozen.read_state().snapshot()
                ));
                assert!(sample.merge_with(&frozen).is_ok());
                assert_eq!(sample.write_state_id(), frozen.write_state_id());
            }
        }
    }
}

#[test]
fn seeded_calls_do_not_consume_shared_random_state_and_are_not_prefix_selection() {
    let connected = fixture(false);
    fastrand::seed(71);
    let expected = fastrand::u64(..);
    fastrand::seed(71);
    let mut seen = HashSet::new();
    for seed in 0..32 {
        seen.extend(
            connected
                .sample(1, Some(seed))
                .unwrap()
                .nodes()
                .iter()
                .copied(),
        );
    }
    assert_eq!(fastrand::u64(..), expected);
    assert_eq!(seen, (0..6).collect());
}

#[test]
fn invalid_connected_scope_rejects_even_zero_or_oversized_samples_and_frozen_samples_survive() {
    let connected = fixture(true);
    let frozen = connected.pin_view().unwrap();
    connected.graph().borrow_mut().remove_node(0).unwrap();
    for count in [0, 2, usize::MAX] {
        assert!(connected.sample(count, Some(1)).is_err());
        assert_eq!(
            frozen.sample(count, Some(1)).unwrap().node_count(),
            count.min(6)
        );
    }
    let invalid = Subgraph::new(connected.graph(), [1].into(), [10].into(), "invalid".into());
    assert!(invalid.sample(0, None).is_err());
    drop(invalid);
    drop(connected);
    let result = std::thread::spawn(move || {
        let all = frozen.sample(usize::MAX, Some(3)).unwrap();
        assert_eq!(
            all.node_attr(0, &"value".into()).unwrap(),
            Some(AttrValue::Int(42))
        );
        all.sample(2, Some(7)).unwrap().node_ids()
    })
    .join()
    .unwrap();
    assert_eq!(result.len(), 2);
}
