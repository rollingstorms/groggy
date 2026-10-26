//! Borrowed, single-buffer replay preserves the public sequential-delta contract.
use groggy::{
    state::{change_tracker::ChangeSet, Delta, GraphSnapshot},
    AttrValue as V,
};
use std::collections::HashMap;
fn delta(nodes_added: Vec<usize>, nodes_removed: Vec<usize>, value: Option<V>) -> Delta {
    let mut changes = ChangeSet::new();
    changes.nodes_added = nodes_added;
    changes.nodes_removed = nodes_removed;
    changes
        .node_attr_changes
        .push((0, "value".into(), None, value));
    Delta::from_changes(changes)
}
fn exact(a: &GraphSnapshot, b: &GraphSnapshot) {
    assert_eq!(a.active_nodes, b.active_nodes);
    assert_eq!(a.edges, b.edges);
    assert_eq!(a.state_id, b.state_id);
    assert_eq!(a.graph_type, b.graph_type);
    // Native serde retains variant and bit distinctions except nonfinite floats;
    // explicit IEEE assertions below additionally cover those cases.
    assert_eq!(a.node_attributes, b.node_attributes);
    assert_eq!(a.edge_attributes, b.edge_attributes);
}
#[test]
fn reconstruction_matches_sequential_updates_deactivation_readdition_unsets_and_nulls() {
    let mut base = GraphSnapshot::empty(5);
    base.active_nodes = vec![7, 0, 1, 7];
    base.node_attributes.insert(
        0,
        HashMap::from([("old".into(), V::Text("retained".into()))]),
    );
    let mut deltas = vec![
        (delta(vec![2, 2], vec![7, 7], Some(V::Float64(-0.0))), 6),
        (delta(vec![7], vec![2], Some(V::Null)), 7),
        (delta(vec![2], vec![], None), 8),
        (delta(vec![0], vec![0], Some(V::Int(3))), 9),
        (
            delta(
                vec![0],
                vec![],
                Some(V::Float64(f64::from_bits(0x7ff8000000000001))),
            ),
            10,
        ),
    ];
    deltas[0].0.edges_added.push((4, 0, 1));
    deltas[1].0.edge_attr_changes.push((
        4,
        "w".into(),
        None,
        Some(V::FloatVec(vec![-0.0, f32::from_bits(0x7fc00001)])),
    ));
    deltas[3].0.edges_removed.push(4);
    for len in 0..=deltas.len() {
        let expected = deltas[..len]
            .iter()
            .fold(base.clone(), |s, (d, id)| s.apply_delta(d, *id));
        let actual = GraphSnapshot::reconstruct_from_deltas(Some(&base), &deltas[..len]).unwrap();
        exact(&expected, &actual);
    }
    let actual = GraphSnapshot::reconstruct_from_deltas(Some(&base), &deltas).unwrap();
    assert_eq!(actual.active_nodes, vec![1, 7, 2, 0]);
    let V::Float64(value) = actual.node_attributes[&0]["value"] else {
        panic!("f64 tag")
    };
    assert_eq!(value.to_bits(), 0x7ff8000000000001);
    assert_eq!(base.active_nodes, vec![7, 0, 1, 7]);
}
#[test]
fn many_small_node_versions_and_sparse_updates_keep_order_and_latest_values() {
    let deltas: Vec<_> = (0..10_000)
        .map(|i| (delta(vec![i], vec![], Some(V::Int(i as i64))), i as u64 + 1))
        .collect();
    let result = GraphSnapshot::reconstruct_from_deltas(None, &deltas).unwrap();
    assert_eq!(result.active_nodes, (0..10_000).collect::<Vec<_>>());
    assert_eq!(result.node_attributes[&0]["value"], V::Int(9999));
    assert_eq!(result.state_id, 10_000);
}
