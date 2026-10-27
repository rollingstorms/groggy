use groggy::{state::ChangeSet, AttrValue as V, Graph, HistoryForest};
use std::{collections::HashSet, sync::Arc};
fn change(value: V) -> ChangeSet {
    let mut c = ChangeSet::new();
    c.node_attr_changes
        .push((0, "value".into(), Some(V::Int(7)), Some(value)));
    c
}
#[test]
fn logical_hash_collisions_preserve_exact_payloads_reuse_exact_duplicates_and_collect_variants() {
    for (a, b) in [
        (V::Float64(0.0), V::Float64(-0.0)),
        (
            V::Float(f32::from_bits(0x7fc00001)),
            V::Float(f32::from_bits(0x7fc00002)),
        ),
        (
            V::Float64Vec(vec![f64::from_bits(0x7ff8000000000001)]),
            V::Float64Vec(vec![f64::from_bits(0x7ff8000000000002)]),
        ),
    ] {
        let mut h = HistoryForest::new();
        let mut root = ChangeSet::new();
        root.nodes_added.push(0);
        let root = h
            .create_commit(root, "root".into(), "test".into(), None)
            .unwrap();
        let first = h
            .create_commit(change(a.clone()), "a".into(), "test".into(), Some(root))
            .unwrap();
        let second = h
            .create_commit(change(b), "b".into(), "test".into(), Some(root))
            .unwrap();
        let repeat = h
            .create_commit(
                change(a.clone()),
                "repeat".into(),
                "test".into(),
                Some(root),
            )
            .unwrap();
        assert_eq!(
            h.get_commit(first).unwrap().delta.content_hash,
            h.get_commit(second).unwrap().delta.content_hash
        );
        assert!(!Arc::ptr_eq(
            &h.get_commit(first).unwrap().delta,
            &h.get_commit(second).unwrap().delta
        ));
        assert!(Arc::ptr_eq(
            &h.get_commit(first).unwrap().delta,
            &h.get_commit(repeat).unwrap().delta
        ));
        assert_eq!(h.statistics().total_deltas, 3);
        assert_eq!(h.garbage_collect(&HashSet::from([second])), 2);
        assert_eq!(h.statistics().total_deltas, 2);
        let third = h
            .create_commit(change(a), "after collect".into(), "test".into(), Some(root))
            .unwrap();
        assert!(!Arc::ptr_eq(
            &h.get_commit(second).unwrap().delta,
            &h.get_commit(third).unwrap().delta
        ));
        assert_eq!(h.statistics().total_deltas, 3);
    }
}
#[test]
fn full_archive_import_keeps_both_signed_zero_payloads_with_their_original_legacy_hashes() {
    let mut g = Graph::new();
    g.add_node().unwrap();
    g.set_node_attr(0, "value".into(), V::Int(7)).unwrap();
    g.create_branch("minus".into()).unwrap();
    g.set_node_attr(0, "value".into(), V::Float64(0.0)).unwrap();
    let plus = g.current_state_id();
    g.checkout_branch("minus".into()).unwrap();
    g.set_node_attr(0, "value".into(), V::Float64(-0.0))
        .unwrap();
    let minus = g.current_state_id();
    let restored = g.prepare_archive().unwrap().restore().unwrap();
    for (state, bits) in [(plus, 0.0_f64.to_bits()), (minus, (-0.0_f64).to_bits())] {
        if let Some(V::Float64(value)) = restored
            .snapshot_at_commit(state)
            .unwrap()
            .node_attr(0, &"value".into())
        {
            assert_eq!(value.to_bits(), bits);
        } else {
            panic!()
        }
    }
    restored.check_integrity().unwrap();
}
