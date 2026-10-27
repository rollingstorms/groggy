use groggy::{
    state::{change, change_tracker, state::GraphSnapshot, Delta},
    AttrValue as V, Graph, StateRestoreMode as M,
};
use std::collections::HashMap;

#[test]
fn snapshot_history_merge_and_restore_use_exact_payload_differences_with_storage_aliases_normalized(
) {
    let mut g = Graph::new();
    g.add_node().unwrap();
    g.set_node_attr(0, "value".into(), V::Float64(0.0)).unwrap();
    let plus = g.current_state_id();
    g.create_branch("plus".into()).unwrap();
    g.set_node_attr(0, "value".into(), V::Float64(-0.0))
        .unwrap();
    let minus = g.current_state_id();
    let diff = g.diff_commits(plus, minus).unwrap();
    assert_eq!(diff.attr_changes.len(), 1);
    let Some(V::Float64(old)) = diff.attr_changes[0].old_value else {
        panic!()
    };
    let Some(V::Float64(new)) = diff.attr_changes[0].new_value else {
        panic!()
    };
    assert_eq!(old.to_bits(), 0.0_f64.to_bits());
    assert_eq!(new.to_bits(), (-0.0_f64).to_bits());
    let state = g.restore_state(plus, minus, M::Tracked).unwrap();
    assert_eq!(g.diff_commits(minus, state).unwrap().attr_changes.len(), 1);
    assert!(g.diff_commits(plus, state).unwrap().attr_changes.is_empty());
    assert_eq!(g.diff_commits(state, minus).unwrap().attr_changes.len(), 1);
}
#[test]
fn exact_snapshot_diffs_retain_width_nan_bits_absence_and_skip_inactive_rows() {
    let mut before = GraphSnapshot::empty(1);
    before.active_nodes = vec![0];
    before.node_attributes.insert(
        0,
        HashMap::from([
            ("width".into(), V::Float(1.0)),
            (
                "nan".into(),
                V::Float64Vec(vec![f64::from_bits(0x7ff8000000000001)]),
            ),
            ("null".into(), V::Null),
            ("alias".into(), V::SmallInt(7)),
        ]),
    );
    let mut after = before.clone();
    after.state_id = 2;
    let row = after.node_attributes.get_mut(&0).unwrap();
    row.insert("width".into(), V::Float64(1.0));
    row.insert(
        "nan".into(),
        V::Float64Vec(vec![f64::from_bits(0x7ff8000000000002)]),
    );
    row.remove("null");
    row.insert("alias".into(), V::Int(7));
    let diff = before.try_diff_with(&after).unwrap();
    assert_eq!(
        diff.attribute_changes
            .iter()
            .map(|c| c.attr_name.as_str())
            .collect::<Vec<_>>(),
        vec!["nan", "null", "width"]
    );
    assert!(before.try_diff_with(&before).unwrap().is_empty());
    // An inactive row must not be mistaken for before-properties on activation.
    before.active_nodes.clear();
    let diff = before.try_diff_with(&after).unwrap();
    assert_eq!(diff.nodes_added, vec![0]);
    assert!(diff.attribute_changes.iter().all(|c| c.old_value.is_none()));
    let mut invalid = after.clone();
    invalid
        .node_attributes
        .get_mut(&0)
        .unwrap()
        .insert("nan".into(), V::Json("{".into()));
    assert!(invalid.try_diff_with(&invalid).is_err());
    assert!(!invalid.diff_with(&invalid).is_empty());
}
#[test]
fn canonical_changeset_legacy_export_is_identical_and_delta_moves_owned_payload_buffers() {
    let payload = vec![1.0_f64; 100_000];
    let pointer = payload.as_ptr();
    let names = vec!["long-property-name".to_owned(); 2];
    let names_pointer = names.as_ptr();
    let changes = change::ChangeSet {
        nodes_added: vec![1, 0],
        node_attr_changes: vec![
            (0, "vector".into(), None, Some(V::Float64Vec(payload))),
            (0, "names".into(), None, Some(V::TextVec(names))),
        ],
        ..change::ChangeSet::new()
    };
    let legacy: change_tracker::ChangeSet = changes;
    let delta = Delta::from_changes(legacy);
    assert_eq!(delta.nodes_added, vec![0, 1]);
    let Some(V::TextVec(names)) = &delta.node_attr_changes[0].3 else {
        panic!()
    };
    let Some(V::Float64Vec(values)) = &delta.node_attr_changes[1].3 else {
        panic!()
    };
    assert_eq!(names.as_ptr(), names_pointer);
    assert_eq!(values.as_ptr(), pointer);
}
