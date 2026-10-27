use groggy::{
    state::delta::{ColumnDelta, ColumnIndexDelta, DeltaObject},
    storage::pool::GraphPool,
    AttrValue as V, ChangeTracker, EdgeEndpointChange, Graph,
};
use std::collections::HashMap;
#[test]
fn pending_index_records_never_fabricate_values_and_complete_raw_delta_preserves_all_changes() {
    let mut pool = GraphPool::new();
    let a = pool.set_attr("value".into(), V::Float64(-0.0), true);
    let b = pool.set_attr("value".into(), V::Text("new".into()), true);
    let e = pool.set_attr("weight".into(), V::Null, false);
    let mut tracker = ChangeTracker::new();
    tracker.record_node_addition(2);
    tracker.record_edge_addition(7, 0, 2);
    tracker.record_attr_change(0usize, "value".into(), Some(a), b, true);
    tracker.record_attr_unset(7, "weight".into(), Some(e), false);
    tracker
        .record_edge_endpoint_change(EdgeEndpointChange {
            edge_id: 8,
            old_endpoints: (0, 1),
            new_endpoints: (2, 1),
        })
        .unwrap();
    assert!(tracker.create_change_set().is_err());
    assert!(tracker.create_changeset().is_err());
    assert!(tracker.get_modified_nodes().contains(&0));
    assert!(tracker.get_modified_edges().contains(&8));
    assert!(tracker.is_edge_modified(8));
    assert!(tracker.memory_usage() > 0);
    let summary = tracker.change_summary();
    assert_eq!(summary.node_attr_changes, 1);
    assert_eq!(summary.edge_attr_changes, 1);
    let physical = tracker.create_delta().unwrap();
    assert!(!physical.is_empty());
    assert_eq!(physical.change_count(), 5);
    let index = physical.index_changes.as_ref().unwrap();
    assert_eq!(index.nodes_added, vec![2]);
    assert_eq!(index.edges_added, vec![(7, 0, 2)]);
    assert_eq!(index.node_attrs["value"].old_column_indices, vec![Some(a)]);
    assert_eq!(index.node_attrs["value"].new_column_indices, vec![Some(b)]);
    assert_eq!(index.edge_attrs["weight"].old_column_indices, vec![Some(e)]);
    assert_eq!(index.edge_attrs["weight"].new_column_indices, vec![None]);
    assert_eq!(physical.edge_endpoint_changes[0].new_endpoints, (2, 1));
    // Logical extraction requires the pool and preserves Null versus absence.
    let logical = tracker.create_changeset_resolved(&pool).unwrap();
    let Some(V::Float64(old)) = logical.node_attr_changes[0].2 else {
        panic!()
    };
    assert_eq!(old.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(logical.node_attr_changes[0].3, Some(V::Text("new".into())));
    assert_eq!(logical.edge_attr_changes[0].2, Some(V::Null));
    assert_eq!(logical.edge_attr_changes[0].3, None);
    tracker.record_attr_change(0usize, "missing".into(), None, 999, true);
    assert!(tracker.create_changeset_resolved(&pool).is_err());
    assert!(tracker.has_changes());
    tracker.clear();
    assert!(tracker.create_change_set().unwrap().is_empty());
    assert!(tracker.create_delta().unwrap().is_empty());
}
#[test]
fn indexed_column_shape_and_order_reject_and_hash_collisions_do_not_equate_different_payloads() {
    for column in [
        ColumnIndexDelta {
            entity_indices: vec![1],
            old_column_indices: vec![],
            new_column_indices: vec![Some(0)],
        },
        ColumnIndexDelta {
            entity_indices: vec![2, 1],
            old_column_indices: vec![None, None],
            new_column_indices: vec![Some(0), Some(1)],
        },
    ] {
        assert!(DeltaObject::new_with_indices(
            [("p".into(), column)].into(),
            HashMap::new(),
            vec![],
            vec![],
            vec![],
            vec![]
        )
        .is_err());
    }
    let make = |value| {
        let mut column = ColumnDelta::new();
        column.add_change(0, V::Float64(value));
        DeltaObject::new(
            [("p".into(), column)].into(),
            HashMap::new(),
            ColumnDelta::new(),
            ColumnDelta::new(),
        )
    };
    let plus = make(0.0);
    let minus = make(-0.0);
    assert_eq!(plus.content_hash, minus.content_hash);
    assert_ne!(plus, minus);
    assert_eq!(plus, make(0.0));
    let mut invalid_column = ColumnDelta::new();
    invalid_column.add_change(0, V::Json("{".into()));
    let invalid = DeltaObject::new(
        [("p".into(), invalid_column)].into(),
        HashMap::new(),
        ColumnDelta::new(),
        ColumnDelta::new(),
    );
    assert_eq!(invalid, invalid.clone());
    let mut a = ColumnIndexDelta::new();
    a.add_index_change(0, None, Some(1));
    let mut b = ColumnIndexDelta::new();
    b.add_index_change(0, None, Some(2));
    let left = DeltaObject::new_with_indices(
        [("p".into(), a)].into(),
        HashMap::new(),
        vec![1],
        vec![],
        vec![],
        vec![],
    )
    .unwrap();
    let mut right = DeltaObject::new_with_indices(
        [("p".into(), b)].into(),
        HashMap::new(),
        vec![1],
        vec![],
        vec![],
        vec![],
    )
    .unwrap();
    right.content_hash = left.content_hash;
    assert_ne!(left, right);
}
#[test]
fn graph_space_cannot_silently_claim_an_empty_pending_history_delta() {
    let graph = Graph::new();
    assert!(graph.space().create_change_delta(&graph.pool()).is_err());
}
