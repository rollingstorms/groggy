use groggy::ChangeTracker;
#[test]
fn pending_tracker_cannot_fabricate_inverse_or_conflict_information_and_failed_merge_preserves_records(
) {
    let mut left = ChangeTracker::new();
    let mut right = ChangeTracker::new();
    assert!(left.generate_reverse_operations().unwrap().is_empty());
    assert!(left.would_conflict_with(&right).unwrap().is_empty());
    left.merge(&right).unwrap();
    left.record_node_addition(0);
    right.record_edge_addition(7, 0, 1);
    assert!(left
        .generate_reverse_operations()
        .unwrap_err()
        .to_string()
        .contains("retained state"));
    assert!(left
        .would_conflict_with(&right)
        .unwrap_err()
        .to_string()
        .contains("merge base"));
    assert!(left
        .merge(&right)
        .unwrap_err()
        .to_string()
        .contains("retained parents"));
    assert_eq!(left.indexed_changes().nodes_added, &[0]);
    assert_eq!(right.indexed_changes().edges_added, &[(7, 0, 1)]);
    right.clear();
    left.merge(&right).unwrap();
    assert!(left.would_conflict_with(&right).is_err());
    assert!(right.would_conflict_with(&left).is_err());
    assert_eq!(left.indexed_changes().nodes_added, &[0]);
}
