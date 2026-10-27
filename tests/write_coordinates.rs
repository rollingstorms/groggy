use groggy::{Graph, StateRestoreMode as M, WriteCoordinate};
fn send_sync<T: Send + Sync>() {}
#[test]
fn write_coordinates_bind_owner_branch_and_head_even_when_branches_share_a_version() {
    send_sync::<WriteCoordinate>();
    let mut g = Graph::new();
    let empty = g.write_coordinate();
    assert_eq!(empty.state_id(), 0);
    g.validate_write_coordinate(&empty).unwrap();
    g.add_node().unwrap();
    let head = g.current_state_id();
    let coordinate = g.write_coordinate();
    assert_eq!(coordinate.branch(), "main");
    assert_eq!(coordinate.state_id(), head);
    g.create_branch("same".into()).unwrap();
    g.checkout_branch("same".into()).unwrap();
    assert!(g.validate_write_coordinate(&coordinate).is_err());
    g.checkout_branch("main".into()).unwrap();
    g.validate_write_coordinate(&coordinate).unwrap();
    let other = g.prepare_archive().unwrap().restore().unwrap();
    assert!(other.validate_write_coordinate(&coordinate).is_err());
    g.add_node().unwrap();
    assert!(g.validate_write_coordinate(&coordinate).is_err());
    let advanced = coordinate.at_state(g.current_state_id());
    g.validate_write_coordinate(&advanced).unwrap();
    assert!(g
        .validate_write_coordinate(&advanced.at_state(999))
        .is_err());
    let transported =
        std::thread::spawn(move || (coordinate.state_id(), coordinate.branch().to_string()))
            .join()
            .unwrap();
    assert_eq!(transported, (head, "main".into()));
}
#[test]
fn coordinates_pin_no_history_and_do_not_bypass_pending_manual_guards() {
    let mut g = Graph::new();
    g.add_node().unwrap();
    let old = g.current_state_id();
    g.add_node().unwrap();
    let future = g.current_state_id();
    let token = g.write_coordinate();
    g.restore_state(old, future, M::Switch).unwrap();
    assert_eq!(g.gc_history().unwrap(), 1);
    assert!(g.snapshot_at_commit(future).is_err());
    assert!(g.validate_write_coordinate(&token).is_err());
    let mut manual = Graph::new_manual();
    let token = manual.write_coordinate();
    manual.add_node().unwrap();
    assert!(manual.validate_write_coordinate(&token).is_err());
}
