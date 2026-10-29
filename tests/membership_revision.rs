//! Full-selection invalidation is independent of property and endpoint-only writes.
use groggy::{state::space::GraphSpace, storage::pool::GraphPool, Graph, StateRestoreMode};
use std::{cell::RefCell, rc::Rc};
#[test]
fn every_scalar_and_bulk_membership_path_changes_the_stamp_only_when_ids_change() {
    let pool = Rc::new(RefCell::new(GraphPool::new()));
    pool.borrow_mut().add_nodes_bulk(8);
    pool.borrow_mut().add_edges(&[(0, 1), (1, 2)]);
    let mut space = GraphSpace::new(pool, 0);
    let mut revisions = vec![space.membership_revision()];
    space.activate_node(0);
    revisions.push(space.membership_revision());
    space.activate_nodes([1, 2]);
    revisions.push(space.membership_revision());
    space.activate_edge(0, 0, 1);
    revisions.push(space.membership_revision());
    space.activate_edges([1]);
    revisions.push(space.membership_revision());
    let stable = space.membership_revision();
    space.activate_node(0);
    space.activate_nodes([0, 1, 2]);
    space.activate_edge(0, 0, 1);
    space.activate_edges([0, 1]);
    space.deactivate_node(7);
    space.deactivate_nodes(&[7]);
    space.deactivate_edge(7);
    space.deactivate_edges(&[7]);
    assert_eq!(stable, space.membership_revision());
    space.deactivate_edge(0);
    revisions.push(space.membership_revision());
    space.deactivate_edges(&[1]);
    revisions.push(space.membership_revision());
    space.deactivate_node(2);
    revisions.push(space.membership_revision());
    space.deactivate_nodes(&[0, 1]);
    revisions.push(space.membership_revision());
    for pair in revisions.windows(2) {
        assert_ne!(pair[0], pair[1]);
    }
    assert_eq!(space.node_count(), 0);
    assert_eq!(space.edge_count(), 0);
}
#[test]
fn properties_and_rewiring_preserve_membership_but_restoration_has_new_instance_identity() {
    let mut graph = Graph::new();
    graph.add_nodes(3).unwrap();
    graph.add_edge(0, 1).unwrap();
    let first = graph.current_state_id();
    let stamp = graph.space().membership_revision();
    graph
        .set_node_attr(0, "value".into(), groggy::AttrValue::Int(7))
        .unwrap();
    graph.set_edge_endpoints(0, 2, 1).unwrap();
    assert_eq!(graph.space().membership_revision(), stamp);
    assert_eq!(graph.edge_endpoints(0).unwrap(), (2, 1));
    graph.remove_edge(0).unwrap();
    graph.add_edge(0, 2).unwrap();
    assert_ne!(graph.space().membership_revision(), stamp);
    let second = graph.current_state_id();
    graph
        .restore_state(first, second, StateRestoreMode::Switch)
        .unwrap();
    let old = graph.space().membership_revision();
    let numeric = graph.space().get_version();
    graph
        .restore_state(second, first, StateRestoreMode::Switch)
        .unwrap();
    assert_eq!(graph.space().get_version(), numeric);
    assert_ne!(graph.space().membership_revision(), old);
}
