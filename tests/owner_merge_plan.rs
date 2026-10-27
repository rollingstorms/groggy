use groggy::{
    state::{SnapshotMergeConflict as C, SnapshotMergeResolution as R},
    AttrValue as V, Graph, GraphMergeOutcome, GraphMergePlan, GraphMergeReadPlan, GraphSchema,
};
use std::collections::HashMap;
fn assert_send_sync<T: Send + Sync>() {}
fn fixture() -> (Graph, u64, u64, u64) {
    let mut g = Graph::new();
    g.add_nodes(2).unwrap();
    g.add_edge(0, 1).unwrap();
    g.set_node_attr(0, "value".into(), V::Int(0)).unwrap();
    let base = g.current_state_id();
    g.create_branch("right".into()).unwrap();
    g.set_node_attr(0, "value".into(), V::Int(1)).unwrap();
    let left = g.current_state_id();
    g.checkout_branch("right".into()).unwrap();
    g.set_node_attr(0, "value".into(), V::Int(2)).unwrap();
    let right = g.current_state_id();
    g.checkout_branch("main".into()).unwrap();
    (g, base, left, right)
}
fn resolved(g: &Graph, right: u64, left: u64) -> GraphMergeOutcome {
    g.prepare_merge_read(right, None, left)
        .unwrap()
        .execute()
        .unwrap()
        .resolve_with(|c| match c {
            C::NodeProperty { right, .. } => Ok(R::Property(right.clone())),
            _ => panic!("unexpected conflict"),
        })
        .unwrap()
}
#[test]
fn captured_merge_reconstructs_off_owner_and_preflight_preserves_every_mutable_coordinate() {
    assert_send_sync::<GraphMergeReadPlan>();
    assert_send_sync::<GraphMergePlan>();
    assert_send_sync::<GraphMergeOutcome>();
    let (mut g, base, left, right) = fixture();
    let versions = g.history_records().len();
    let pin = g.read_view().unwrap();
    let capture = g.prepare_merge_read(right, None, left).unwrap();
    let plan = std::thread::spawn(move || capture.execute().unwrap())
        .join()
        .unwrap();
    assert_eq!(plan.base_state(), base);
    assert_eq!(plan.parents(), [left, right]);
    assert_eq!(plan.conflicts().len(), 1);
    let outcome = plan
        .resolve_with(|c| match c {
            C::NodeProperty { right, .. } => Ok(R::Property(right.clone())),
            _ => panic!(),
        })
        .unwrap();
    assert_eq!(outcome.base_state(), base);
    assert_eq!(outcome.parents(), [left, right]);
    assert_eq!(outcome.snapshot().state_id, 0);
    assert_eq!(outcome.snapshot().node_attributes[&0]["value"], V::Int(2));
    g.compact_pool_with_reclamation(true).unwrap();
    g.validate_merge_outcome(&outcome, None).unwrap();
    assert_eq!(g.current_state_id(), left);
    assert_eq!(g.history_records().len(), versions);
    assert_eq!(
        g.get_node_attr(0, &"value".into()).unwrap(),
        Some(V::Int(1))
    );
    assert_eq!(pin.node_attr(0, &"value".into()).unwrap(), Some(V::Int(1)));
    assert_eq!(g.add_node().unwrap(), 2); // capture/resolve/validate allocates no IDs
}
#[test]
fn stale_foreign_changed_branch_and_collected_parent_outcomes_reject_before_mutation() {
    let (mut g, _, left, right) = fixture();
    let outcome = resolved(&g, right, left);
    let foreign = g.prepare_archive().unwrap().restore().unwrap();
    assert!(foreign.validate_merge_outcome(&outcome, None).is_err());
    g.create_branch("same".into()).unwrap();
    g.checkout_branch("same".into()).unwrap();
    assert!(g.validate_merge_outcome(&outcome, None).is_err());
    g.checkout_branch("main".into()).unwrap();
    g.prune_branch("same").unwrap();
    g.prune_branch("right").unwrap();
    assert!(g.validate_merge_outcome(&outcome, None).is_err());
    assert_eq!(g.current_state_id(), left);
    let (mut g, _, left, right) = fixture();
    let outcome = resolved(&g, right, left);
    g.add_node().unwrap();
    assert!(g.validate_merge_outcome(&outcome, None).is_err());
}
#[test]
fn capture_rejects_dirty_unknown_identical_noncommon_and_unrelated_implicit_bases() {
    let (g, base, left, right) = fixture();
    assert!(g.prepare_merge_read(right, None, base).is_err());
    assert!(g.prepare_merge_read(left, None, left).is_err());
    assert!(g.prepare_merge_read(999, None, left).is_err());
    assert!(g.prepare_merge_read(right, Some(left), left).is_err());
    assert!(g.prepare_merge_read(right, Some(999), left).is_err());
    let mut manual = Graph::new_manual();
    manual.add_node().unwrap();
    assert!(manual.prepare_merge_read(right, None, 0).is_err());
    let mut roots = Graph::new();
    roots.create_branch("other".into()).unwrap();
    roots.add_node().unwrap();
    let first = roots.current_state_id();
    roots.checkout_branch("other".into()).unwrap();
    roots.add_node().unwrap();
    let second = roots.current_state_id();
    assert!(roots.prepare_merge_read(first, None, second).is_err());
    let outcome = roots
        .prepare_merge_read(first, Some(0), second)
        .unwrap()
        .execute()
        .unwrap()
        .into_outcome()
        .unwrap();
    roots.validate_merge_outcome(&outcome, None).unwrap();
    assert_eq!(outcome.snapshot().active_nodes, vec![0, 1]);
}
#[test]
fn owner_preflight_rejects_missing_groups_unallocated_resolution_ids_and_stricter_schema() {
    let (g, _, left, right) = fixture();
    let missing = g
        .prepare_merge_read(right, None, left)
        .unwrap()
        .execute()
        .unwrap()
        .resolve_with(|_| Ok(R::Property(Some(V::SubgraphRef(999)))))
        .unwrap();
    assert!(g.validate_merge_outcome(&missing, None).is_err());
    let mut g = g;
    g.set_node_labels(0, vec!["Entity".into()]).unwrap();
    let updated = g.current_state_id();
    let outcome = resolved(&g, right, updated);
    let schema =
        GraphSchema::from_json(r#"{"node_labels":{"Entity":{"value":{"type":"text"}}}}"#).unwrap();
    assert!(g.validate_merge_outcome(&outcome, Some(&schema)).is_err());
    // An explicit topology answer may name a node, but only allocated owner IDs pass preflight.
    let mut g = Graph::new();
    g.add_nodes(3).unwrap();
    g.add_edge(0, 1).unwrap();
    let base = g.current_state_id();
    g.create_branch("right".into()).unwrap();
    g.set_edge_endpoints(0, 1, 2).unwrap();
    let left = g.current_state_id();
    g.checkout_branch("right".into()).unwrap();
    g.set_edge_endpoints(0, 2, 0).unwrap();
    let right = g.current_state_id();
    g.checkout_branch("main".into()).unwrap();
    let outcome = g
        .prepare_merge_read(right, Some(base), left)
        .unwrap()
        .execute()
        .unwrap()
        .resolve_with(|c| match c {
            C::EdgeEndpoints { .. } => Ok(R::Endpoints((0, 999))),
            C::Topology { .. } => Ok(R::EndpointNodes(vec![(999, HashMap::new())])),
            _ => panic!(),
        })
        .unwrap();
    assert!(g.validate_merge_outcome(&outcome, None).is_err());
    assert_eq!(g.edge_endpoints(0).unwrap(), (1, 2));
}

#[test]
fn custom_entity_resolution_cannot_steal_a_reserved_external_identity() {
    let mut g = Graph::new();
    g.add_nodes(2).unwrap();
    g.writer("test")
        .set_node_external_id(0, "reserved".into())
        .unwrap();
    let base = g.current_state_id();
    g.create_branch("right".into()).unwrap();
    g.remove_node(1).unwrap();
    let left = g.current_state_id();
    g.checkout_branch("right".into()).unwrap();
    g.set_node_attr(1, "changed".into(), V::Int(1)).unwrap();
    let right = g.current_state_id();
    g.checkout_branch("main".into()).unwrap();
    let outcome = g
        .prepare_merge_read(right, Some(base), left)
        .unwrap()
        .execute()
        .unwrap()
        .resolve_with(|c| match c {
            C::Node {
                right: Some(record),
                ..
            } => {
                let mut record = record.clone();
                record
                    .properties
                    .insert("external_id".into(), V::Text("reserved".into()));
                Ok(R::Entity(Some(record)))
            }
            _ => panic!(),
        })
        .unwrap();
    assert!(g.validate_merge_outcome(&outcome, None).is_err());
    assert_eq!(g.node_by_external_id("reserved", None), Some(0));
    assert_eq!(g.current_state_id(), left);
    assert!(!g.has_node(1));
}
