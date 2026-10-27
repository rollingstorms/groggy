use groggy::{
    state::{
        merge_snapshots, plan_snapshot_merge, validate_snapshot, ChangeSet, GraphSnapshot,
        SnapshotMergeConflict,
    },
    AttrValue as V, HistoryForest,
};
use std::collections::HashMap;
fn fixture() -> GraphSnapshot {
    let mut b = GraphSnapshot::empty(1);
    b.active_nodes = vec![0, 1, 2];
    b.edges.insert(0, (0, 1));
    b.node_attributes.insert(
        0,
        [("a".into(), V::Int(1)), ("null".into(), V::Null)].into(),
    );
    b.edge_attributes
        .insert(0, [("weight".into(), V::Int(1))].into());
    b
}
#[test]
fn independent_columns_membership_endpoints_and_parallel_ids_merge_without_policy() {
    let b = fixture();
    let mut l = b.clone();
    let mut r = b.clone();
    l.node_attributes
        .get_mut(&0)
        .unwrap()
        .insert("a".into(), V::Int(2));
    r.node_attributes.get_mut(&0).unwrap().remove("null");
    r.node_attributes
        .get_mut(&0)
        .unwrap()
        .insert("new".into(), V::Null);
    l.edges.insert(0, (2, 1));
    r.edge_attributes
        .get_mut(&0)
        .unwrap()
        .insert("weight".into(), V::Int(5));
    l.edges.insert(1, (0, 1));
    r.edges.insert(2, (0, 1));
    l.active_nodes.push(3);
    r.active_nodes.push(4);
    let merged = merge_snapshots(&b, &l, &r, 99).unwrap();
    assert_eq!(merged.state_id, 99);
    assert_eq!(merged.active_nodes, vec![0, 1, 2, 3, 4]);
    assert_eq!(merged.edges[&0], (2, 1));
    assert_eq!(merged.edges.len(), 3);
    assert_eq!(merged.node_attributes[&0]["a"], V::Int(2));
    assert!(!merged.node_attributes[&0].contains_key("null"));
    assert_eq!(merged.node_attributes[&0]["new"], V::Null);
    assert_eq!(merged.edge_attributes[&0]["weight"], V::Int(5));
    assert_eq!(b.edges[&0], (0, 1));
}
#[test]
fn typed_conflicts_keep_absence_null_delete_modify_endpoint_and_creation_evidence() {
    let b = fixture();
    let mut l = b.clone();
    let mut r = b.clone();
    l.node_attributes
        .get_mut(&0)
        .unwrap()
        .insert("a".into(), V::Null);
    r.node_attributes.get_mut(&0).unwrap().remove("a");
    l.edges.insert(0, (2, 1));
    r.edges.insert(0, (0, 2));
    l.edge_attributes
        .get_mut(&0)
        .unwrap()
        .insert("weight".into(), V::Int(3));
    r.edge_attributes
        .get_mut(&0)
        .unwrap()
        .insert("weight".into(), V::Int(4));
    l.active_nodes.push(3);
    r.active_nodes.push(3);
    l.node_attributes
        .insert(3, [("x".into(), V::Int(1))].into());
    r.node_attributes
        .insert(3, [("x".into(), V::Int(2))].into());
    l.edges.insert(1, (1, 2));
    r.edges.insert(1, (2, 2));
    let plan = plan_snapshot_merge(&b, &l, &r, 4).unwrap();
    assert_eq!(plan.conflicts().len(), 5);
    assert!(matches!(
        &plan.conflicts()[0],
        SnapshotMergeConflict::NodeProperty {
            id: 0,
            base: Some(V::Int(1)),
            left: Some(V::Null),
            right: None,
            ..
        }
    ));
    assert!(matches!(
        &plan.conflicts()[1],
        SnapshotMergeConflict::Node { id: 3, .. }
    ));
    assert!(matches!(
        &plan.conflicts()[2],
        SnapshotMergeConflict::EdgeEndpoints { id: 0, .. }
    ));
    assert!(matches!(
        &plan.conflicts()[3],
        SnapshotMergeConflict::EdgeProperty { id: 0, .. }
    ));
    assert!(matches!(
        &plan.conflicts()[4],
        SnapshotMergeConflict::Edge { id: 1, .. }
    ));
    assert!(plan.into_snapshot().is_err());
    let mut l = b.clone();
    let mut r = b.clone();
    l.edges.clear();
    l.edge_attributes.clear();
    r.edge_attributes
        .get_mut(&0)
        .unwrap()
        .insert("weight".into(), V::Int(2));
    assert!(matches!(
        &plan_snapshot_merge(&b, &l, &r, 4).unwrap().conflicts()[0],
        SnapshotMergeConflict::Edge {
            base: Some(_),
            left: None,
            right: Some(_),
            ..
        }
    ));
}
#[test]
fn deletion_vs_new_incident_edge_reports_topology_instead_of_silent_cascade() {
    let b = fixture();
    let mut l = b.clone();
    let mut r = b.clone();
    l.active_nodes.retain(|&id| id != 2);
    r.edges.insert(1, (0, 2));
    let p = plan_snapshot_merge(&b, &l, &r, 3).unwrap();
    assert!(matches!(
        p.conflicts(),
        [SnapshotMergeConflict::Topology {
            id: 1,
            endpoints: (0, 2)
        }]
    ));
    assert!(p.into_snapshot().is_err());
    let mut r = b.clone();
    r.node_attributes
        .insert(2, [("changed".into(), V::Bool(true))].into());
    assert!(matches!(
        plan_snapshot_merge(&b, &l, &r, 4).unwrap().conflicts(),
        [SnapshotMergeConflict::Node { id: 2, .. }]
    ));
}
#[test]
fn merge_comparison_preserves_float_bits_and_property_types() {
    let mut b = fixture();
    let nan = f64::from_bits(0x7ff8000000000042);
    b.node_attributes
        .get_mut(&0)
        .unwrap()
        .insert("float".into(), V::Float64(nan));
    let mut l = b.clone();
    let mut r = b.clone();
    l.node_attributes
        .get_mut(&0)
        .unwrap()
        .insert("a".into(), V::Float64(1.0));
    r.node_attributes
        .get_mut(&0)
        .unwrap()
        .insert("a".into(), V::Float(1.0));
    let p = plan_snapshot_merge(&b, &l, &r, 4).unwrap();
    assert_eq!(p.conflicts().len(), 1); // same NaN bits are unchanged
    assert!(p.into_snapshot().is_err());
    l = b.clone();
    r = b.clone();
    l.node_attributes
        .get_mut(&0)
        .unwrap()
        .insert("float".into(), V::Float64(-0.0));
    r.node_attributes
        .get_mut(&0)
        .unwrap()
        .insert("float".into(), V::Float64(0.0));
    assert_eq!(
        plan_snapshot_merge(&b, &l, &r, 5)
            .unwrap()
            .conflicts()
            .len(),
        1
    );
    let unchanged = merge_snapshots(&b, &b, &b, 6).unwrap();
    if let V::Float64(value) = unchanged.node_attributes[&0]["float"] {
        assert_eq!(value.to_bits(), nan.to_bits());
    } else {
        panic!("lost float type");
    }
}
#[test]
fn structural_validation_rejects_malformed_inputs_and_allows_empty_legacy_rows() {
    let b = fixture();
    let mut broken = b.clone();
    broken.active_nodes.push(0);
    assert!(validate_snapshot(&broken).is_err());
    broken = b.clone();
    broken.edges.insert(99, (0, 99));
    assert!(validate_snapshot(&broken).is_err());
    broken = b.clone();
    broken
        .node_attributes
        .insert(99, [("v".into(), V::Null)].into());
    assert!(validate_snapshot(&broken).is_err());
    broken = b.clone();
    broken
        .edge_attributes
        .insert(99, [("v".into(), V::Int(1))].into());
    assert!(validate_snapshot(&broken).is_err());
    assert!(plan_snapshot_merge(&b, &broken, &b, 4).is_err());
    broken = b.clone();
    broken.node_attributes.insert(99, HashMap::new());
    validate_snapshot(&broken).unwrap();
    broken = b.clone();
    broken.graph_type = groggy::types::GraphType::Directed;
    assert!(plan_snapshot_merge(&b, &b, &broken, 4).is_err());
}
fn version(h: &mut HistoryForest, parent: Option<u64>) -> u64 {
    h.create_commit(ChangeSet::new(), "test".into(), "test".into(), parent)
        .unwrap()
}
fn merge(h: &mut HistoryForest, parents: Vec<u64>) -> u64 {
    h.create_merge_commit(ChangeSet::new(), "merge".into(), "test".into(), parents)
        .unwrap()
}
#[test]
fn merge_bases_do_not_choose_older_dfs_match_or_arbitrary_criss_cross_base() {
    let mut h = HistoryForest::new();
    let root = version(&mut h, None);
    let a = version(&mut h, Some(root));
    let other = version(&mut h, Some(root));
    let left = merge(&mut h, vec![a, other]);
    let right = merge(&mut h, vec![a, root]); // old stack traversal reached root before a
    assert_eq!(h.merge_bases(left, right).unwrap(), vec![a]);
    assert_eq!(h.find_common_ancestor(left, right).unwrap(), Some(a));
    let x = merge(&mut h, vec![a, other]);
    let y = merge(&mut h, vec![other, a]);
    assert_eq!(h.merge_bases(x, y).unwrap(), vec![a, other]);
    assert!(h.find_common_ancestor(x, y).is_err());
    assert_eq!(h.merge_bases(a, left).unwrap(), vec![a]);
    assert_eq!(h.merge_bases(x, x).unwrap(), vec![x]);
    let unrelated = version(&mut h, None);
    assert_eq!(h.find_common_ancestor(root, unrelated).unwrap(), None);
    assert!(h.find_common_ancestor(999, 999).is_err());
    assert!(h.merge_bases(0, left).is_err());
    assert!(h.is_ancestor(a, left).unwrap());
    assert!(!h.is_ancestor(left, a).unwrap());
    assert!(h.is_ancestor(999, 999).is_err());
    assert!(h.is_ancestor(0, left).is_err());
}

#[test]
fn caller_resolves_whole_typed_values_and_topology_explicitly_without_mutating_inputs() {
    use groggy::state::SnapshotMergeResolution as R;
    let b = fixture();
    let mut l = b.clone();
    let mut r = b.clone();
    l.node_attributes
        .get_mut(&0)
        .unwrap()
        .insert("a".into(), V::Int(2));
    r.node_attributes
        .get_mut(&0)
        .unwrap()
        .insert("a".into(), V::Int(3));
    l.edges.insert(0, (2, 1));
    r.edges.insert(0, (0, 2));
    let merged = plan_snapshot_merge(&b, &l, &r, 10)
        .unwrap()
        .resolve_with(|conflict| {
            Ok(match conflict {
                SnapshotMergeConflict::NodeProperty { right, .. } => R::Property(right.clone()),
                SnapshotMergeConflict::EdgeEndpoints { left, .. } => R::Endpoints(*left),
                _ => panic!("unexpected conflict"),
            })
        })
        .unwrap();
    assert_eq!(merged.node_attributes[&0]["a"], V::Int(3));
    assert_eq!(merged.edges[&0], (2, 1));
    assert_eq!(b.edges[&0], (0, 1));
    assert!(plan_snapshot_merge(&b, &l, &r, 11)
        .unwrap()
        .resolve_with(|_| Ok(R::DropEdge))
        .is_err());
    assert!(plan_snapshot_merge(&b, &l, &r, 12)
        .unwrap()
        .resolve_with(|_| Err(groggy::GraphError::InvalidInput("caller aborted".into())))
        .is_err());

    let mut l = b.clone();
    let mut r = b.clone();
    l.active_nodes.retain(|&id| id != 2);
    r.edges.insert(1, (0, 2));
    let merged = plan_snapshot_merge(&b, &l, &r, 13)
        .unwrap()
        .resolve_with(|_| Ok(R::DropEdge))
        .unwrap();
    assert!(!merged.edges.contains_key(&1));
    let merged = plan_snapshot_merge(&b, &l, &r, 14)
        .unwrap()
        .resolve_with(|_| {
            Ok(R::EndpointNodes(vec![(
                2,
                [("restored".into(), V::Bool(true))].into(),
            )]))
        })
        .unwrap();
    assert_eq!(merged.edges[&1], (0, 2));
    assert_eq!(merged.node_attributes[&2]["restored"], V::Bool(true));
    assert!(plan_snapshot_merge(&b, &l, &r, 15)
        .unwrap()
        .resolve_with(|_| Ok(R::EndpointNodes(vec![(0, HashMap::new())])))
        .is_err());
    assert!(plan_snapshot_merge(&b, &l, &r, 16)
        .unwrap()
        .resolve_with(|_| Ok(R::EndpointNodes(vec![
            (2, HashMap::new()),
            (2, HashMap::new())
        ])))
        .is_err());
}

#[test]
fn resolved_entity_choices_recompute_dependencies_and_reject_wrong_record_shapes() {
    use groggy::state::{MergeEntity, SnapshotMergeResolution as R};
    let b = fixture();
    let mut l = b.clone();
    let mut r = b.clone();
    l.active_nodes.retain(|&id| id != 2);
    r.node_attributes
        .insert(2, [("changed".into(), V::Int(1))].into());
    r.edges.insert(1, (0, 2));
    let mut callbacks = 0;
    let merged = plan_snapshot_merge(&b, &l, &r, 10)
        .unwrap()
        .resolve_with(|c| {
            callbacks += 1;
            match c {
                SnapshotMergeConflict::Node { right, .. } => Ok(R::Entity(right.clone())),
                _ => panic!("retaining the node must resolve its provisional topology dependency"),
            }
        })
        .unwrap();
    assert_eq!(callbacks, 1);
    assert_eq!(merged.edges[&1], (0, 2));
    let merged = plan_snapshot_merge(&b, &l, &r, 11)
        .unwrap()
        .resolve_with(|c| {
            Ok(match c {
                SnapshotMergeConflict::Node { .. } => R::Entity(None),
                SnapshotMergeConflict::Topology { .. } => R::DropEdge,
                _ => panic!(),
            })
        })
        .unwrap();
    assert!(!merged.active_nodes.contains(&2));
    assert!(!merged.edges.contains_key(&1));
    assert!(plan_snapshot_merge(&b, &l, &r, 12)
        .unwrap()
        .resolve_with(|_| Ok(R::Entity(Some(MergeEntity {
            endpoints: Some((0, 1)),
            properties: HashMap::new()
        }))))
        .is_err());
    l = b.clone();
    r = b.clone();
    l.edges.clear();
    l.edge_attributes.clear();
    r.edges.insert(0, (2, 1));
    assert!(plan_snapshot_merge(&b, &l, &r, 13)
        .unwrap()
        .resolve_with(|_| Ok(R::Entity(Some(MergeEntity {
            endpoints: None,
            properties: HashMap::new()
        }))))
        .is_err());
}
