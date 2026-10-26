use groggy::{AttrValue as V, BatchNodeRef, DurableGraph, Graph, GraphBatch};
#[test]
fn detached_integrity_scan_covers_branches_groups_values_id_reservations_and_legacy_versions() {
    let mut graph = Graph::new();
    graph.add_nodes(3).unwrap();
    graph.add_edge(0, 1).unwrap();
    graph
        .writer("test")
        .set_node_external_id(0, "original".into())
        .unwrap();
    graph.create_branch("old".into()).unwrap();
    graph
        .set_node_attr(0, "value".into(), V::Float64(-0.0))
        .unwrap();
    let group = graph
        .retain_group(&graph.read_view().unwrap(), "source".into())
        .unwrap();
    let meta = graph.create_meta_node(group).unwrap();
    graph.remove_node(2).unwrap();
    let selected = graph.current_state_id();
    let history = graph.history_records().len();
    let plan = graph.prepare_integrity_check().unwrap();
    graph
        .set_node_attr(meta, "title".into(), V::Text("later".into()))
        .unwrap();
    let report = std::thread::spawn(move || plan.execute())
        .join()
        .unwrap()
        .unwrap();
    assert_eq!(report.selected_state, selected);
    assert_eq!(report.retained_versions, history);
    assert_eq!(report.verified_states, history + 1);
    assert_eq!(report.active_nodes, 3);
    assert_eq!(report.allocated_nodes, 4);
    assert_eq!(report.branches, 2);
    assert_eq!(report.retained_group_references, 1);
    assert_eq!(report.external_id_reservations, 1);
    assert_eq!(report.legacy_versions, 0);
    assert!(!report.current_schema_checked);
    assert!(report.node_values >= 4);
    assert_eq!(report.maximum_active_nodes, 4);
    let mut legacy = Graph::new_manual();
    legacy.add_node().unwrap();
    assert!(legacy.check_integrity().is_err());
    legacy.commit("accept".into(), "test".into()).unwrap();
    assert_eq!(legacy.check_integrity().unwrap().legacy_versions, 1);
}
#[test]
fn durable_integrity_checks_schema_and_keeps_file_versions_and_locks_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal");
    let mut store = DurableGraph::create(&path, Default::default(), None).unwrap();
    store
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                node_external_ids: vec![(BatchNodeRef::Added(0), "node".into())],
                ..Default::default()
            },
            0,
            "root".into(),
            "test".into(),
        )
        .unwrap();
    store
        .set_schema(Some(groggy::GraphSchema::from_json("{}").unwrap()), 1, 0)
        .unwrap();
    let before = std::fs::read(&path).unwrap();
    let report = store.check_integrity().unwrap();
    assert!(report.current_schema_checked);
    assert_eq!(report.selected_state, 1);
    assert_eq!(report.external_id_reservations, 1);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert!(DurableGraph::open(&path).is_err());
}

#[test]
fn entities_created_or_edited_then_deactivated_in_one_version_leave_no_visible_properties() {
    let mut graph = Graph::new();
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                remove_nodes: vec![BatchNodeRef::Added(0)],
                ..Default::default()
            },
            0,
            "ephemeral".into(),
            "test".into(),
        )
        .unwrap();
    assert_eq!(graph.node_count(), 0);
    assert!(graph.check_integrity().is_ok());
    assert!(graph.prepare_archive().is_ok());
    let mut manual = Graph::new_manual();
    manual.add_nodes(2).unwrap();
    let edge = manual.add_edge(0, 1).unwrap();
    manual.set_node_attr(0, "x".into(), V::Int(1)).unwrap();
    manual.set_edge_attr(edge, "w".into(), V::Int(1)).unwrap();
    manual.commit("base".into(), "test".into()).unwrap();
    let old = manual.read_view().unwrap();
    manual.set_node_attr(0, "x".into(), V::Int(2)).unwrap();
    manual.set_edge_attr(edge, "w".into(), V::Int(2)).unwrap();
    manual.remove_node(0).unwrap();
    manual
        .commit("remove edited node".into(), "test".into())
        .unwrap();
    assert!(manual.check_integrity().is_ok());
    assert_eq!(old.node_attr(0, &"x".into()).unwrap(), Some(V::Int(1)));
    let snap = manual
        .snapshot_at_commit(manual.current_state_id())
        .unwrap();
    assert_eq!(snap.node_attr(0, &"x".into()), None);
    assert_eq!(snap.edge_attr(edge, &"w".into()), None);
}
