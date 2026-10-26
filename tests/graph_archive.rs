use groggy::{
    AttrValue as V, BatchNodeRef as N, ExternalEntityRef, Graph, GraphBatch, GraphConfig,
};
use std::collections::HashMap;
fn seed(graph: &mut Graph) -> u64 {
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 2,
                add_edges: vec![(N::Added(0), N::Added(1))],
                node_external_ids: vec![(N::Added(0), "original".into())],
                node_attrs: HashMap::from([(
                    "value".into(),
                    vec![(
                        N::Added(0),
                        V::Float64Vec(vec![
                            -0.0,
                            f64::from_bits(0x7ff800000000007b),
                            f64::INFINITY,
                        ]),
                    )],
                )]),
                effective_at: Some(10),
                ..Default::default()
            },
            0,
            "seed".into(),
            "test".into(),
        )
        .unwrap()
        .state_id
}
#[test]
fn archive_round_trip_retains_branches_columns_clocks_inactive_ids_and_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.gga");
    let mut config = GraphConfig::performance_optimized();
    config.graph_type = groggy::types::GraphType::Directed;
    config.query_timeout_ms = 123;
    let mut graph = Graph::with_config_manual(config.clone()).unwrap()
        .into_publishing()
        .unwrap();
    let root = seed(&mut graph);
    let clocks = graph.publication_metadata(root).unwrap();
    graph.checkpoint("named".into(), "test".into()).unwrap();
    graph.create_branch("old".into()).unwrap();
    graph
        .writer("test")
        .set_node_external_id(0, "renamed".into())
        .unwrap();
    graph
        .writer("test")
        .with_effective_at(Some(20))
        .unset_node_attr(0, "value".into())
        .unwrap();
    let inactive = graph.try_add_node().unwrap();
    graph.writer("test").remove_nodes(vec![inactive]).unwrap();
    let tip = graph.current_state_id();
    let count = graph
        .pool()
        .get_node_attribute_column(&"value".into())
        .unwrap()
        .len();
    graph.save_archive(&path).unwrap();
    drop(graph);
    let mut restored = Graph::load_archive(&path).unwrap();
    assert_eq!(restored.current_state_id(), tip);
    assert_eq!(restored.publication_metadata(root).unwrap(), clocks);
    assert_eq!(restored.checkpoints().len(), 1);
    assert_eq!(
        serde_json::to_value(restored.configuration()).unwrap(),
        serde_json::to_value(config).unwrap()
    );
    assert_eq!(
        restored
            .pool()
            .get_node_attribute_column(&"value".into())
            .unwrap()
            .len(),
        count
    );
    assert_eq!(
        restored.allocated_entity_for_external_id("original"),
        Some(ExternalEntityRef::Node(0))
    );
    assert_eq!(
        restored.allocated_entity_for_external_id("renamed"),
        Some(ExternalEntityRef::Node(0))
    );
    assert_eq!(restored.get_node_attr(0, &"value".into()).unwrap(), None);
    restored.checkout_branch("old".into()).unwrap();
    let Some(V::Float64Vec(values)) = restored.get_node_attr(0, &"value".into()).unwrap() else {
        panic!("missing value")
    };
    assert_eq!(
        values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vec![
            (-0.0f64).to_bits(),
            0x7ff800000000007b,
            f64::INFINITY.to_bits()
        ]
    );
    assert_eq!(
        restored
            .pool()
            .get_node_attribute_column(&"value".into())
            .unwrap()
            .len(),
        count
    );
    assert_eq!(
        restored
            .read_view_at(10, None)
            .unwrap()
            .node_attr(0, &"value".into())
            .unwrap(),
        Some(V::Float64Vec(values))
    );
    assert_eq!(restored.try_add_node().unwrap(), inactive + 1);
    assert!(restored.current_state_id() > tip);
}
#[test]
fn archive_empty_and_clean_manual_graphs_restore_their_publication_policy() {
    let dir = tempfile::tempdir().unwrap();
    for publishing in [false, true] {
        let mut graph = if publishing {
            Graph::new_publishing()
        } else {
            Graph::new_manual()
        };
        let path = dir.path().join(format!("empty-{publishing}"));
        graph.save_archive(&path).unwrap();
        let mut copy = Graph::load_archive(&path).unwrap();
        let before = copy.current_state_id();
        assert_eq!(copy.automatic_publication(), publishing);
        copy.insert_nodes(1).unwrap();
        assert_eq!(copy.current_state_id() > before, publishing);
        assert_eq!(copy.has_uncommitted_changes(), !publishing);
        seed(&mut graph);
        let path = dir.path().join(format!("seed-{publishing}"));
        graph.save_archive(&path).unwrap();
        assert_eq!(Graph::load_archive(&path).unwrap().node_count(), 2);
    }
}
#[test]
fn archive_rejects_dirty_graph_overwrite_truncation_checksum_and_trailing_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("archive");
    let mut graph = Graph::new_manual();
    graph.add_node().unwrap();
    assert!(graph.save_archive(&path).is_err());
    assert!(!path.exists());
    let mut graph = Graph::new_publishing();
    seed(&mut graph);
    graph.save_archive(&path).unwrap();
    let original = std::fs::read(&path).unwrap();
    assert!(graph.save_archive(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let corrupt = dir.path().join("bad");
    for end in [0, 7, 31, 32, original.len() - 1] {
        std::fs::write(&corrupt, &original[..end]).unwrap();
        assert!(Graph::load_archive(&corrupt).is_err());
    }
    let mut bytes = original.clone();
    bytes[40] ^= 1;
    std::fs::write(&corrupt, &bytes).unwrap();
    assert!(Graph::load_archive(&corrupt).is_err());
    bytes = original.clone();
    bytes.push(0);
    std::fs::write(&corrupt, &bytes).unwrap();
    assert!(Graph::load_archive(&corrupt).is_err());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
}
#[test]
fn archive_preserves_frozen_nested_sources_after_owner_changes_and_drop() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("groups");
    let mut graph = Graph::new_publishing();
    seed(&mut graph);
    let source = graph
        .retain_group(&graph.read_view().unwrap(), "source".into())
        .unwrap();
    let meta = graph.create_meta_node(source).unwrap();
    let nested = graph
        .retain_group(&graph.read_view().unwrap(), "nested".into())
        .unwrap();
    let outer = graph.create_meta_node(nested).unwrap();
    graph
        .writer("test")
        .set_node_attr(0, "value".into(), V::Text("new".into()))
        .unwrap();
    graph.save_archive(&path).unwrap();
    drop(graph);
    let restored = Graph::load_archive(&path).unwrap();
    let view = restored.read_view().unwrap();
    let nested = view.meta_subgraph(outer).unwrap();
    let original = nested.meta_subgraph(meta).unwrap();
    assert!(matches!(
        original.node_attr(0, &"value".into()).unwrap(),
        Some(V::Float64Vec(_))
    ));
    assert_eq!(
        view.node_attr(0, &"value".into()).unwrap(),
        Some(V::Text("new".into()))
    );
}
#[test]
fn archive_preserves_pruned_version_and_entity_high_water_marks_without_resurrecting_history() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pruned");
    let mut graph = Graph::new_publishing();
    let root = seed(&mut graph);
    graph.create_branch("discard".into()).unwrap();
    graph.checkout_branch("discard".into()).unwrap();
    let id = graph.try_add_node().unwrap();
    let discarded = graph.current_state_id();
    graph.checkout_branch("main".into()).unwrap();
    assert_eq!(graph.prune_branch("discard").unwrap(), 1);
    graph.save_archive(&path).unwrap();
    let mut copy = Graph::load_archive(&path).unwrap();
    assert_eq!(copy.current_state_id(), root);
    assert!(copy.snapshot_at_commit(discarded).is_err());
    assert_eq!(copy.try_add_node().unwrap(), id + 1);
    assert!(copy.current_state_id() > discarded);
}

#[test]
fn archive_preserves_unset_only_edge_column_and_clean_legacy_clock_absence() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = Graph::new_publishing();
    seed(&mut graph);
    graph
        .writer("test")
        .unset_edge_attr(0, "absent".into())
        .unwrap();
    graph
        .writer("test")
        .set_edge_attr(0, "only".into(), V::Null)
        .unwrap();
    graph
        .writer("test")
        .unset_edge_attr(0, "only".into())
        .unwrap();
    let path = dir.path().join("unset");
    graph.save_archive(&path).unwrap();
    let copy = Graph::load_archive(&path).unwrap();
    assert_eq!(copy.get_edge_attr(0, &"only".into()).unwrap(), None);
    assert_eq!(copy.get_edge_attr(0, &"absent".into()).unwrap(), None);
    let mut legacy = Graph::new_manual();
    legacy.add_node().unwrap();
    let state = legacy.commit("legacy".into(), "test".into()).unwrap();
    let path = dir.path().join("legacy");
    legacy.save_archive(&path).unwrap();
    let copy = Graph::load_archive(&path).unwrap();
    assert_eq!(copy.publication_metadata(state).unwrap(), None);
    assert!(!copy.automatic_publication());
}

#[test]
fn archive_keeps_independently_scoped_frozen_source_ids_and_typed_id_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("independent");
    let mut source = Graph::new_publishing();
    source.try_add_nodes(10).unwrap();
    let mut graph = Graph::new_publishing();
    let group = graph
        .retain_group(&source.read_view().unwrap(), "external".into())
        .unwrap();
    let meta = graph.create_meta_node(group).unwrap();
    graph
        .writer("test")
        .set_node_attr(meta, "ids".into(), V::NodeArray(vec![999]))
        .unwrap();
    graph.save_archive(&path).unwrap();
    drop(source);
    let copy = Graph::load_archive(&path).unwrap();
    assert_eq!(copy.node_count(), 1);
    assert_eq!(
        copy.read_view()
            .unwrap()
            .meta_subgraph(meta)
            .unwrap()
            .nodes()
            .len(),
        10
    );
    assert_eq!(
        copy.get_node_attr(meta, &"ids".into()).unwrap(),
        Some(V::NodeArray(vec![999]))
    );
}
