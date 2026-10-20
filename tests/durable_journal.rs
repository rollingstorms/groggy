use groggy::{AttrValue, BatchNodeRef as N, DurableGraph, GraphBatch, GraphConfig, GraphSchema};
use std::collections::HashMap;
fn node(name: &str) -> GraphBatch {
    GraphBatch {
        add_nodes: 1,
        node_external_ids: vec![(N::Added(0), name.into())],
        node_attrs: HashMap::from([(
            "payload".into(),
            vec![(
                N::Added(0),
                AttrValue::FloatVec(vec![-0.0, f32::from_bits(0x7fc0_0001)]),
            )],
        )]),
        effective_at: Some(-10),
        ..Default::default()
    }
}
#[test]
fn binary_journal_recovers_history_identity_values_and_allocation_after_branches() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.ggj");
    let mut config = GraphConfig::new();
    config.query_timeout_ms = 123;
    let mut store = DurableGraph::create(&path, config, None).unwrap();
    let first = store
        .apply_batch(node("ada"), 0, "first".into(), "test".into())
        .unwrap();
    let metadata = store
        .graph()
        .unwrap()
        .publication_metadata(first.state_id)
        .unwrap();
    store.create_branch("other".into(), first.state_id).unwrap();
    store
        .apply_batch(
            node("retained"),
            first.state_id,
            "second".into(),
            "test".into(),
        )
        .unwrap();
    let second = store.graph().unwrap().current_state_id();
    store.checkout_branch("other".into(), second).unwrap();
    let hashes: Vec<_> = store
        .graph()
        .unwrap()
        .history_records()
        .iter()
        .map(|c| (c.id, c.content_hash))
        .collect();
    drop(store);
    let mut recovered = DurableGraph::open(&path).unwrap();
    let graph = recovered.graph().unwrap();
    assert_eq!(
        graph
            .history_records()
            .iter()
            .map(|c| (c.id, c.content_hash))
            .collect::<Vec<_>>(),
        hashes
    );
    assert_eq!(graph.current_state_id(), first.state_id);
    assert_eq!(graph.commit_history().len(), 2);
    assert_eq!(graph.node_by_external_id("ada", None), Some(0));
    assert!(graph.node_by_external_id("retained", None).is_none());
    assert!(graph.allocated_entity_for_external_id("retained").is_some());
    assert_eq!(graph.configuration().query_timeout_ms, 123);
    assert_eq!(
        graph.publication_metadata(first.state_id).unwrap(),
        metadata
    );
    let value = graph
        .get_node_attr(0, &"payload".into())
        .unwrap()
        .unwrap()
        .get_float_vec()
        .unwrap();
    assert_eq!(
        value.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vec![(-0.0f32).to_bits(), 0x7fc0_0001]
    );
    let next = recovered
        .apply_batch(node("new"), first.state_id, "third".into(), "test".into())
        .unwrap();
    assert_eq!(next.added_nodes, vec![2]);
}
#[test]
fn single_writer_lock_and_schema_persist_across_restart() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.ggj");
    let schema =
        GraphSchema::from_json(r#"{"node_labels":{"Entity":{"required":{"type":"text"}}}}"#)
            .unwrap();
    let mut store = DurableGraph::create(&path, GraphConfig::new(), Some(schema)).unwrap();
    assert!(DurableGraph::open(&path).is_err());
    let length = std::fs::metadata(&path).unwrap().len();
    let mut bad = node("unreserved");
    bad.node_attrs.insert(
        "labels".into(),
        vec![(N::Added(0), AttrValue::TextVec(vec!["Entity".into()]))],
    );
    assert!(store
        .apply_batch(bad, 0, "bad".into(), "test".into())
        .is_err());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), length);
    assert!(store
        .graph()
        .unwrap()
        .allocated_entity_for_external_id("unreserved")
        .is_none());
    store.set_schema(None, 0, 0).unwrap();
    drop(store);
    let reopened = DurableGraph::open(&path).unwrap();
    assert_eq!(reopened.schema_revision(), 1);
    assert!(reopened.schema().is_none());
}
#[test]
fn partial_tail_is_recovered_but_complete_corruption_is_not_silently_truncated() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.ggj");
    let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
    store
        .apply_batch(node("ada"), 0, "first".into(), "test".into())
        .unwrap();
    drop(store);
    let valid = std::fs::read(&path).unwrap();
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"GFRAME01\x01")
        .unwrap();
    let reopened = DurableGraph::open(&path).unwrap();
    assert_eq!(reopened.recovered_tail_bytes(), 9);
    assert_eq!(reopened.graph().unwrap().node_count(), 1);
    drop(reopened);
    assert_eq!(std::fs::read(&path).unwrap(), valid);
    let mut corrupt = valid.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 1;
    std::fs::write(&path, &corrupt).unwrap();
    assert!(DurableGraph::open(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), corrupt);
}
#[test]
fn invalid_or_unsupported_values_do_not_append_or_mutate() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.ggj");
    let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
    let before = std::fs::read(&path).unwrap();
    let mut invalid = node("invalid");
    invalid.node_attrs.insert(
        "unsupported".into(),
        vec![(N::Added(0), AttrValue::SubgraphRef(999))],
    );
    assert!(store
        .apply_batch(invalid, 0, "bad".into(), "test".into())
        .is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(store.graph().unwrap().node_count(), 0);
}

#[test]
fn additional_policy_is_checked_before_append_and_replayed_with_accepted_batch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.ggj");
    let stricter =
        GraphSchema::from_json(r#"{"node_labels":{"Entity":{"required":{"type":"text"}}}}"#)
            .unwrap();
    let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
    let mut batch = node("ada");
    batch.node_attrs.insert(
        "labels".into(),
        vec![(N::Added(0), AttrValue::TextVec(vec!["Entity".into()]))],
    );
    let before = std::fs::read(&path).unwrap();
    assert!(store
        .apply_batch_with_schema(
            batch.clone(),
            0,
            "reject".into(),
            "test".into(),
            Some(&stricter)
        )
        .is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    batch.node_attrs.insert(
        "required".into(),
        vec![(N::Added(0), AttrValue::Text("present".into()))],
    );
    let state = store
        .apply_batch_with_schema(batch, 0, "accept".into(), "test".into(), Some(&stricter))
        .unwrap()
        .state_id;
    drop(store);
    let recovered = DurableGraph::open(&path).unwrap();
    assert_eq!(recovered.graph().unwrap().current_state_id(), state);
    assert_eq!(
        recovered.graph().unwrap().node_by_external_id("ada", None),
        Some(0)
    );
    // A call-local policy is audited during replay, without becoming installed policy.
    assert!(recovered.schema().is_none());
}

#[test]
fn inline_value_families_and_transient_entities_remain_replayable() {
    use groggy::BatchEdgeRef as E;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.ggj");
    let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
    let values = vec![
        AttrValue::Null,
        AttrValue::Int(i64::MIN),
        AttrValue::Float(f32::INFINITY),
        AttrValue::Bool(true),
        AttrValue::Text("π".into()),
        AttrValue::FloatVec(vec![]),
        AttrValue::IntVec(vec![i64::MAX]),
        AttrValue::BoolVec(vec![false]),
        AttrValue::TextVec(vec!["text".into()]),
        AttrValue::Bytes(vec![0, 255]),
        AttrValue::NodeArray(vec![0, 1]),
        AttrValue::EdgeArray(vec![0]),
        AttrValue::Json("{\"big\":123456789012345678901234567890}".into()),
    ];
    let mut batch = GraphBatch {
        add_nodes: 2,
        add_edges: vec![(N::Added(0), N::Added(1))],
        ..Default::default()
    };
    for (index, value) in values.iter().enumerate() {
        batch
            .node_attrs
            .insert(format!("v{index}"), vec![(N::Added(0), value.clone())]);
        batch
            .edge_attrs
            .insert(format!("v{index}"), vec![(E::Added(0), value.clone())]);
    }
    let first = store
        .apply_batch(batch, 0, "types".into(), "test".into())
        .unwrap()
        .state_id;
    // Add+remove has a real lifecycle delta, even when its net topology is unchanged.
    let second = store
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                remove_nodes: vec![N::Added(0)],
                add_edges: vec![(N::Existing(0), N::Existing(1))],
                remove_edges: vec![E::Added(0)],
                ..Default::default()
            },
            first,
            "transient".into(),
            "test".into(),
        )
        .unwrap()
        .state_id;
    drop(store);
    let mut recovered = DurableGraph::open(&path).unwrap();
    assert_eq!(recovered.graph().unwrap().current_state_id(), second);
    for (index, value) in values.into_iter().enumerate() {
        assert_eq!(
            recovered
                .graph()
                .unwrap()
                .get_node_attr(0, &format!("v{index}"))
                .unwrap(),
            Some(value.clone())
        );
        assert_eq!(
            recovered
                .graph()
                .unwrap()
                .get_edge_attr(0, &format!("v{index}"))
                .unwrap(),
            Some(value)
        );
    }
    assert_eq!(
        recovered
            .apply_batch(
                GraphBatch {
                    add_nodes: 1,
                    ..Default::default()
                },
                second,
                "next".into(),
                "test".into()
            )
            .unwrap()
            .added_nodes,
        vec![3]
    );
}

#[test]
fn backup_is_independent_complete_and_never_overwrites_existing_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("live.ggj");
    let backup = dir.path().join("backup.ggj");
    let mut live = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
    let first = live
        .apply_batch(node("ada"), 0, "first".into(), "test".into())
        .unwrap()
        .state_id;
    live.create_branch("archive".into(), first).unwrap();
    let second = live
        .apply_batch(node("retained"), first, "second".into(), "test".into())
        .unwrap()
        .state_id;
    live.checkout_branch("archive".into(), second).unwrap();
    let original = std::fs::read(&path).unwrap();
    live.backup(&backup).unwrap();
    assert!(DurableGraph::open(&path).is_err());
    assert_eq!(std::fs::read(&backup).unwrap(), original);
    assert!(live.backup(&backup).is_err());
    assert!(live.backup(&path).is_err());
    assert_eq!(std::fs::read(&backup).unwrap(), original);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let recovered = DurableGraph::open(&backup).unwrap();
    assert_eq!(recovered.graph().unwrap().current_state_id(), first);
    assert_eq!(recovered.graph().unwrap().commit_history().len(), 2);
    assert!(recovered
        .graph()
        .unwrap()
        .allocated_entity_for_external_id("retained")
        .is_some());
    assert_eq!(recovered.graph().unwrap().list_branches().len(), 2);
    // Copying uses a cloned file descriptor; later appends must still target EOF.
    let third = live
        .apply_batch(node("next"), first, "third".into(), "test".into())
        .unwrap()
        .state_id;
    drop(live);
    assert_eq!(
        DurableGraph::open(&path)
            .unwrap()
            .graph()
            .unwrap()
            .current_state_id(),
        third
    );
    assert_eq!(recovered.graph().unwrap().current_state_id(), first);
    assert_eq!(std::fs::read(&backup).unwrap(), original);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
}

#[test]
fn failed_backup_preserves_destination_and_source_and_cleans_temporary_files() {
    use std::io::{Seek, SeekFrom, Write};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("live.ggj");
    let backup = dir.path().join("backup.ggj");
    let mut live = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
    live.apply_batch(node("ada"), 0, "first".into(), "test".into())
        .unwrap();
    let valid = std::fs::read(&path).unwrap();
    let mut incomplete = valid.clone();
    incomplete.extend_from_slice(b"GFRAME01\x01");
    std::fs::write(&path, &incomplete).unwrap();
    assert!(live.backup(&backup).is_err());
    assert!(!backup.exists());
    assert_eq!(std::fs::read(&path).unwrap(), incomplete);
    std::fs::write(&path, valid).unwrap();
    // External complete corruption must be detected before publishing a backup.
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::End(-1)).unwrap();
    file.write_all(&[0xff]).unwrap();
    file.sync_all().unwrap();
    let corrupt = std::fs::read(&path).unwrap();
    assert!(live.backup(&backup).is_err());
    assert!(!backup.exists());
    assert_eq!(std::fs::read(&path).unwrap(), corrupt);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}
