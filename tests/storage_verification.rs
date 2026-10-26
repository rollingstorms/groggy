#![cfg(unix)]
use groggy::{AttrValue as V, BatchNodeRef as N, DurableGraph, GraphBatch};
use std::collections::HashMap;
use std::io::{Seek, SeekFrom, Write};
fn populated(path: &std::path::Path) -> DurableGraph {
    let mut store = DurableGraph::create(path, Default::default(), None).unwrap();
    store
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                node_attrs: HashMap::from([("x".into(), vec![(N::Added(0), V::Int(1))])]),
                ..Default::default()
            },
            0,
            "root".into(),
            "test".into(),
        )
        .unwrap();
    store
}
#[test]
fn actual_file_verification_preserves_bytes_cursor_locks_state_and_continued_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal");
    let mut store = populated(&path);
    store
        .set_schema(Some(groggy::GraphSchema::default()), 1, 0)
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let memory = store.check_integrity().unwrap();
    let plan = store.prepare_storage_verification().unwrap();
    assert!(store.graph().is_err());
    let outcome = std::thread::spawn(move || plan.execute()).join().unwrap();
    assert!(DurableGraph::open(&path).is_err());
    let report = store.finish_storage_verification(outcome).unwrap();
    assert_eq!(report.file_bytes, bytes.len() as u64);
    assert_eq!(report.logical_records, 3);
    assert_eq!(report.graph, memory);
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    store
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                ..Default::default()
            },
            1,
            "after".into(),
            "test".into(),
        )
        .unwrap();
    assert_eq!(store.verify_storage().unwrap().graph.active_nodes, 2);
    let state = store.graph().unwrap().current_state_id();
    store.compact_storage(state).unwrap();
    let report = store.verify_storage().unwrap();
    assert_eq!(report.logical_records, 1);
    assert_eq!(report.graph.selected_state, state);
    assert_eq!(report.graph.active_nodes, 2);
}
#[test]
fn incomplete_tail_verification_never_repairs_source_and_requires_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal");
    let mut store = populated(&path);
    let valid = std::fs::read(&path).unwrap();
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"GFRAME01")
        .unwrap();
    let incomplete = std::fs::read(&path).unwrap();
    let error = store.verify_storage().unwrap_err().to_string();
    assert!(error.contains("incomplete trailing bytes"), "{error}");
    assert_eq!(std::fs::read(&path).unwrap(), incomplete);
    assert!(store.graph().is_err());
    drop(store);
    let mut recovered = DurableGraph::open(&path).unwrap();
    assert_eq!(recovered.recovered_tail_bytes(), 8);
    assert_eq!(std::fs::read(&path).unwrap(), valid);
    recovered.verify_storage().unwrap();
}
#[test]
fn checksum_corruption_is_reported_without_truncation_or_write_authority() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal");
    let mut store = populated(&path);
    let mut corrupt = std::fs::read(&path).unwrap();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 1;
    let mut outsider = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    outsider.seek(SeekFrom::Start(last as u64)).unwrap();
    outsider.write_all(&corrupt[last..]).unwrap();
    let error = store.verify_storage().unwrap_err().to_string();
    assert!(error.contains("checksum"), "{error}");
    assert_eq!(std::fs::read(&path).unwrap(), corrupt);
    assert!(store
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                ..Default::default()
            },
            1,
            "reject".into(),
            "test".into()
        )
        .is_err());
    drop(store);
    assert!(DurableGraph::open(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), corrupt);
}
#[test]
fn detached_verification_outcome_keeps_writer_lock_after_owner_drop_and_cannot_finish_elsewhere() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal");
    let other = dir.path().join("other");
    let mut store = populated(&path);
    let plan = store.prepare_storage_verification().unwrap();
    drop(store);
    let outcome = std::thread::spawn(move || plan.execute()).join().unwrap();
    assert!(DurableGraph::open(&path).is_err());
    let mut foreign = populated(&other);
    assert!(foreign.finish_storage_verification(outcome).is_err());
    assert!(foreign.graph().is_err());
    assert!(DurableGraph::open(&path).is_ok());
}
#[test]
fn physical_verification_reuses_chunked_baseline_parser_and_retained_history_checks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal");
    let mut source = groggy::Graph::new();
    source.add_node().unwrap();
    source
        .set_node_attr(0, "large".into(), V::Text("x".repeat(9 * 1024 * 1024)))
        .unwrap();
    let mut store = DurableGraph::create_from_graph(&path, &source, None).unwrap();
    let length = std::fs::metadata(&path).unwrap().len();
    let report = store.verify_storage().unwrap();
    assert_eq!(report.file_bytes, length);
    assert_eq!(report.logical_records, 1);
    assert_eq!(report.graph.retained_versions, 2);
    assert_eq!(report.graph.verified_states, 3);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), length);
}

#[test]
fn changed_named_path_rejects_before_or_after_capture_without_touching_replacement() {
    for after_capture in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal");
        let moved = dir.path().join("moved");
        let mut store = populated(&path);
        let original = std::fs::read(&path).unwrap();
        let plan = if after_capture {
            Some(store.prepare_storage_verification().unwrap())
        } else {
            None
        };
        std::fs::rename(&path, &moved).unwrap();
        std::fs::write(&path, b"unrelated").unwrap();
        let error = if let Some(plan) = plan {
            store
                .finish_storage_verification(plan.execute())
                .unwrap_err()
        } else {
            store.verify_storage().unwrap_err()
        };
        assert!(error.to_string().contains("path changed"));
        assert_eq!(std::fs::read(&path).unwrap(), b"unrelated");
        assert_eq!(std::fs::read(&moved).unwrap(), original);
        assert!(store.graph().is_err());
        drop(store);
        DurableGraph::open(&moved)
            .unwrap()
            .verify_storage()
            .unwrap();
    }
}
