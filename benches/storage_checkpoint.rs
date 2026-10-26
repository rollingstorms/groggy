//! Replay versus explicit materialized baseline, retaining the raw pool after prune.
use groggy::{AttrValue, BatchNodeRef, DurableGraph, GraphBatch, GraphConfig};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for versions in [1_000usize, 10_000] {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("journal");
        let mut store = DurableGraph::create(&path, GraphConfig::new(), None)?;
        store.apply_batch(
            GraphBatch {
                add_nodes: 1,
                node_attrs: HashMap::from([(
                    "v".into(),
                    vec![(BatchNodeRef::Added(0), AttrValue::Int(0))],
                )]),
                effective_at: Some(0),
                ..Default::default()
            },
            0,
            "root".into(),
            "bench".into(),
        )?;
        let root = store.graph()?.current_state_id();
        store.create_branch("discard".into(), root)?;
        store.checkout_branch("discard".into(), root)?;
        for i in 1..=versions {
            let state = store.graph()?.current_state_id();
            store.apply_batch(
                GraphBatch {
                    node_attrs: HashMap::from([(
                        "v".into(),
                        vec![(BatchNodeRef::Existing(0), AttrValue::Int(i as i64))],
                    )]),
                    effective_at: Some(i as i64),
                    ..Default::default()
                },
                state,
                "update".into(),
                "bench".into(),
            )?;
            if i == versions / 2 {
                store.create_branch("shared".into(), store.graph()?.current_state_id())?;
            }
        }
        let head = store.graph()?.current_state_id();
        store.checkout_branch("main".into(), head)?;
        store.prune_branch("discard".into(), root)?;
        drop(store);
        let started = Instant::now();
        let mut store = DurableGraph::open(&path)?;
        let replay = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let report = store.checkpoint_storage(root)?;
        let rewrite = started.elapsed().as_secs_f64();
        drop(store);
        let started = Instant::now();
        let store = DurableGraph::open(&path)?;
        let baseline = started.elapsed().as_secs_f64();
        let count = store
            .graph()?
            .pool()
            .get_node_attribute_column(&"v".into())
            .map_or(0, |c| c.len());
        assert_eq!(count, versions + 1);
        assert_eq!(store.graph()?.history_records().len(), versions / 2 + 1);
        assert_eq!(
            store
                .graph()?
                .snapshot_at_commit(root)?
                .node_attr(0, &"v".into()),
            Some(AttrValue::Int(0))
        );
        cases.push(json!({"original_property_versions":versions,"retained_versions":report.retained_versions,
            "retained_raw_values":count,"previous_bytes":report.previous_bytes,"checkpoint_bytes":report.current_bytes,
            "previous_records":report.previous_records,"replay_open_seconds":replay,"checkpoint_seconds_including_sync_and_verification":rewrite,"baseline_open_seconds":baseline}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"benchmark":"storage_checkpoint","cases":cases}))?
    );
    Ok(())
}
