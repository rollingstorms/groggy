//! Explicit physical-file audit cost; fixture setup excluded.
use groggy::{AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch};
use serde_json::json;
use std::collections::HashMap;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut samples = Vec::new();
    for _ in 0..3 {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("journal");
        let mut source = Graph::new();
        source.add_nodes(1_000)?;
        source.set_node_attr_column("v".into(), (0..1000).map(|n| (n, V::Int(0))).collect())?;
        for i in 0..1000 {
            source.set_node_attr(i, "v".into(), V::Int(i as i64 + 1))?;
        }
        let mut store = DurableGraph::create_from_graph(&path, &source, None)?;
        for i in 0..10 {
            let state = store.graph()?.current_state_id();
            store.apply_batch(
                GraphBatch {
                    node_attrs: HashMap::from([("v".into(), vec![(N::Existing(i), V::Int(9000))])]),
                    ..Default::default()
                },
                state,
                "append".into(),
                "bench".into(),
            )?;
        }
        for baseline in [false, true] {
            if baseline {
                let state = store.graph()?.current_state_id();
                store.checkpoint_storage(state)?;
            }
            let start = Instant::now();
            let report = store.verify_storage()?;
            let seconds = start.elapsed().as_secs_f64();
            assert_eq!(report.graph.retained_versions, 1012);
            assert_eq!(report.graph.verified_states, 1013);
            assert_eq!(report.graph.active_nodes, 1000);
            samples.push(json!({"materialized_baseline_only":baseline,"verification_seconds":seconds,
                "file_bytes":report.file_bytes,"logical_records":report.logical_records,"retained_versions":report.graph.retained_versions}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"storage_verification","scope":"local release; 1000 nodes, 1010 scalar property revisions plus initial nodes/column; physical file replay plus full retained audit; warm-cache; setup excluded; no mutation/FFI/RSS claim","samples":samples})
        )?
    );
    Ok(())
}
