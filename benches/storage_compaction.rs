//! Physical baseline deduplication while retaining every recorded change.
use groggy::{AttrValue, DurableGraph, Graph};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut samples = Vec::new();
    for _ in 0..3 {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("journal");
        let mut source = Graph::new();
        source.add_nodes(1_000)?;
        for _ in 0..10 {
            source.set_node_attr_column(
                "text".into(),
                (0..1_000)
                    .map(|id| (id, AttrValue::Text(format!("{:0256}", id % 10))))
                    .collect(),
            )?;
        }
        let state = source.current_state_id();
        drop(DurableGraph::create_from_graph(&path, &source, None)?);
        drop(source);
        let start = Instant::now();
        let mut store = DurableGraph::open(&path)?;
        let open_before = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let plan = store.prepare_storage_compaction(state)?;
        let capture = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let outcome = plan.execute();
        let execute = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let report = store.finish_storage_checkpoint(outcome)?;
        let install = start.elapsed().as_secs_f64();
        let slots = report.pool_compaction.unwrap();
        assert_eq!(store.graph()?.history_records().len(), 11);
        drop(store);
        let start = Instant::now();
        let restored = DurableGraph::open(&path)?;
        let open_after = start.elapsed().as_secs_f64();
        assert_eq!(restored.graph()?.current_state_id(), state);
        assert_eq!(
            restored
                .graph()?
                .pool()
                .get_node_attribute_column(&"text".into())
                .unwrap()
                .len(),
            10
        );
        samples.push(json!({"capture_seconds":capture,"execute_and_file_verification_seconds":execute,
            "install_seconds":install,"reopen_before_seconds":open_before,"reopen_after_seconds":open_after,
            "file_bytes_before":report.previous_bytes,"file_bytes_after":report.current_bytes,
            "node_slots_before":slots.previous_node_values,"node_slots_after":slots.current_node_values,
            "retained_versions":report.retained_versions}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"storage_compaction","scope":"local release; 1000 nodes, ten bulk text revisions, ten distinct 256-byte strings; full history retained; warm-cache reopen; not isolated RSS/power-loss evidence","samples":samples})
        )?
    );
    Ok(())
}
