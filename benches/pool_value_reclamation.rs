//! Explicit orphan raw-value reclamation after branch/history pruning.
use groggy::{AttrValue, DurableGraph, Graph};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut samples = Vec::new();
    for _ in 0..3 {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("journal");
        let mut graph = Graph::new();
        graph.add_nodes(1_000)?;
        graph.set_node_attr_column(
            "value".into(),
            (0..1_000)
                .map(|id| (id, AttrValue::Text(format!("base-{id:0250}"))))
                .collect(),
        )?;
        graph.create_branch("discard".into())?;
        graph.checkout_branch("discard".into())?;
        for revision in 0..20 {
            graph.set_node_attr_column(
                "value".into(),
                (0..1_000)
                    .map(|id| {
                        (
                            id,
                            AttrValue::Text(format!("orphan-{revision:02}-{id:0250}")),
                        )
                    })
                    .collect(),
            )?;
        }
        let frozen = graph.read_view()?;
        graph.checkout_branch("main".into())?;
        assert_eq!(graph.prune_branch("discard")?, 20);
        let state = graph.current_state_id();
        let before = graph
            .pool()
            .get_node_attribute_column(&"value".into())
            .unwrap()
            .memory_usage();
        let mut store = DurableGraph::create_from_graph(&path, &graph, None)?;
        let start = Instant::now();
        let plan = graph.prepare_pool_compaction_with_reclamation(true)?;
        let capture = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let outcome = plan.execute()?;
        let execute = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let report = graph.finish_pool_compaction(outcome)?;
        let install = start.elapsed().as_secs_f64();
        assert_eq!(report.reclaimed_unreferenced_node_values, 20_000);
        let after = graph
            .pool()
            .get_node_attribute_column(&"value".into())
            .unwrap()
            .memory_usage();
        let start = Instant::now();
        let durable = store.compact_storage_with_reclamation(state, true)?;
        let durable_seconds = start.elapsed().as_secs_f64();
        assert_eq!(
            durable
                .pool_compaction
                .unwrap()
                .reclaimed_unreferenced_node_values,
            20_000
        );
        assert_eq!(
            frozen.node_attr(0, &"value".into())?,
            Some(AttrValue::Text(format!("orphan-19-{:0250}", 0)))
        );
        assert_eq!(graph.history_records().len(), 2);
        drop(store);
        let reopened = DurableGraph::open(&path)?;
        assert_eq!(
            reopened
                .graph()?
                .pool()
                .get_node_attribute_column(&"value".into())
                .unwrap()
                .len(),
            1_000
        );
        samples.push(json!({"capture_seconds":capture,"execute_seconds":execute,"install_seconds":install,
            "raw_node_slots_before":report.previous_node_values,"raw_node_slots_after":report.current_node_values,
            "reclaimed_orphan_node_slots":report.reclaimed_unreferenced_node_values,
            "estimated_value_column_bytes_before":before,"estimated_value_column_bytes_after":after,
            "durable_total_seconds":durable_seconds,"file_bytes_before":durable.previous_bytes,"file_bytes_after":durable.current_bytes}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"pool_value_reclamation","scope":"local release; 1000 nodes, 20 pruned bulk versions, 20000 distinct orphan values; all two retained versions and an open frozen view preserved; setup excluded; column bytes are estimated, not peak RSS","samples":samples})
        )?
    );
    Ok(())
}
