//! Explicit retained-slot deduplication, with prepared inputs and retained memory estimates.
use groggy::{AttrValue, Graph};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for distinct in [10, 10_000] {
        let mut samples = Vec::new();
        for _ in 0..3 {
            let mut graph = Graph::new();
            graph.add_nodes(10_000)?;
            for _ in 0..5 {
                graph.set_node_attr_column(
                    "text".into(),
                    (0..10_000)
                        .map(|id| (id, AttrValue::Text(format!("{:0256}", id % distinct))))
                        .collect(),
                )?;
            }
            let before = graph
                .pool()
                .get_node_attribute_column(&"text".into())
                .unwrap()
                .memory_usage();
            let start = Instant::now();
            let plan = graph.prepare_pool_compaction()?;
            let capture = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let outcome = plan.execute()?;
            let execute = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let report = graph.finish_pool_compaction(outcome)?;
            let install = start.elapsed().as_secs_f64();
            let column = graph.pool();
            let column = column.get_node_attribute_column(&"text".into()).unwrap();
            assert_eq!(column.len(), distinct);
            samples.push(json!({"capture_seconds":capture,"execute_seconds":execute,"install_seconds":install,
                "node_slots_before":report.previous_node_values,"node_slots_after":report.current_node_values,
                "text_column_slots_before":50000,"text_column_slots_after":column.len(),
                "text_column_estimated_bytes_before":before,"text_column_estimated_bytes_after":column.memory_usage()}));
        }
        cases.push(json!({"nodes":10000,"bulk_revisions":5,"distinct_text_values":distinct,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"pool_compaction","scope":"local release; fixture setup excluded; memory_usage is an in-house retained-column estimate, not peak RSS; detached capture copies raw values; history payload remains retained","cases":cases})
        )?
    );
    Ok(())
}
