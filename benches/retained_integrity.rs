//! Explicit retained-state verification over linear and interleaved branch histories.
use groggy::{AttrValue, Graph};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for branches in [1, 8] {
        let mut graph = Graph::new();
        graph.add_nodes(1_000)?;
        graph.set_node_attr_column(
            "value".into(),
            (0..1_000).map(|node| (node, AttrValue::Int(0))).collect(),
        )?;
        let names: Vec<_> = (0..branches).map(|b| format!("branch-{b}")).collect();
        for name in &names {
            graph.create_branch(name.clone())?;
        }
        for update in 0..1_000 {
            graph.checkout_branch(names[update % branches].clone())?;
            graph.set_node_attr(
                update % 1_000,
                "value".into(),
                AttrValue::Int(update as i64 + 1),
            )?;
        }
        let mut samples = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            let plan = graph.prepare_integrity_check()?;
            let capture = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let report = plan.execute()?;
            let execute = start.elapsed().as_secs_f64();
            assert_eq!(report.retained_versions, 1002);
            assert_eq!(report.verified_states, 1003);
            assert_eq!(report.maximum_active_nodes, 1000);
            samples
                .push(json!({"capture_seconds":capture,"detached_verification_seconds":execute}));
        }
        cases.push(json!({"nodes":1000,"property_updates":1000,"interleaved_mutation_branches":branches,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"retained_integrity","scope":"release; full retained-state audit, current capture and detached verification; setup excluded; no universal latency/RSS budget","cases":cases})
        )?
    );
    Ok(())
}
