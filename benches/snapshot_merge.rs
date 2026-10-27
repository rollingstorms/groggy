//! Pure native three-way planning cost. Setup and input copies are excluded.
use groggy::{
    state::{plan_snapshot_merge, GraphSnapshot},
    AttrValue,
};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for nodes in [1_000, 100_000] {
        let mut samples = Vec::new();
        for _ in 0..3 {
            let mut base = GraphSnapshot::empty(1);
            base.active_nodes = (0..nodes).collect();
            base.edges = (1..nodes).map(|id| (id - 1, (id - 1, id))).collect();
            base.node_attributes = (0..nodes)
                .map(|id| (id, [("value".into(), AttrValue::Int(0))].into()))
                .collect();
            let mut left = base.clone();
            let mut right = base.clone();
            for id in 0..nodes {
                let selected = if id % 2 == 0 { &mut left } else { &mut right };
                selected
                    .node_attributes
                    .get_mut(&id)
                    .unwrap()
                    .insert("value".into(), AttrValue::Int(id as i64 + 1));
            }
            let start = Instant::now();
            let plan = plan_snapshot_merge(&base, &left, &right, 4)?;
            let planning = start.elapsed().as_secs_f64();
            assert!(plan.conflicts().is_empty());
            let start = Instant::now();
            let result = plan.into_snapshot()?;
            let finish = start.elapsed().as_secs_f64();
            assert_eq!(result.active_nodes.len(), nodes);
            assert_eq!(result.edges.len(), nodes - 1);
            for id in [0, nodes - 1] {
                assert_eq!(
                    result.node_attributes[&id]["value"],
                    AttrValue::Int(id as i64 + 1)
                );
            }
            samples.push(
                json!({"plan_seconds":planning,"validate_finish_seconds":finish,"conflicts":0}),
            );
        }
        cases.push(json!({"nodes":nodes,"edges":nodes-1,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"snapshot_merge","scope":"three local native release samples; chain and one integer column; alternating independent left/right property changes; setup/input copies excluded; pure logical plan, no graph publication, FFI, durability, RSS or production budget claim","cases":cases})
        )?
    );
    Ok(())
}
