//! Matched historical comparison control and canonical exact differences.
#[path = "support/previous_snapshot_diff.rs"]
mod previous;
use groggy::{state::GraphSnapshot, AttrValue as V};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for (nodes, width) in [(100_000, 0), (10_000, 256)] {
        let mut before = GraphSnapshot::empty(1);
        before.active_nodes = (0..nodes).collect();
        if width > 0 {
            for id in 0..nodes {
                before.node_attributes.insert(
                    id,
                    [("embedding".into(), V::Float64Vec(vec![id as f64; width]))].into(),
                );
            }
        }
        let mut after = before.clone();
        after.state_id = 2;
        if width > 0 {
            after
                .node_attributes
                .get_mut(&0)
                .unwrap()
                .insert("embedding".into(), V::Float64Vec(vec![1.0; width]));
        }
        let mut samples = Vec::new();
        for repeat in 0..5 {
            let (old_seconds, new_seconds) = if repeat % 2 == 0 {
                let t = Instant::now();
                let a = previous::previous_diff(&before, &after);
                let old = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let b = before.try_diff_with(&after)?;
                let new = t.elapsed().as_secs_f64();
                assert_eq!(a.attribute_changes.len(), usize::from(width > 0));
                assert_eq!(b.attribute_changes.len(), a.attribute_changes.len());
                (old, new)
            } else {
                let t = Instant::now();
                let b = before.try_diff_with(&after)?;
                let new = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let a = previous::previous_diff(&before, &after);
                let old = t.elapsed().as_secs_f64();
                assert_eq!(b.attribute_changes.len(), usize::from(width > 0));
                assert_eq!(b.attribute_changes.len(), a.attribute_changes.len());
                (old, new)
            };
            samples.push(json!({"previous_diff_seconds":old_seconds,"canonical_exact_diff_seconds":new_seconds}));
        }
        cases.push(json!({"nodes":nodes,"f64_vector_width":width,"changed_properties":usize::from(width>0),"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"canonical_state_changes","control_commit":"d2abe891","scope":"five alternating-order matched native release samples; setup excluded; finite values where both algorithms agree; previous logical equality is not an exact-history correctness reference; no RSS/FFI/production claim","cases":cases})
        )?
    );
    Ok(())
}
