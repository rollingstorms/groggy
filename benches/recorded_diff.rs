//! Owner capture versus detached logical reconstruction/diff, without Python output.
use groggy::state::change_tracker::ChangeSet;
use groggy::{AttrValue as V, HistoryForest};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for (nodes, edits) in [(100_000, 0), (1, 50_000)] {
        let mut history = HistoryForest::new();
        let mut head = history.create_commit(
            ChangeSet {
                nodes_added: (0..nodes).collect(),
                ..ChangeSet::new()
            },
            "initial".into(),
            "bench".into(),
            None,
        )?;
        let mut old = None;
        for i in 0..edits {
            let new = Some(V::Int(i));
            head = history.create_commit(
                ChangeSet {
                    node_attr_changes: vec![(0, "value".into(), old, new.clone())],
                    ..ChangeSet::new()
                },
                "edit".into(),
                "bench".into(),
                Some(head),
            )?;
            old = new;
        }
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let plan = history.prepare_commit_diff(0, head)?;
            let capture = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let result = plan.execute()?;
            let execute = start.elapsed().as_secs_f64();
            assert_eq!(result.nodes_added.len(), nodes);
            assert_eq!(result.attr_changes.len(), usize::from(edits > 0));
            let start = Instant::now();
            let same = history.prepare_commit_diff(head, head)?;
            let same_capture = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let empty = same.execute()?;
            let same_execute = start.elapsed().as_secs_f64();
            assert!(empty.nodes_added.is_empty() && empty.attr_changes.is_empty());
            samples.push(json!({"owner_capture_seconds":capture,"detached_execute_seconds":execute,"same_state_capture_seconds":same_capture,"same_state_execute_seconds":same_execute}));
        }
        cases.push(json!({"nodes":nodes,"property_versions":edits,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "benchmark":"recorded_diff", "scope":"local release; setup excluded; owner capture includes first-parent traversal, cycle checks and shared Arc handles; detached execute includes both reconstructions and typed diff; no Python marshal, RSS or before/after speed claim", "cases":cases
        }))?
    );
    Ok(())
}
