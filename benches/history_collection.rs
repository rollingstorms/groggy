//! Explicit history-only reachability collection, retaining an open cold read plan.
use groggy::{AttrValue, BatchNodeRef, Graph, GraphBatch};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for versions in [1_000, 10_000, 50_000] {
        let mut graph = Graph::new_publishing();
        graph.apply_batch(
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
        graph.create_branch("discard".into())?;
        graph.checkout_branch("discard".into())?;
        for i in 1..=versions {
            graph
                .writer("bench")
                .with_effective_at(Some(i))
                .set_node_attr(0, "v".into(), AttrValue::Int(i))?;
            if i == versions / 2 {
                graph.create_branch("shared".into())?;
            }
        }
        let plan = graph.prepare_read_at(versions, None)?;
        let last = graph.current_state_id();
        graph.checkout_branch("main".into())?;
        let before = graph
            .pool()
            .get_node_attribute_column(&"v".into())
            .map_or(0, |c| c.len());
        let started = Instant::now();
        let removed = graph.prune_branch("discard")?;
        let elapsed = started.elapsed().as_secs_f64();
        let after = graph
            .pool()
            .get_node_attribute_column(&"v".into())
            .map_or(0, |c| c.len());
        assert_eq!(removed, versions as usize / 2);
        assert_eq!(before, after);
        assert!(graph.snapshot_at_commit(last).is_err());
        assert_eq!(
            plan.read_view()?.node_attr(0, &"v".into())?,
            Some(AttrValue::Int(versions))
        );
        assert_eq!(graph.gc_history()?, 0);
        graph
            .writer("bench")
            .set_node_attr(0, "v".into(), AttrValue::Int(-1))?;
        assert!(graph.current_state_id() > last);
        cases.push(json!({"property_versions":versions,"removed_versions":removed,"prune_seconds":elapsed,
            "retained_versions_before_followup":versions/2+1,"column_values_before":before,"column_values_after":after,
            "open_cold_plan_still_readable":true}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"benchmark":"history_collection","cases":cases}))?
    );
    Ok(())
}
