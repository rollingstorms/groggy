//! Creation and validation of immutable retained membership references.
use groggy::{AttrValue, Graph};
use serde_json::json;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for members in [1_000, 10_000, 100_000] {
        for publishing in [false, true] {
            let mut graph = Graph::new_manual();
            graph.insert_nodes(members)?;
            let group = graph.retain_group(&graph.read_view()?, "retained".into())?;
            graph.commit("fixture".into(), "benchmark".into())?;
            graph.set_automatic_publication(publishing)?;
            let mut samples = Vec::new();
            for offset in 0..5 {
                let before = graph.current_state_id();
                let started = Instant::now();
                let node = graph.create_meta_node(group)?;
                let seconds = started.elapsed().as_secs_f64();
                let validate = Instant::now();
                let actual = graph.get_meta_nodes();
                let validation_seconds = validate.elapsed().as_secs_f64();
                if node != members + offset
                    || actual.len() != offset + 1
                    || graph.current_state_id() != before + u64::from(publishing)
                    || graph.get_node_attr(node, &"contains_subgraph".into())?
                        != Some(AttrValue::SubgraphRef(group))
                    || graph.pool().subgraph_count() != 1
                {
                    return Err("meta creation/reference validation mismatch".into());
                }
                samples.push(json!({"seconds":seconds,"validation_seconds":validation_seconds,"node":node,"state_id":graph.current_state_id()}));
            }
            cases.push(json!({"members":members,"publishing":publishing,"samples":samples}));
        }
    }
    let path = std::env::var("GROGGY_META_CREATION_PROFILE")
        .unwrap_or_else(|_| "meta_creation.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native optimized creation of five meta nodes referencing one retained immutable frozen source group. Setup and output checks excluded. Creation includes preparation, storage and optional publication. Separate get_meta_nodes timing scans all active nodes; it is not an O(1) global selection claim. No matched speedup, allocation/RSS, FFI or portable-pool guarantee.","cases":cases
        }))?,
    )?;
    println!("Wrote meta creation receipt to {path}");
    Ok(())
}
