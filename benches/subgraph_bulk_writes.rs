//! Native subgraph bulk-write receipt; fixtures excluded and no SLA claim.
use groggy::{subgraphs::Subgraph, AttrValue, Graph};
use serde_json::json;
use std::{cell::RefCell, collections::HashSet, rc::Rc, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        for workload in ["node_bulk", "edge_bulk"] {
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut graph = Graph::new_publishing();
                graph.insert_nodes(count)?;
                graph.add_edges(&(0..count - 1).map(|id| (id, id + 1)).collect::<Vec<_>>())?;
                let shared = Rc::new(RefCell::new(graph));
                let selection = Subgraph::from_nodes(
                    shared.clone(),
                    (0..count).collect::<HashSet<_>>(),
                    "benchmark".into(),
                )?;
                let name = "value".into();
                let value = AttrValue::Float64(1.0000000001);
                let started = Instant::now();
                match workload {
                    "node_bulk" => selection.set_node_attribute_bulk(&name, value.clone())?,
                    _ => selection.set_edge_attribute_bulk(&name, value.clone())?,
                }
                let seconds = started.elapsed().as_secs_f64();
                let graph = shared.borrow();
                if graph.current_state_id() != 3 || graph.has_uncommitted_changes() {
                    return Err("Unexpected bulk publication count or dirty state".into());
                }
                let actual = if workload == "node_bulk" {
                    graph.get_node_attr(count - 1, &name)?
                } else {
                    graph.get_edge_attr(count - 2, &name)?
                };
                if actual != Some(value) {
                    return Err("Incorrect bulk property result".into());
                }
                samples.push(json!({"seconds":seconds, "state_id":graph.current_state_id(), "assignments":if workload == "node_bulk" {count} else {count-1}}));
            }
            cases.push(json!({"nodes":count,"workload":workload,"samples":samples}));
        }
    }
    let output = std::env::var("GROGGY_SUBGRAPH_WRITE_PROFILE")
        .unwrap_or_else(|_| "subgraph_bulk_writes.json".into());
    std::fs::write(
        &output,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release chain graph bulk property writes; fixture construction excluded. Five observations; verifies one accepted state and a typed result. No before/after speedup, FFI latency, allocation/RSS or production budget claim. Previous edge helper retained one version per edge, so its history semantics differ.",
            "cases":cases
        }))?,
    )?;
    println!("Wrote subgraph bulk-write receipt to {output}");
    Ok(())
}
