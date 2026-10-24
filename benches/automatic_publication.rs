//! Native publication cost receipt; bulk and scalar workloads have different histories.
use groggy::{AttrValue, Graph};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        for workload in ["bulk_insert", "bulk_set", "scalar_set"] {
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut graph = Graph::new_publishing();
                if workload != "bulk_insert" {
                    graph.insert_nodes(count)?;
                }
                let values: Vec<_> = (0..count)
                    .map(|id| (id, AttrValue::Int(id as i64)))
                    .collect();
                let started = Instant::now();
                match workload {
                    "bulk_insert" => {
                        std::hint::black_box(graph.insert_nodes(count)?);
                    }
                    "bulk_set" => {
                        graph.set_node_attrs(HashMap::from([("value".into(), values)]))?
                    }
                    "scalar_set" => {
                        for (id, value) in values {
                            graph.set_node_attr(id, "value".into(), value)?;
                        }
                    }
                    _ => unreachable!(),
                }
                let seconds = started.elapsed().as_secs_f64();
                let expected = match workload {
                    "bulk_insert" => 1,
                    "bulk_set" => 2,
                    _ => count as u64 + 1,
                };
                if graph.current_state_id() != expected || graph.has_uncommitted_changes() {
                    return Err("Unexpected publication count or dirty state".into());
                }
                samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id(),"nodes":graph.node_count()}));
            }
            cases.push(json!({"workload":workload,"count":count,"samples":samples}));
        }
    }
    let output = std::env::var("GROGGY_PUBLICATION_PROFILE")
        .unwrap_or_else(|_| "automatic_publication.json".into());
    std::fs::write(
        &output,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release operations; fixture construction excluded except bulk_insert. Bulk and scalar history semantics differ; no speedup/SLA claim. Scalar_set includes its per-call name/column preparation; bulk_set includes column-map construction. No FFI or RSS measurement.",
            "cases":cases
        }))?,
    )?;
    println!("Wrote publication receipt to {output}");
    Ok(())
}
