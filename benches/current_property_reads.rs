//! Current scalar, row projection and column costs over wide logical property sets.
use groggy::{AttrValue as V, Graph};
use serde_json::json;
use std::{collections::HashMap, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    let attribute = String::from("v7");
    for count in [1_000, 10_000] {
        let mut graph = Graph::new();
        let nodes = graph.add_nodes(count);
        let edges = graph.add_edges(&nodes.iter().map(|&node| (node, node)).collect::<Vec<_>>())?;
        for node_kind in [false, true] {
            let columns: HashMap<_, _> = (0..32)
                .map(|column| {
                    (
                        format!("v{column}"),
                        (0..count)
                            .map(|id| (id, V::Float64(id as f64 + 0.25)))
                            .collect(),
                    )
                })
                .collect();
            if node_kind {
                graph.set_node_attrs(columns)?;
            } else {
                graph.set_edge_attrs(columns)?;
            }
        }
        for node_kind in [false, true] {
            let ids = if node_kind { &nodes } else { &edges };
            for mode in ["scalar", "projection", "column"] {
                let mut samples = Vec::new();
                for _ in 0..5 {
                    let started = Instant::now();
                    let values = match mode {
                        "scalar" => ids
                            .iter()
                            .map(|&id| {
                                if node_kind {
                                    graph.get_node_attr(id, &attribute)
                                } else {
                                    graph.get_edge_attr(id, &attribute)
                                }
                            })
                            .collect::<groggy::GraphResult<Vec<_>>>()?,
                        "projection" => {
                            let rows = if node_kind {
                                graph.get_node_attrs_bulk(
                                    ids.clone(),
                                    vec!["v7".into(), "absent".into()],
                                )?
                            } else {
                                graph.get_edge_attrs_bulk(
                                    ids.clone(),
                                    vec!["v7".into(), "absent".into()],
                                )?
                            };
                            if rows.len() != count || rows.values().any(|row| row.len() != 1) {
                                return Err("projection shape mismatch".into());
                            }
                            ids.iter()
                                .map(|id| rows.get(id).and_then(|row| row.get("v7")).cloned())
                                .collect()
                        }
                        _ => {
                            if node_kind {
                                graph.get_nodes_attrs(&attribute, ids)?
                            } else {
                                graph.get_edges_attrs(&attribute, ids)?
                            }
                        }
                    };
                    let seconds = started.elapsed().as_secs_f64();
                    for (id, value) in ids.iter().zip(values) {
                        if value != Some(V::Float64(*id as f64 + 0.25)) {
                            return Err("read value mismatch".into());
                        }
                    }
                    samples.push(json!({"seconds":seconds}));
                }
                cases.push(json!({"entities":count,"entity_kind":if node_kind {"node"} else {"edge"},"mode":mode,"columns":32,"samples":samples}));
            }
        }
    }
    let path = std::env::var("GROGGY_PROPERTY_READ_PROFILE")
        .unwrap_or_else(|_| "current_property_reads.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native optimized valid f64 scalar, requested two-key row projection and aligned one-column reads, 32 stored properties per kind. Five observations; fixture setup and final value checks excluded. Projection includes map-shape checks and extraction into ordered values, so modes are distinct workloads. No matched speedup, allocation/RSS or FFI guarantee.","cases":cases
        }))?,
    )?;
    println!("Wrote current property read receipt to {path}");
    Ok(())
}
