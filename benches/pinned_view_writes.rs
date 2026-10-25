//! Complete immutable selection writeback costs, including prevalidated output capture.
use groggy::{AttrValue as V, BatchEdgeRef as E, BatchNodeRef as N, Graph, GraphBatch};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        for mode in ["properties", "mixed", "cascade"] {
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut graph = Graph::new_publishing();
                graph.insert_nodes(count)?;
                graph.add_edges(&(0..count - 1).map(|id| (id, id + 1)).collect::<Vec<_>>())?;
                graph.set_node_attrs(HashMap::from([
                    (
                        "score".into(),
                        (0..count).map(|id| (id, V::Float64(-1.0))).collect(),
                    ),
                    (
                        "embedding".into(),
                        (0..count)
                            .map(|id| (id, V::Float64Vec(vec![id as f64; 8])))
                            .collect(),
                    ),
                ]))?;
                let view = graph
                    .read_view()?
                    .select_nodes(&(0..count / 2).collect::<Vec<_>>())?;
                let before = graph.current_state_id();
                let batch = match mode {
                    "properties" => GraphBatch {
                        node_attrs: HashMap::from([(
                            "score".into(),
                            (0..count / 2)
                                .map(|id| (N::Existing(id), V::Float64(id as f64 + 0.25)))
                                .collect(),
                        )]),
                        ..Default::default()
                    },
                    "mixed" => GraphBatch {
                        add_nodes: 1,
                        add_edges: vec![(N::Existing(0), N::Added(0))],
                        node_attrs: HashMap::from([(
                            "score".into(),
                            vec![(N::Added(0), V::Float64(0.25))],
                        )]),
                        remove_edges: vec![E::Existing(0)],
                        ..Default::default()
                    },
                    _ => GraphBatch {
                        remove_nodes: (1..count / 2 - 1).map(N::Existing).collect(),
                        ..Default::default()
                    },
                };
                let started = Instant::now();
                let result =
                    graph.apply_view_batch(&view, batch, "benchmark".into(), "benchmark".into())?;
                let seconds = started.elapsed().as_secs_f64();
                if result.batch.state_id != before + 1
                    || result.view.write_state_id() != Some(before + 1)
                    || graph.has_uncommitted_changes()
                {
                    return Err("acceptance mismatch".into());
                }
                let expected_nodes = match mode {
                    "mixed" => count / 2 + 1,
                    "cascade" => 2,
                    _ => count / 2,
                };
                let expected_edges = if mode == "cascade" { 0 } else { count / 2 - 1 };
                if result.view.node_count() != expected_nodes
                    || result.view.edge_count() != expected_edges
                    || view.node_count() != count / 2
                    || view.edge_count() != count / 2 - 1
                {
                    return Err("selected membership mismatch".into());
                }
                let current = graph.read_view()?;
                let a = current.read_state().snapshot();
                let b = result.view.read_state().snapshot();
                if a.active_nodes != b.active_nodes
                    || a.edges != b.edges
                    || a.node_attributes != b.node_attributes
                    || a.edge_attributes != b.edge_attributes
                {
                    return Err("prevalidated output differs from accepted state".into());
                }
                for node in result.view.node_ids() {
                    if mode == "properties"
                        && result.view.node_attr(node, &"score".into())?
                            != Some(V::Float64(node as f64 + 0.25))
                    {
                        return Err("value mismatch".into());
                    }
                }
                samples.push(json!({"seconds":seconds,"state_id":result.batch.state_id,"selected_nodes":result.view.node_count(),"selected_edges":result.view.edge_count()}));
            }
            cases.push(json!({"nodes":count,"mode":mode,"samples":samples}));
        }
    }
    let path = std::env::var("GROGGY_PINNED_WRITE_PROFILE")
        .unwrap_or_else(|_| "pinned_view_writes.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized chain, half-graph immutable selection, f64 scalar and eight-element vectors. Five observations per case. Includes current-state validation/capture, complete scope/publication preparation, output snapshot/selection construction, mutation and publication. Fixture, initial view and exhaustive validation excluded. No matched speedup, allocation/RSS, FFI, durable-fsync or general scale guarantee.","cases":cases}),
        )?,
    )?;
    println!("Wrote pinned view write receipt to {path}");
    Ok(())
}
