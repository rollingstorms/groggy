//! Publication cost versus graph size and unrelated property-history depth.
use groggy::{AttrValue, BatchNodeRef as N, Graph, GraphBatch};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000, 100_000] {
        for depth in [0, 1_000] {
            let mut graph = Graph::new_publishing();
            graph.apply_batch(
                GraphBatch {
                    add_nodes: count,
                    effective_at: Some(0),
                    ..Default::default()
                },
                0,
                "initial".into(),
                "bench".into(),
            )?;
            for at in 1..=depth {
                graph.apply_batch(
                    GraphBatch {
                        effective_at: Some(at),
                        node_attrs: HashMap::from([(
                            "score".into(),
                            vec![(N::Existing(0), AttrValue::Int(at))],
                        )]),
                        ..Default::default()
                    },
                    graph.current_state_id(),
                    "property".into(),
                    "bench".into(),
                )?;
            }
            let mut samples = Vec::new();
            for _ in 0..21 {
                let start = Instant::now();
                let result = graph.apply_batch(
                    GraphBatch {
                        add_edges: vec![(N::Existing(0), N::Existing(1))],
                        effective_at: Some(2_000),
                        ..Default::default()
                    },
                    graph.current_state_id(),
                    "edge".into(),
                    "bench".into(),
                )?;
                samples.push(start.elapsed().as_secs_f64());
                assert_eq!(result.added_edges.len(), 1);
            }
            samples.sort_by(f64::total_cmp);
            assert_eq!(graph.read_view_at(2_000, None)?.edge_count(), 21);
            cases.push(json!({"nodes":count,"prior_property_versions":depth,"samples":21,"edge_publication_median_seconds":samples[10]}));
        }
    }
    let report = json!({"build":"release","fixture":"isolated nodes born at time zero; unrelated scalar property versions; publish 21 parallel edges at time 2000", "limits":"local full-operation timings, including publication and validation; no allocator/RSS, large degree, deep lifecycle, contention or production-capacity guarantee", "cases":cases});
    let output = serde_json::to_string_pretty(&report)?;
    if let Ok(path) = std::env::var("GROGGY_EFFECTIVE_TOPOLOGY_PROFILE") {
        std::fs::write(path, &output)?;
    }
    println!("{output}");
    Ok(())
}
