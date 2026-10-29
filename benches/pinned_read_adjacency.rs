//! Isolate/sparse/connected capture and read costs, runnable per fixture for RSS.
use groggy::{Graph, GraphBatch};
use serde_json::json;
use std::{hint::black_box, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shape = std::env::args().nth(1).unwrap_or_else(|| "isolates".into());
    let count = 100_000usize;
    let mut graph = Graph::new_directed_manual();
    graph.apply_batch(
        GraphBatch {
            add_nodes: count,
            ..Default::default()
        },
        0,
        "nodes".into(),
        "bench".into(),
    )?;
    let edges: Vec<_> = match shape.as_str() {
        "isolates" => Vec::new(),
        "sparse" => (0..1_000).map(|i| (i, i + 1)).collect(),
        "chain" => (0..count - 1).map(|i| (i, i + 1)).collect(),
        "star" => (1..count).map(|i| (0, i)).collect(),
        _ => return Err("unknown fixture".into()),
    };
    graph.add_edges(&edges)?;
    graph.commit("edges".into(), "bench".into())?;
    let mut capture_seconds = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let pin = graph.pin_current_state()?;
        capture_seconds.push(start.elapsed().as_secs_f64());
        assert_eq!(pin.existence().nodes().len(), count);
        assert_eq!(pin.snapshot().edges.len(), edges.len());
        black_box(pin);
    }
    let pin = graph.pin_current_state()?;
    let mut lookup_seconds_per_node = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let mut neighbor_count = 0;
        let mut incident_count = 0;
        for id in 0..count {
            let neighbors = pin.neighbors(black_box(id)).ok_or("missing node")?;
            let neighbor_edges = pin.neighbor_edges(black_box(id)).ok_or("missing node")?;
            assert_eq!(neighbors.len(), neighbor_edges.len());
            neighbor_count += neighbors.len();
            incident_count += pin
                .incident_edges(black_box(id))
                .ok_or("missing node")?
                .len();
        }
        assert_eq!(neighbor_count, edges.len());
        assert_eq!(incident_count, edges.len() * 2);
        lookup_seconds_per_node.push(start.elapsed().as_secs_f64() / count as f64);
    }
    assert!(pin.neighbors(count).is_none());
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "profile":"release", "shape":shape, "nodes":count, "edges":edges.len(),
            "capture_seconds":capture_seconds, "lookup_seconds_per_node":lookup_seconds_per_node,
            "lookup_work":"neighbors, aligned edge IDs and incident IDs for each node, including isolates",
            "capture_work":"cold capture; last reader released between samples; normal built-in node attributes"
        }))?
    );
    Ok(())
}
