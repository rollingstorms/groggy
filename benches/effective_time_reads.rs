//! Current reconstruction costs, including full replay; no indexed-read claim.
use groggy::{AttrValue, BatchNodeRef, Graph, GraphBatch};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for nodes in [1_000, 10_000, 100_000] {
        let mut graph = Graph::new_publishing();
        graph.apply_batch(
            GraphBatch {
                add_nodes: nodes,
                add_edges: (0..nodes - 1)
                    .map(|id| (BatchNodeRef::Added(id), BatchNodeRef::Added(id + 1)))
                    .collect(),
                effective_at: Some(0),
                ..Default::default()
            },
            0,
            "initial".into(),
            "bench".into(),
        )?;
        for at in 1..=100 {
            graph.apply_batch(
                GraphBatch {
                    effective_at: Some(at),
                    node_attrs: HashMap::from([(
                        "score".into(),
                        vec![(BatchNodeRef::Existing(0), AttrValue::Int(at))],
                    )]),
                    ..Default::default()
                },
                graph.current_state_id(),
                "change".into(),
                "bench".into(),
            )?;
        }
        let mut prepare_seconds = Vec::new();
        let mut replay_seconds = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let plan = graph.prepare_read_at(50, None)?;
            prepare_seconds.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            let view = plan.read_view()?;
            replay_seconds.push(start.elapsed().as_secs_f64());
            assert_eq!(view.node_count(), nodes);
            assert_eq!(view.edge_count(), nodes - 1);
            assert_eq!(
                view.node_attr(0, &"score".into())?,
                Some(AttrValue::Int(50))
            );
            std::hint::black_box(view);
        }
        prepare_seconds.sort_by(f64::total_cmp);
        replay_seconds.sort_by(f64::total_cmp);
        cases.push(json!({"nodes":nodes, "edges":nodes-1, "versions":101, "effective_at":50, "prepare_median_seconds":prepare_seconds[2], "replay_median_seconds":replay_seconds[2], "repetitions":5}));
    }
    let report = json!({"fixture":"chain topology at time zero; 100 scalar corrections on node zero; read at time 50", "build":"release", "limitations":"single-process local timings; full lineage replay and frozen adjacency construction; no indexes, RSS, durable-open, concurrent writers or production-scale guarantees", "cases":cases});
    let output = serde_json::to_string_pretty(&report)?;
    if let Ok(path) = std::env::var("GROGGY_EFFECTIVE_READ_PROFILE") {
        std::fs::write(path, &output)?;
    }
    println!("{output}");
    Ok(())
}
