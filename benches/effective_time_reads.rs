//! Cold effective index construction, warm prefix replay and empty-prefix reads.
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
        let start = Instant::now();
        let cold = graph.prepare_read_at(50, None)?;
        let cold_prepare = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let cold = cold.read_view()?;
        let cold_replay = start.elapsed().as_secs_f64();
        assert_eq!(cold.node_count(), nodes);
        drop(cold);
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
        let mut empty = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let view = graph.read_view_at(-1, None)?;
            empty.push(start.elapsed().as_secs_f64());
            assert_eq!(view.node_count(), 0);
        }
        empty.sort_by(f64::total_cmp);
        cases.push(json!({"nodes":nodes,"edges":nodes-1,"versions":101,"effective_at":50,"cold_prepare_seconds":cold_prepare,"cold_index_and_replay_seconds":cold_replay,"warm_prepare_median_seconds":prepare_seconds[2],"warm_replay_median_seconds":replay_seconds[2],"warm_empty_prefix_median_seconds":empty[2],"repetitions":5}));
    }
    let report = json!({"fixture":"chain topology at time zero; 100 scalar corrections on node zero; read at time 50", "build":"release", "limitations":"single-process local timings; cold index build versus warm admitted-prefix replay plus frozen adjacency; no read checkpoints, RSS, durable-open, concurrent writers or production-scale guarantees", "cases":cases});
    let output = serde_json::to_string_pretty(&report)?;
    if let Ok(path) = std::env::var("GROGGY_EFFECTIVE_READ_PROFILE") {
        std::fs::write(path, &output)?;
    }
    println!("{output}");
    Ok(())
}
