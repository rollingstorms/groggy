//! Cold, shared and invalidated immutable captures; no hidden long-lived reader.
use groggy::{AttrValue as V, BatchNodeRef as N, Graph, GraphBatch};
use serde_json::json;
use std::{hint::black_box, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 100_000] {
        let mut graph = Graph::new_manual();
        graph.apply_batch(
            GraphBatch {
                add_nodes: count,
                node_attrs: [(
                    "score".into(),
                    (0..count)
                        .map(|i| (N::Added(i), V::Int(i as i64)))
                        .collect(),
                )]
                .into(),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "bench".into(),
        )?;
        let start = Instant::now();
        let first = graph.pin_current_state()?;
        let cold = start.elapsed().as_secs_f64();
        let mut shared = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..10_000 {
                let pin = graph.pin_current_state()?;
                assert!(std::ptr::eq(first.snapshot(), pin.snapshot()));
                black_box(pin);
            }
            shared.push(start.elapsed().as_secs_f64() / 10_000.);
        }
        let start = Instant::now();
        graph.set_node_attr(0, "score".into(), V::Int(-1))?;
        let mutation = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let changed = graph.pin_current_state()?;
        let invalidated = start.elapsed().as_secs_f64();
        assert_eq!(first.node_attr(0, &"score".into()), Some(V::Int(0)));
        assert_eq!(changed.node_attr(0, &"score".into()), Some(V::Int(-1)));
        drop(first);
        drop(changed);
        let start = Instant::now();
        let uncached = graph.pin_current_state()?;
        let after_release = start.elapsed().as_secs_f64();
        black_box(uncached);
        cases.push(
            json!({"nodes":count,"cold_seconds":cold,"shared_seconds":shared,
            "mutation_seconds":mutation,"invalidated_seconds":invalidated,
            "after_last_reader_release_seconds":after_release}),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "profile":"release", "fixture":"one score column, no edges; manual property mutation after accepted bulk fixture",
            "shared_iterations":10_000, "cases":cases
        }))?
    );
    Ok(())
}
