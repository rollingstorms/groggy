//! Sampling excludes owner construction and the separately measured immutable capture.
use groggy::ownership::OwnerCell;
use groggy::{subgraphs::Subgraph, Graph};
use serde_json::json;
use std::collections::HashSet;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 100_000] {
        let mut graph = Graph::new_manual();
        graph.add_nodes(count)?;
        graph.add_edges(&(1..count).map(|id| (id - 1, id)).collect::<Vec<_>>())?;
        let owner = Arc::new(OwnerCell::new(graph));
        let live = Subgraph::new(
            owner.clone(),
            (0..count).collect(),
            (0..count - 1).collect(),
            "full".into(),
        );
        let started = Instant::now();
        let frozen = live.pin_view()?;
        let capture = started.elapsed().as_secs_f64();
        live.sample(0, Some(11))?; // Exclude first connected scope validation.
        for k in [0, 16, usize::MAX] {
            let expected = frozen.sample(k, Some(11))?;
            let expected_nodes = expected.nodes();
            let expected_edges: HashSet<_> = (0..count - 1)
                .filter(|edge| {
                    expected_nodes.contains(edge) && expected_nodes.contains(&(edge + 1))
                })
                .collect();
            assert_eq!(expected.edges(), &expected_edges);
            let mut connected = Vec::new();
            let mut captured = Vec::new();
            for _ in 0..5 {
                let started = Instant::now();
                let result = live.sample(k, Some(11))?;
                connected.push(started.elapsed().as_secs_f64());
                assert_eq!(result.nodes(), expected_nodes);
                assert_eq!(result.edges(), &expected_edges);
                black_box(result);
                let started = Instant::now();
                let result = frozen.sample(k, Some(11))?;
                captured.push(started.elapsed().as_secs_f64());
                assert_eq!(result.nodes(), expected_nodes);
                assert_eq!(result.edges(), &expected_edges);
                black_box(result);
            }
            cases.push(json!({"nodes":count,"edges":count-1,"requested":k.to_string(),"sampled_nodes":expected.node_count(),"sampled_edges":expected.edge_count(),"immutable_capture_seconds":capture,"warm_connected_seconds":connected,"captured_seconds":captured}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"fixture":"release native chain; 5 samples; fixed local seed; owner setup and first connected validation excluded; immutable capture reported separately; timings checked against analytic chain edges; no before/after speedup claim", "cases":cases})
        )?
    );
    Ok(())
}
