//! Explicit retained integrity with and without an eligible current topology cache.
use groggy::{BatchNodeRef as N, Graph, GraphBatch};
use serde_json::json;
use std::time::Instant;
fn fixture(nodes: usize) -> groggy::GraphResult<Graph> {
    let mut graph = Graph::new();
    let mut edges: Vec<_> = (1..nodes).map(|n| (N::Added(n - 1), N::Added(n))).collect();
    edges.push((N::Added(0), N::Added(0)));
    edges.push((N::Added(0), N::Added(1)));
    graph.apply_batch(
        GraphBatch {
            add_nodes: nodes,
            add_edges: edges,
            ..Default::default()
        },
        0,
        "fixture".into(),
        "bench".into(),
    )?;
    Ok(graph)
}
fn measure(graph: &Graph) -> groggy::GraphResult<serde_json::Value> {
    let start = Instant::now();
    let plan = graph.prepare_integrity_check()?;
    let capture = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let report = plan.execute()?;
    let execute = start.elapsed().as_secs_f64();
    assert_eq!(report.retained_versions, 1);
    Ok(
        json!({"capture_seconds":capture,"detached_verify_seconds":execute,"active_nodes":report.active_nodes,"active_edges":report.active_edges}),
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for nodes in [1000, 100_000] {
        let cold = fixture(nodes)?;
        let warm = fixture(nodes)?;
        warm.neighbors(0)?;
        let mut samples = Vec::new();
        for repeat in 0..5 {
            let (cold, warm) = if repeat % 2 == 0 {
                (measure(&cold)?, measure(&warm)?)
            } else {
                let w = measure(&warm)?;
                let c = measure(&cold)?;
                (c, w)
            };
            assert_eq!(cold["active_edges"], warm["active_edges"]);
            samples.push(json!({"cold":cold,"warm":warm}));
        }
        cases.push(json!({"nodes":nodes,"edges":nodes+1,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"current_topology_integrity","scope":"five alternating-order native release samples; setup and warming excluded; same retained one-version chain/parallel/loop topology; capture versus detached full archive/history/schema-independent verification; warm adds current-cache checks; no ordinary mutation, RSS or production latency claim","cases":cases})
        )?
    );
    Ok(())
}
