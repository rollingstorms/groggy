//! Scoped traversal cost, including the temporary connected compatibility capture boundary.
use groggy::ownership::OwnerCell;
use groggy::{
    query::{TraversalEngine, TraversalOptions},
    subgraphs::Subgraph,
    traits::SubgraphOperations,
    Graph,
};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 100_000] {
        let mut graph = Graph::new_manual();
        graph.add_nodes(count)?;
        for node in 1..count {
            graph.add_edge(node - 1, node)?;
        }
        let nodes = (0..count).collect();
        let edges = (0..count - 1).collect();
        let owner = Arc::new(OwnerCell::new(graph));
        let connected = Subgraph::new(owner.clone(), nodes, edges, "benchmark".into());
        let start = Instant::now();
        let view = connected.pin_view()?;
        let capture = start.elapsed().as_secs_f64();
        let mut frozen = Vec::new();
        let mut live = Vec::new();
        let mut engine = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            assert_eq!(view.bfs(0, Some(3))?.node_count(), 4);
            frozen.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            assert_eq!(connected.bfs(0, Some(3))?.node_count(), 4);
            live.push(start.elapsed().as_secs_f64());
            let graph = owner.borrow();
            let start = Instant::now();
            let result = TraversalEngine::new().bfs(
                &graph.pool(),
                graph.space(),
                0,
                TraversalOptions {
                    max_depth: Some(3),
                    ..Default::default()
                },
            )?;
            assert_eq!(result.nodes.len(), 4);
            engine.push(start.elapsed().as_secs_f64());
        }
        drop(view);
        let start = Instant::now();
        assert_eq!(connected.bfs(0, Some(3))?.node_count(), 4);
        let after_reader_release = start.elapsed().as_secs_f64();
        cases.push(json!({"nodes":count,"edges":count-1,"initial_capture_seconds":capture,"captured_bfs_seconds":frozen,"connected_bfs_with_payload_retained_seconds":live,"whole_owner_engine_bfs_seconds":engine,"connected_after_last_reader_release_seconds":after_reader_release}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"undirected chain, no properties; three-hop BFS of a full selection; global engine is a performance reference only, invalid as a selected-scope substitute","cases":cases})
        )?
    );
    Ok(())
}
