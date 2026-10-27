//! Stable-ID endpoint revisions through one mixed checked publication.
use groggy::{BatchNodeRef as N, Graph, GraphBatch};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for edges in [1_000, 100_000] {
        let mut samples = Vec::new();
        for _ in 0..3 {
            let mut graph = Graph::new();
            graph.apply_batch(
                GraphBatch {
                    add_nodes: 3,
                    add_edges: vec![(N::Added(0), N::Added(1)); edges],
                    effective_at: Some(10),
                    ..Default::default()
                },
                0,
                "initial".into(),
                "bench".into(),
            )?;
            let start = Instant::now();
            let plain_index = graph.build_temporal_index()?;
            let plain_index_seconds = start.elapsed().as_secs_f64();
            assert_eq!(
                plain_index.edges_at_commit(graph.current_state_id()).len(),
                edges
            );
            drop(plain_index);
            let pin = graph.read_view()?;
            let rows = (0..edges).map(|id| (id, 2, 1)).collect();
            let start = Instant::now();
            let state = graph
                .writer("bench")
                .with_effective_at(Some(20))
                .set_edge_endpoints_bulk(rows)?;
            let wall = start.elapsed().as_secs_f64();
            assert_eq!(graph.edge_count(), edges);
            assert_eq!(graph.edge_endpoints(edges - 1)?, (2, 1));
            assert_eq!(pin.edge_endpoints(edges - 1)?, (0, 1));
            let start = Instant::now();
            let effective = graph.read_view_at(20, None)?;
            let read = start.elapsed().as_secs_f64();
            assert_eq!(effective.edge_endpoints(edges - 1)?, (2, 1));
            assert_eq!(
                graph
                    .history_records()
                    .iter()
                    .find(|c| c.id == state)
                    .unwrap()
                    .delta
                    .edge_endpoint_changes
                    .len(),
                edges
            );
            let start = Instant::now();
            let revised_index = graph.build_temporal_index()?;
            let revised_index_seconds = start.elapsed().as_secs_f64();
            assert_eq!(revised_index.neighbors_at_commit(2, state).len(), edges);
            samples.push(json!({"checked_mutation_seconds":wall,"effective_read_seconds":read,"edge_count":edges,"recorded_endpoint_changes":edges,"added_edge_ids":0,"static_index_build_seconds":plain_index_seconds,"revised_index_build_seconds":revised_index_seconds}));
        }
        cases.push(json!({"edges":edges,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"endpoint_revisions","scope":"local release; setup, row preparation and old-view capture excluded; one acceptance preserves stable IDs and old immutable topology; checked mutation and effective read measured separately; no FFI/RSS/production claim","cases":cases})
        )?
    );
    Ok(())
}
