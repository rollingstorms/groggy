//! Explicit full captures and repeated selected-state restoration allocation bounds.
use groggy::{AttrValue, BatchEdgeRef, BatchNodeRef, Graph, GraphBatch};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
fn lengths(graph: &Graph) -> (usize, usize, usize) {
    let pool = graph.pool();
    (
        pool.get_node_attribute_column(&"entity_type".into())
            .map_or(0, |c| c.len()),
        pool.get_node_attribute_column(&"score".into())
            .map_or(0, |c| c.len()),
        pool.get_edge_attribute_column(&"weight".into())
            .map_or(0, |c| c.len()),
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for nodes in [1_000, 10_000, 100_000] {
        let mut graph = Graph::new_publishing();
        graph.apply_batch(
            GraphBatch {
                add_nodes: nodes,
                add_edges: (0..nodes - 1)
                    .map(|i| (BatchNodeRef::Added(i), BatchNodeRef::Added(i + 1)))
                    .collect(),
                node_attrs: HashMap::from([(
                    "score".into(),
                    (0..nodes)
                        .map(|i| (BatchNodeRef::Added(i), AttrValue::Float64(i as f64)))
                        .collect(),
                )]),
                edge_attrs: HashMap::from([(
                    "weight".into(),
                    (0..nodes - 1)
                        .map(|i| (BatchEdgeRef::Added(i), AttrValue::Float64(1.0)))
                        .collect(),
                )]),
                effective_at: Some(10),
                ..Default::default()
            },
            0,
            "initial".into(),
            "bench".into(),
        )?;
        let first = graph.current_state_id();
        graph.create_branch("before".into())?;
        graph.writer("bench").set_node_attrs(HashMap::from([(
            "score".into(),
            (0..nodes)
                .map(|i| (i, AttrValue::Float64(i as f64 + 1.0)))
                .collect(),
        )]))?;
        let initial_lengths = lengths(&graph);
        let start = Instant::now();
        graph.checkout_branch("before".into())?;
        graph.checkout_branch("main".into())?;
        let cold_pair_seconds = start.elapsed().as_secs_f64();
        let mut warm_pair_seconds = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            graph.checkout_branch("before".into())?;
            graph.checkout_branch("main".into())?;
            warm_pair_seconds.push(start.elapsed().as_secs_f64());
        }
        let after_lengths = lengths(&graph);
        assert_eq!(after_lengths, initial_lengths);
        let start = Instant::now();
        let snapshot = graph.owned_snapshot_at_commit(first)?;
        let capture_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut copy = snapshot.to_graph()?;
        let restore_seconds = start.elapsed().as_secs_f64();
        assert_eq!(copy.node_count(), nodes);
        assert_eq!(copy.edge_count(), nodes - 1);
        assert_eq!(
            copy.get_node_attr(0, &"score".into())?,
            Some(AttrValue::Float64(0.0))
        );
        assert_eq!(copy.try_add_node()?, nodes);
        let column_bytes = graph
            .pool()
            .get_node_attribute_column(&"score".into())
            .map_or(0, |c| c.memory_usage());
        cases.push(json!({"nodes":nodes,"edges":nodes-1,"cold_switch_pair_seconds":cold_pair_seconds,
            "warm_switch_pair_seconds":warm_pair_seconds,"column_lengths_before":initial_lengths,
            "column_lengths_after":after_lengths,"owned_capture_seconds":capture_seconds,
            "owned_restore_seconds":restore_seconds,"score_column_estimated_bytes_with_lookup":column_bytes}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"owned_snapshot_restoration","cases":cases})
        )?
    );
    Ok(())
}
