//! Native/durable checked merge costs; setup, input allocation and archive creation excluded.
use groggy::{
    AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch, GraphMergeOptions,
};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for nodes in [1_000, 100_000] {
        for sparse in [false, true] {
            let mut samples = Vec::new();
            for _ in 0..3 {
                let mut g = Graph::new();
                let root = g
                    .apply_batch(
                        GraphBatch {
                            add_nodes: nodes,
                            add_edges: (1..nodes).map(|i| (N::Added(i - 1), N::Added(i))).collect(),
                            node_attrs: [(
                                "value".into(),
                                (0..nodes).map(|id| (N::Added(id), V::Int(0))).collect(),
                            )]
                            .into(),
                            effective_at: Some(10),
                            ..Default::default()
                        },
                        0,
                        "root".into(),
                        "bench".into(),
                    )?
                    .state_id;
                g.create_branch("right".into())?;
                let left_rows = if sparse {
                    vec![(N::Existing(0), V::Int(1))]
                } else {
                    (0..nodes)
                        .filter(|id| id % 2 == 0)
                        .map(|id| (N::Existing(id), V::Int(1)))
                        .collect()
                };
                let left = g
                    .apply_batch(
                        GraphBatch {
                            node_attrs: [("value".into(), left_rows)].into(),
                            effective_at: Some(20),
                            ..Default::default()
                        },
                        root,
                        "left".into(),
                        "bench".into(),
                    )?
                    .state_id;
                g.checkout_branch("right".into())?;
                let right_rows = if sparse {
                    vec![(N::Existing(1), V::Int(2))]
                } else {
                    (0..nodes)
                        .filter(|id| id % 2 == 1)
                        .map(|id| (N::Existing(id), V::Int(2)))
                        .collect()
                };
                let right = g
                    .apply_batch(
                        GraphBatch {
                            node_attrs: [("value".into(), right_rows)].into(),
                            effective_at: Some(30),
                            ..Default::default()
                        },
                        root,
                        "right".into(),
                        "bench".into(),
                    )?
                    .state_id;
                g.checkout_branch("main".into())?;
                let pin = g.read_view()?;
                let dir = tempfile::tempdir()?;
                let path = dir.path().join("graph");
                let mut durable = DurableGraph::create_from_graph(&path, &g, None)?;
                let start = Instant::now();
                let plan = g.prepare_merge_read(right, None, left)?;
                let capture = start.elapsed().as_secs_f64();
                let start = Instant::now();
                let outcome = plan.execute()?.into_outcome()?;
                let reconstruct_resolve = start.elapsed().as_secs_f64();
                let changed = outcome.changes().node_attr_changes.len();
                let start = Instant::now();
                let state = g.finish_merge(
                    outcome,
                    GraphMergeOptions {
                        effective_at: Some(40),
                        ..Default::default()
                    },
                    "merge".into(),
                    "bench".into(),
                )?;
                let publish = start.elapsed().as_secs_f64();
                assert_eq!(g.get_node_attr(1, &"value".into())?, Some(V::Int(2)));
                assert_eq!(pin.node_attr(1, &"value".into())?, Some(V::Int(0)));
                let outcome = durable
                    .graph()?
                    .prepare_merge_read(right, None, left)?
                    .execute()?
                    .into_outcome()?;
                let before = std::fs::metadata(&path)?.len();
                let start = Instant::now();
                let durable_state = durable.finish_merge(
                    outcome,
                    GraphMergeOptions {
                        effective_at: Some(40),
                        ..Default::default()
                    },
                    "merge".into(),
                    "bench".into(),
                )?;
                let durable_publish = start.elapsed().as_secs_f64();
                let bytes = std::fs::metadata(&path)?.len() - before;
                assert_eq!(durable_state, state);
                drop(durable);
                let start = Instant::now();
                let reopened = DurableGraph::open(&path)?;
                let reopen = start.elapsed().as_secs_f64();
                assert_eq!(reopened.graph()?.current_state_id(), state);
                samples.push(json!({"capture_seconds":capture,"reconstruct_resolve_delta_seconds":reconstruct_resolve,"native_validate_publish_seconds":publish,"durable_encode_sync_publish_seconds":durable_publish,"reopen_seconds":reopen,"journal_growth_bytes":bytes,"resolution_property_changes":changed}));
            }
            cases.push(json!({"nodes":nodes,"edges":nodes-1,"workload":if sparse {"one-key"}else{"independent-column"},"samples":samples}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"checked_merge_publication","scope":"three local native release samples; setup/input/archive creation excluded; immutable first-parent view preserved; full native and durable publication measured separately; journal stores only exact resolution delta; no FFI/RSS/production latency claim","cases":cases})
        )?
    );
    Ok(())
}
