//! Explicit state restoration phases, separately measured on native columnar input.
use groggy::{AttrValue, BatchNodeRef as N, Graph, GraphBatch};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for nodes in [1_000, 100_000] {
        let mut samples = Vec::new();
        for _ in 0..3 {
            let mut graph = Graph::new();
            let target = graph
                .apply_batch(
                    GraphBatch {
                        add_nodes: nodes,
                        add_edges: (1..nodes).map(|n| (N::Added(n - 1), N::Added(n))).collect(),
                        node_attrs: [(
                            "value".into(),
                            (0..nodes)
                                .map(|n| (N::Added(n), AttrValue::Int(n as i64)))
                                .collect(),
                        )]
                        .into(),
                        ..Default::default()
                    },
                    0,
                    "initial".into(),
                    "bench".into(),
                )?
                .state_id;
            let future = graph
                .apply_batch(
                    GraphBatch {
                        node_attrs: [(
                            "value".into(),
                            (0..nodes)
                                .map(|n| (N::Existing(n), AttrValue::Int(-(n as i64) - 1)))
                                .collect(),
                        )]
                        .into(),
                        ..Default::default()
                    },
                    target,
                    "future".into(),
                    "bench".into(),
                )?
                .state_id;
            let pin = graph.read_view()?;
            let versions = graph.history_records().len();
            let start = Instant::now();
            let plan = graph.prepare_state_restore(target, future)?;
            let capture = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let outcome = plan.execute()?;
            let replay = start.elapsed().as_secs_f64();
            let start = Instant::now();
            graph.finish_state_restore(outcome)?;
            let install = start.elapsed().as_secs_f64();
            assert_eq!(graph.node_count(), nodes);
            assert_eq!(graph.edge_count(), nodes - 1);
            assert_eq!(
                graph.get_node_attr(nodes - 1, &"value".into())?,
                Some(AttrValue::Int(nodes as i64 - 1))
            );
            assert_eq!(
                pin.node_attr(nodes - 1, &"value".into())?,
                Some(AttrValue::Int(-(nodes as i64)))
            );
            assert_eq!(graph.history_records().len(), versions);
            assert!(graph.snapshot_at_commit(future).is_ok());
            assert_eq!(graph.add_node()?, nodes);
            samples.push(json!({"capture_seconds":capture,"reconstruction_seconds":replay,"validation_install_seconds":install,"retained_versions":versions,"new_mutation_versions":0,"reused_entity_ids":0}));
        }
        cases.push(json!({"nodes":nodes,"edges":nodes - 1,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"state_restore","scope":"three local native release samples; setup, batch input construction and immutable future pin excluded; plan capture, detached replay and owner validation/install measured separately; no FFI/RSS/production budget claim","cases":cases})
        )?
    );
    Ok(())
}
