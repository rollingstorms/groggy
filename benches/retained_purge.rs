//! Explicit complete purge over orphan owner topology, values, groups and names.
use groggy::{BatchNodeRef as N, DurableGraph, Graph, GraphBatch};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for size in [1_000, 100_000] {
        let mut samples = Vec::new();
        for _ in 0..3 {
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("journal");
            let mut graph = Graph::new();
            graph.add_node()?;
            graph.create_branch("discard".into())?;
            graph.checkout_branch("discard".into())?;
            graph.apply_batch(
                GraphBatch {
                    add_nodes: size,
                    add_edges: (0..size).map(|id| (N::Existing(0), N::Added(id))).collect(),
                    node_external_ids: (0..size.min(1000))
                        .map(|id| (N::Added(id), format!("orphan-{id}")))
                        .collect(),
                    ..Default::default()
                },
                graph.current_state_id(),
                "orphans".into(),
                "bench".into(),
            )?;
            let group = graph.retain_group(&graph.read_view()?, "orphan group".into())?;
            graph.checkout_branch("main".into())?;
            assert_eq!(graph.prune_branch("discard")?, 1);
            let state = graph.current_state_id();
            let mut store = DurableGraph::create_from_graph(&path, &graph, None)?;
            let start = Instant::now();
            let plan = graph.prepare_purge()?;
            let capture = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let outcome = plan.execute()?;
            let execute = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let report = graph.finish_purge(outcome)?;
            let install = start.elapsed().as_secs_f64();
            assert_eq!(report.purged_nodes, size);
            assert_eq!(report.purged_edges, size);
            assert!(report.removed_groups.contains(&group));
            assert_eq!(report.node_ranges, vec![(1, size + 1)]);
            assert_eq!(graph.pool().get_edge_endpoints(0), None);
            let start = Instant::now();
            let durable = store.purge_unreferenced(state)?;
            let durable_seconds = start.elapsed().as_secs_f64();
            assert_eq!(durable.purge.unwrap(), report);
            drop(store);
            let mut reopened = DurableGraph::open(&path)?;
            assert_eq!(reopened.verify_storage()?.graph.purged_nodes, size);
            assert_eq!(reopened.graph()?.history_records().len(), 1);
            samples.push(json!({"capture_seconds":capture,"candidate_seconds":execute,"validate_install_seconds":install,
                "durable_total_seconds":durable_seconds,"file_bytes_before":durable.previous_bytes,"file_bytes_after":durable.current_bytes,
                "purged_nodes":report.purged_nodes,"purged_edges":report.purged_edges,"released_names":report.released_external_ids.len(),
                "removed_group_registrations":report.removed_groups.len(),"node_interval_count":report.node_ranges.len(),"edge_interval_count":report.edge_ranges.len()}));
        }
        cases.push(json!({"orphan_nodes_and_edges":size,"samples":samples}));
    }
    // Measure the concrete lookup trade-off introduced only by explicit purge.
    let mut graph = Graph::new();
    graph.add_nodes(2)?;
    let batch = GraphBatch {
        add_edges: vec![(N::Existing(0), N::Existing(1)); 50_000],
        ..Default::default()
    };
    graph.apply_batch(
        batch.clone(),
        graph.current_state_id(),
        "live".into(),
        "bench".into(),
    )?;
    graph.create_branch("discard".into())?;
    graph.checkout_branch("discard".into())?;
    graph.apply_batch(
        batch,
        graph.current_state_id(),
        "discard".into(),
        "bench".into(),
    )?;
    graph.checkout_branch("main".into())?;
    graph.prune_branch("discard")?;
    let dense = graph.pool().clone();
    graph.purge_unreferenced()?;
    let sparse = graph.pool().clone();
    let mut lookup_samples = Vec::new();
    for _ in 0..5 {
        let mut times = Vec::new();
        for pool in [&dense, &sparse] {
            let start = Instant::now();
            for i in 0..500_000 {
                assert_eq!(
                    std::hint::black_box(pool.get_edge_endpoints(std::hint::black_box(i % 50_000))),
                    Some((0, 1))
                );
            }
            times.push(start.elapsed().as_secs_f64());
        }
        lookup_samples
            .push(json!({"dense_seconds":times[0], "post_purge_sparse_seconds":times[1]}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"retained_purge","scope":"local release; setup/history prune excluded; source copies and full candidate validation included; one retained live node/version; orphan frozen group unregistered; no isolated RSS/FFI/production claim","cases":cases,"raw_endpoint_lookup":{"retained_edges":50_000,"purged_edges":50_000,"lookups_per_sample":500_000,"samples":lookup_samples}})
        )?
    );
    Ok(())
}
