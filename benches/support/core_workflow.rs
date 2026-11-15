//! One-process vertical fixture: bulk ingest, captured query/traversal, time and recovery.
use groggy::{
    query::{AttributeFilter, NodeFilter, QueryOptions, TraversalDirection},
    AttrValue as V, BatchNodeRef as N, Graph, GraphBatch,
};
use serde_json::json;
use std::{hint::black_box, time::Instant};

pub fn run(
    mut checkpoint: impl FnMut(&'static str),
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let count: usize = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "10000".into())
        .parse()?;
    if count < 1_000 || !count.is_multiple_of(100) {
        return Err("nodes must be >= 1000 and divisible by 100".into());
    }
    let mut graph = Graph::new_directed();
    let start = Instant::now();
    let input = GraphBatch {
        add_nodes: count,
        add_edges: (0..count)
            .map(|i| (N::Added(i), N::Added((i + 1) % count)))
            .collect(),
        node_attrs: [(
            "bucket".into(),
            (0..count)
                .map(|i| (N::Added(i), V::Int((i % 100) as i64)))
                .collect(),
        )]
        .into(),
        effective_at: Some(10),
        ..Default::default()
    };
    let input_seconds = start.elapsed().as_secs_f64();
    checkpoint("input");
    let start = Instant::now();
    graph.apply_batch(input, 0, "ring".into(), "bench".into())?;
    let ingest_seconds = start.elapsed().as_secs_f64();
    checkpoint("ingest");
    let first = graph.current_state_id();
    let start = Instant::now();
    let view = graph.query()?;
    let capture_seconds = start.elapsed().as_secs_f64();
    checkpoint("capture");
    let start = Instant::now();
    let index = view.build_index(&["bucket".into()], &[], false)?;
    let index_build_seconds = start.elapsed().as_secs_f64();
    checkpoint("index");
    let filter = NodeFilter::AttributeFilter {
        name: "bucket".into(),
        filter: AttributeFilter::Equals(V::Int(42)),
    };
    let options = QueryOptions {
        limit: Some(10),
        ..Default::default()
    };
    let expected: Vec<_> = (0..10).map(|i| 42 + 100 * i).collect();
    let start = Instant::now();
    assert_eq!(view.query_nodes(&filter, &options)?, expected);
    let scan_page_seconds = start.elapsed().as_secs_f64();
    let mut indexed_page_seconds = Vec::new();
    let mut neighborhood_seconds = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..100 {
            let rows = index.query_nodes(&filter, &options, false)?;
            assert_eq!(rows, expected);
            black_box(rows);
        }
        indexed_page_seconds.push(start.elapsed().as_secs_f64() / 100.);
        let start = Instant::now();
        for _ in 0..100 {
            let child = view.within_hops(&[0], 3, TraversalDirection::Outgoing)?;
            assert_eq!(child.node_count(), 4);
            assert_eq!(child.edge_count(), 3);
            assert!(child.contains_node(3) && !child.contains_node(4));
            black_box(child);
        }
        neighborhood_seconds.push(start.elapsed().as_secs_f64() / 100.);
    }
    checkpoint("queries");
    let start = Instant::now();
    graph.apply_batch(
        GraphBatch {
            node_attrs: [("bucket".into(), vec![(N::Existing(0), V::Int(-1))])].into(),
            effective_at: Some(20),
            ..Default::default()
        },
        first,
        "correction".into(),
        "bench".into(),
    )?;
    let update_seconds = start.elapsed().as_secs_f64();
    checkpoint("update");
    assert_eq!(view.node_attr(0, &"bucket".into())?, Some(V::Int(0)));
    let start = Instant::now();
    let plan = graph.prepare_read_at(15, None)?;
    let temporal_prepare_seconds = start.elapsed().as_secs_f64();
    checkpoint("temporal_prepare");
    let start = Instant::now();
    let historical = plan.read_view()?;
    let temporal_read_seconds = start.elapsed().as_secs_f64();
    checkpoint("temporal_read");
    assert_eq!(historical.node_count(), count);
    assert_eq!(historical.edge_count(), count);
    assert_eq!(historical.node_attr(0, &"bucket".into())?, Some(V::Int(0)));
    drop(historical);
    drop(index);
    drop(view);
    checkpoint("release_reads");
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("graph.archive");
    let start = Instant::now();
    graph.save_archive(&path)?;
    let archive_save_seconds = start.elapsed().as_secs_f64();
    checkpoint("archive_save");
    let archive_bytes = std::fs::metadata(&path)?.len();
    let head = graph.current_state_id();
    drop(graph);
    checkpoint("release_owner");
    let start = Instant::now();
    let recovered = Graph::load_archive(&path)?;
    let archive_recovery_seconds = start.elapsed().as_secs_f64();
    checkpoint("archive_recovery");
    assert_eq!(recovered.current_state_id(), head);
    assert_eq!(recovered.node_count(), count);
    assert_eq!(recovered.edge_count(), count);
    assert_eq!(
        recovered.get_node_attr(0, &"bucket".into())?,
        Some(V::Int(-1))
    );
    assert_eq!(recovered.edge_endpoints(count - 1)?, (count - 1, 0));
    for id in (1..count).step_by((count / 100).max(1)) {
        assert_eq!(
            recovered.get_node_attr(id, &"bucket".into())?,
            Some(V::Int((id % 100) as i64))
        );
    }
    let recovered_past = recovered.read_view_at(15, None)?;
    assert_eq!(
        recovered_past.node_attr(0, &"bucket".into())?,
        Some(V::Int(0))
    );
    checkpoint("recovered_reads");
    Ok(json!({
        "profile":"release", "nodes":count,"edges":count,"shape":"directed ring",
        "properties":"one repeating i64 bucket column plus built-in entity_type; one explicit correction",
        "versions":recovered.history_records().len(),"input_seconds":input_seconds,
        "ingest_seconds":ingest_seconds,"capture_seconds":capture_seconds,
        "index_build_seconds":index_build_seconds,"scan_page_seconds":scan_page_seconds,
        "indexed_page_seconds":indexed_page_seconds,"neighborhood_seconds":neighborhood_seconds,
        "matching_rows":count/100,"page_rows":10,"neighborhood_hops":3,
        "update_seconds":update_seconds,"temporal_prepare_seconds":temporal_prepare_seconds,
        "temporal_read_seconds":temporal_read_seconds,"archive_save_seconds":archive_save_seconds,
        "archive_bytes":archive_bytes,"archive_recovery_seconds":archive_recovery_seconds,
        "scope":"active owner, explicit transient property index, effective T15 after correction T20; full retained archive recovery, not journal throughput"
    }))
}
