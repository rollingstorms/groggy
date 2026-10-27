//! Matched immutable-index versus explicit scan ID queries; setup excluded.
use groggy::{
    query::{AttributeFilter, EdgeFilter, NodeFilter},
    AttrValue, BatchNodeRef, Graph, GraphBatch,
};
use serde_json::json;
use std::{hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 100_000] {
        let mut graph = Graph::new();
        graph.apply_batch(
            GraphBatch {
                add_nodes: count,
                add_edges: (1..count)
                    .map(|id| (BatchNodeRef::Added(id - 1), BatchNodeRef::Added(id)))
                    .collect(),
                node_attrs: std::collections::HashMap::from([(
                    "value".into(),
                    (0..count)
                        .map(|id| (BatchNodeRef::Added(id), AttrValue::Int(id as i64)))
                        .collect(),
                )]),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "bench".into(),
        )?;
        let view = graph.read_view()?;
        let scan = view.build_index(&[], &[], false)?;
        let started = Instant::now();
        let indexed = view.build_index(&["value".into()], &[], true)?;
        let build_seconds = started.elapsed().as_secs_f64();
        let equal = NodeFilter::AttributeEquals {
            name: "value".into(),
            value: AttrValue::Int((count - 1) as i64),
        };
        let range = NodeFilter::AttributeFilter {
            name: "value".into(),
            filter: AttributeFilter::GreaterThanOrEqual(AttrValue::Int((count - 100) as i64)),
        };
        let endpoint = EdgeFilter::ConnectsNodes {
            source: count - 2,
            target: count - 1,
        };
        assert_eq!(
            indexed.find_nodes(&equal, false)?,
            scan.find_nodes(&equal, true)?
        );
        assert_eq!(
            indexed.find_nodes(&range, false)?,
            scan.find_nodes(&range, true)?
        );
        assert_eq!(
            indexed.find_edges(&endpoint, false)?,
            scan.find_edges(&endpoint, true)?
        );
        let mut samples = Vec::new();
        let iterations = if count == 1000 { 100 } else { 10 };
        for sample in 0..5 {
            let measure =
                |scan_mode: bool| -> Result<serde_json::Value, Box<dyn std::error::Error>> {
                    let index = if scan_mode { &scan } else { &indexed };
                    let start = Instant::now();
                    for _ in 0..iterations {
                        black_box(index.find_nodes(&equal, scan_mode)?);
                    }
                    let equality = start.elapsed().as_secs_f64() / iterations as f64;
                    let start = Instant::now();
                    for _ in 0..iterations {
                        black_box(index.find_nodes(&range, scan_mode)?);
                    }
                    let range = start.elapsed().as_secs_f64() / iterations as f64;
                    let start = Instant::now();
                    for _ in 0..iterations {
                        black_box(index.find_edges(&endpoint, scan_mode)?);
                    }
                    let endpoint = start.elapsed().as_secs_f64() / iterations as f64;
                    Ok(
                        json!({"equality_seconds":equality,"top_100_range_seconds":range,"endpoint_seconds":endpoint}),
                    )
                };
            let (indexed, scan) = if sample % 2 == 0 {
                (measure(false)?, measure(true)?)
            } else {
                let s = measure(true)?;
                (measure(false)?, s)
            };
            samples.push(json!({"indexed":indexed,"explicit_scan":scan}));
        }
        cases.push(json!({"nodes":count,"edges":count-1,"build_seconds":build_seconds,"iterations":iterations,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"captured_query_indexes","scope":"five alternating-order release samples, same immutable chain and values; sorted ID queries only; snapshot capture and index construction excluded from query timings; initial index build separately reported; no view materialization, RSS, production throughput or billion-node claim","cases":cases})
        )?
    );
    Ok(())
}
