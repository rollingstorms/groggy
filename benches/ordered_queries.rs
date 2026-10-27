//! Bounded release receipts for ordered pages and captured hop selections.
use groggy::{
    query::{NodeFilter, QueryOptions, TraversalDirection},
    AttrValue, BatchNodeRef as N, Graph, GraphBatch,
};
use serde_json::json;
use std::{collections::HashMap, hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 100_000] {
        let mut graph = Graph::new_directed();
        graph.apply_batch(
            GraphBatch {
                add_nodes: count,
                add_edges: (1..count).map(|n| (N::Added(n - 1), N::Added(n))).collect(),
                node_attrs: HashMap::from([(
                    "value".into(),
                    (0..count)
                        .map(|n| (N::Added(n), AttrValue::Int(((n * 7919) % count) as i64)))
                        .collect(),
                )]),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "bench".into(),
        )?;
        let start = Instant::now();
        let view = graph.read_view()?;
        let capture = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let index = view.build_index(&["value".into()], &[], false)?;
        let build = start.elapsed().as_secs_f64();
        let filter = NodeFilter::HasAttribute {
            name: "value".into(),
        };
        let options = QueryOptions {
            order_by: Some("value".into()),
            descending: true,
            offset: 10,
            limit: Some(100),
        };
        let mut expected: Vec<_> = (0..count).collect();
        expected.sort_unstable_by_key(|&id| std::cmp::Reverse((id * 7919) % count));
        assert_eq!(
            index.query_nodes(&filter, &options, false)?,
            expected[10..110]
        );
        assert_eq!(
            view.within_hops(&[0], 2, TraversalDirection::Outgoing)?
                .node_ids(),
            vec![0, 1, 2]
        );
        assert_eq!(
            view.within_hops(&[0], 100, TraversalDirection::Outgoing)?
                .node_count(),
            101
        );
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let result = index.query_nodes(&filter, &options, false)?;
            let ordered = start.elapsed().as_secs_f64();
            assert_eq!(result, expected[10..110]);
            black_box(result);
            let start = Instant::now();
            for _ in 0..100 {
                black_box(view.within_hops(&[0], 2, TraversalDirection::Outgoing)?);
            }
            let two = start.elapsed().as_secs_f64() / 100.;
            let start = Instant::now();
            for _ in 0..10 {
                black_box(view.within_hops(&[0], 100, TraversalDirection::Outgoing)?);
            }
            let hundred = start.elapsed().as_secs_f64() / 10.;
            samples.push(json!({"ordered_page_seconds":ordered,"two_hop_view_seconds":two,"hundred_hop_view_seconds":hundred}));
        }
        cases.push(json!({"nodes":count,"edges":count-1,"capture_seconds":capture,"property_index_build_seconds":build,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"ordered_pinned_queries","scope":"five local native release samples; fixed shuffled integer property and directed chain; ordered query filters/validates all matches, partitions a page and sorts 100 IDs after offset 10; hop queries include induced selected-edge view construction; snapshot capture/index build reported separately; no RSS, production, top-k optimization or arbitrary graph-shape claim","cases":cases})
        )?
    );
    Ok(())
}
