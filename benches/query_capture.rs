//! Explicit capture, scan/page and indexed/page costs over the same immutable state.
use groggy::{
    query::{AttributeFilter, NodeFilter, QueryOptions},
    AttrValue as V, BatchNodeRef as N, Graph, GraphBatch,
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
                node_attrs: [(
                    "score".into(),
                    (0..count)
                        .map(|i| (N::Added(i), V::Int(i as i64)))
                        .collect(),
                )]
                .into(),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "bench".into(),
        )?;
        let head = graph.current_state_id();
        let start = Instant::now();
        let view = graph.query()?;
        let capture = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let index = view.build_index(&["score".into()], &[], false)?;
        let build = start.elapsed().as_secs_f64();
        let filter = NodeFilter::AttributeFilter {
            name: "score".into(),
            filter: AttributeFilter::GreaterThanOrEqual(V::Int(count as i64 - 100)),
        };
        let options = QueryOptions {
            order_by: Some("score".into()),
            descending: true,
            limit: Some(10),
            ..Default::default()
        };
        let expected: Vec<_> = (count - 10..count).rev().collect();
        let mut samples = Vec::new();
        for iteration in 0..5 {
            let mut times = [0.; 2];
            for kind in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                for _ in 0..10 {
                    let rows = if kind == 0 {
                        view.query_nodes(&filter, &options)?
                    } else {
                        index.query_nodes(&filter, &options, false)?
                    };
                    assert_eq!(rows, expected);
                    black_box(rows);
                }
                times[kind] = start.elapsed().as_secs_f64() / 10.;
            }
            samples.push(json!({"scan_page_seconds":times[0],"indexed_page_seconds":times[1]}));
        }
        assert_eq!(graph.current_state_id(), head);
        cases.push(json!({"nodes":count,"matching_rows":100,"page_rows":10,"capture_seconds":capture,"index_build_seconds":build,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"one bulk-created score column; largest hundred values ordered descending with ten-row page","cases":cases})
        )?
    );
    Ok(())
}
