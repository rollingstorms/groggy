//! Bounded pattern receipts; correctness checked against analytic chain matches.
use groggy::{
    query::{GraphPattern, PatternEdge, PatternNode, PatternOptions, TraversalDirection},
    BatchNodeRef as N, Graph, GraphBatch,
};
use serde_json::json;
use std::{hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 10_000, 100_000] {
        let mut graph = Graph::new_directed();
        graph.apply_batch(
            GraphBatch {
                add_nodes: count,
                add_edges: (1..count).map(|n| (N::Added(n - 1), N::Added(n))).collect(),
                ..Default::default()
            },
            0,
            "chain".into(),
            "bench".into(),
        )?;
        let start = Instant::now();
        let index = graph.build_index(&[], &[], true)?;
        let build = start.elapsed().as_secs_f64();
        let pattern = GraphPattern {
            nodes: ["a", "b", "c"]
                .iter()
                .map(|v| PatternNode {
                    variable: (*v).into(),
                    filter: None,
                })
                .collect(),
            edges: [("ab", "a", "b"), ("bc", "b", "c")]
                .iter()
                .map(|&(name, a, b)| PatternEdge {
                    variable: name.into(),
                    source: a.into(),
                    target: b.into(),
                    direction: TraversalDirection::Outgoing,
                    filter: None,
                })
                .collect(),
        };
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let full = index.match_pattern(&pattern, &PatternOptions::default(), false)?;
            let full_seconds = start.elapsed().as_secs_f64();
            assert_eq!(full.rows().len(), count - 2);
            for (n, row) in full.rows().iter().enumerate() {
                assert_eq!(row.nodes, vec![n, n + 1, n + 2]);
                assert_eq!(row.edges, vec![n, n + 1]);
            }
            let start = Instant::now();
            let page = index.match_pattern(
                &pattern,
                &PatternOptions {
                    limit: Some(100),
                    ..Default::default()
                },
                false,
            )?;
            let page_seconds = start.elapsed().as_secs_f64();
            assert_eq!(page.rows(), &full.rows()[..100]);
            assert!(page.truncated());
            samples.push(
                json!({"full_seconds":full_seconds,"page_100_seconds":page_seconds,
                "full_expansions":full.expansions(),"page_expansions":page.expansions()}),
            );
            black_box((full, page));
        }
        cases.push(json!({"nodes":count,"edges":count-1,"capture_and_index_seconds":build,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"fixture":"directed two-edge chain pattern, no predicates", "profile":"release", "cases":cases})
        )?
    );
    Ok(())
}
