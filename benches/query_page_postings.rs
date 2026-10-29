//! ID-only pages over immutable sorted presence/equality postings.
use groggy::{
    query::{AttributeFilter, NodeFilter, QueryOptions},
    AttrValue as V, BatchNodeRef as N, Graph, GraphBatch,
};
use serde_json::json;
use std::{hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 100_000, 1_000_000] {
        let mut graph = Graph::new();
        graph.apply_batch(
            GraphBatch {
                add_nodes: count,
                node_attrs: [(
                    "bucket".into(),
                    (0..count)
                        .map(|id| (N::Added(id), V::Int((id % 100) as i64)))
                        .collect(),
                )]
                .into(),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "bench".into(),
        )?;
        let index = graph.query()?.build_index(&["bucket".into()], &[], false)?;
        for descending in [false, true] {
            for equality in [false, true] {
                let filter = if equality {
                    NodeFilter::AttributeFilter {
                        name: "bucket".into(),
                        filter: AttributeFilter::Equals(V::Int(42)),
                    }
                } else {
                    NodeFilter::HasAttribute {
                        name: "bucket".into(),
                    }
                };
                let mut expected: Vec<_> = (0..count)
                    .filter(|id| !equality || id % 100 == 42)
                    .collect();
                if descending {
                    expected.reverse();
                }
                let options = QueryOptions {
                    descending,
                    offset: 1,
                    limit: Some(10),
                    ..Default::default()
                };
                let expected: Vec<_> = expected.into_iter().skip(1).take(10).collect();
                let iterations = if equality { 100 } else { 4 };
                let mut samples = Vec::new();
                for _ in 0..5 {
                    let start = Instant::now();
                    for _ in 0..iterations {
                        let rows = index.query_nodes(&filter, &options, false)?;
                        assert_eq!(rows, expected);
                        black_box(rows);
                    }
                    samples.push(start.elapsed().as_secs_f64() / iterations as f64);
                }
                cases.push(json!({"nodes":count,"predicate":if equality {"equals"} else {"has_attribute"},"descending":descending,"offset":1,"limit":10,"iterations":iterations,"seconds_per_query":samples}));
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"one repeating i64 bucket plus built-in entity_type; immutable index explicit; setup excluded; one-row offset with ten-row page","cases":cases})
        )?
    );
    Ok(())
}
