//! Native six-policy graph-table merge current costs.
use groggy::{
    storage::{
        graph_table::MergeOptions,
        table::{ConflictResolution as S, EdgesTable, GraphTable, NodesTable, Table},
    },
    AttrValue as V,
};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
fn fixture(n: usize, offset: usize, value: i64) -> Result<GraphTable, groggy::GraphError> {
    let nodes = NodesTable::new((offset..offset + n).collect()).with_attributes(
        "x".into(),
        (offset..offset + n)
            .map(|i| (i, V::Int(value)))
            .collect::<HashMap<_, _>>(),
    )?;
    let edges = EdgesTable::new(
        (0..n - 1)
            .map(|i| (offset + i, offset + i, offset + i + 1))
            .collect(),
    )
    .with_attributes(
        "weight".into(),
        (0..n - 1)
            .map(|i| (offset + i, V::Float64(1.0000000000000002)))
            .collect(),
    )?;
    Ok(GraphTable::new(nodes, edges))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for n in [1_000, 10_000] {
        let a = fixture(n, 0, 1)?;
        let b = fixture(n, 0, 2)?;
        let disjoint = fixture(n, n, 2)?;
        for (name, strategy) in [
            ("fail", S::Fail),
            ("keep_first", S::KeepFirst),
            ("keep_second", S::KeepSecond),
            ("merge_attributes", S::MergeAttributes),
            ("domain_prefix", S::DomainPrefix),
            ("auto_remap", S::AutoRemap),
        ] {
            let mut samples = Vec::new();
            for _ in 0..5 {
                let inputs = vec![
                    a.clone(),
                    if strategy == S::Fail {
                        disjoint.clone()
                    } else {
                        b.clone()
                    },
                ];
                let start = Instant::now();
                let result = GraphTable::merge(inputs, MergeOptions::new(strategy.clone()))?;
                let seconds = start.elapsed().as_secs_f64();
                let multiplier = if matches!(strategy, S::Fail | S::DomainPrefix | S::AutoRemap) {
                    2
                } else {
                    1
                };
                if result.nodes().nrows() != n * multiplier
                    || result.edges().nrows() != (n - 1) * multiplier
                    || result
                        .edges()
                        .base()
                        .column("weight")
                        .and_then(|c| c.get(0))
                        != Some(&V::Float64(1.0000000000000002))
                {
                    return Err("merge output mismatch".into());
                }
                samples.push(seconds);
            }
            cases.push(json!({"nodes_per_input":n,"strategy":name,"seconds":samples}));
        }
    }
    let path = std::env::var("GROGGY_GRAPH_TABLE_MERGE_PROFILE")
        .unwrap_or_else(|_| "graph_table_merge.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized six-policy merge of two graph tables; five observations. Chain topology, node integer and edge precise-f64 properties. Fail uses disjoint IDs; other policies collide. Input preparation/cloning and final checks excluded. Complete merge validation, identity planning, property transfer and output construction timed. No matched speedup, RSS/allocation or FFI latency claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote {path}");
    Ok(())
}
