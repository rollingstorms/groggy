//! Matched clone/drop cost for one immutable visualization snapshot.
use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, EdgesTable, GraphTable, NodesTable},
    },
    viz::streaming::DataSource,
    AttrValue as V,
};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for n in [1_000, 10_000, 100_000] {
        let nodes = NodesTable::from_base_table(BaseTable::from_columns(HashMap::from([
            (
                "node_id".into(),
                BaseArray::from_attr_values((0..n).map(|id| V::Int(id as i64)).collect()),
            ),
            (
                "name".into(),
                BaseArray::from_attr_values(
                    (0..n).map(|id| V::Text(format!("node {id}"))).collect(),
                ),
            ),
            (
                "score".into(),
                BaseArray::from_attr_values(
                    (0..n).map(|id| V::Float64(id as f64 + 0.125)).collect(),
                ),
            ),
        ]))?)?;
        let source = GraphTable::new(
            nodes,
            EdgesTable::new((0..n - 1).map(|id| (id, id, id + 1)).collect()),
        )
        .to_viz_data_source(false)?;
        let mut seconds = Vec::new();
        let iterations = 100;
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..iterations {
                let clone = std::hint::black_box(source.clone());
                assert_eq!(clone.total_rows(), n);
                drop(clone);
            }
            seconds.push(start.elapsed().as_secs_f64() / iterations as f64);
        }
        cases.push(json!({"nodes":n,"edges":n-1,"iterations_per_observation":iterations,"clone_drop_seconds_amortized":seconds}));
    }
    let path = std::env::var("GROGGY_VIZ_CLONE_PROFILE")
        .unwrap_or_else(|_| "viz_snapshot_sharing.json".into());
    std::fs::write(
        path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Optimized native snapshot clone/drop, same benchmark source and 100 iterations per observation, five observations. Dense chain with two node properties. Fixture and original capture construction excluded; clone, black_box, O(1) row-count assertion and clone destruction included. No DataSource get_graph_nodes/get_graph_edges payload export, server, FFI, Python, RSS or whole-graph capture measurement. Each clone is dropped before the next. Amortized timing exceeds clock resolution.","cases":cases}),
        )?,
    )?;
    Ok(())
}
