//! Current alternative costs; not a historical FFI speedup comparison.
use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, EdgesTable, GraphTable, NodesTable, Table},
    },
    viz::streaming::{DataSource, GraphDataSource},
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
        let edges = EdgesTable::new((0..n - 1).map(|id| (id, id, id + 1)).collect()).with_column(
            "weight".into(),
            BaseArray::from_attr_values(vec![V::SmallInt(2); n - 1]),
        )?;
        let table = GraphTable::new(nodes, edges);
        for operation in [
            "direct_checked_capture",
            "checked_graph_materialization_then_capture",
        ] {
            let mut seconds = Vec::new();
            for _ in 0..3 {
                let start = Instant::now();
                let (source, graph) = if operation == "direct_checked_capture" {
                    (table.to_viz_data_source(false)?, None)
                } else {
                    let graph = table.to_graph()?;
                    (GraphDataSource::new(&graph), Some(graph))
                };
                seconds.push(start.elapsed().as_secs_f64());
                let metadata = source.get_graph_metadata();
                assert_eq!((metadata.node_count, metadata.edge_count), (n, n - 1));
                std::hint::black_box((&source, &graph));
            }
            cases.push(json!({"nodes":n,"edges":n-1,"operation":operation,"seconds":seconds}));
        }
    }
    let path = std::env::var("GROGGY_TABLE_VIZ_PROFILE")
        .unwrap_or_else(|_| "table_visualization.json".into());
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native optimized current alternatives, three observations. Dense chain with two node properties and one edge property. Direct checked table-to-viz capture versus checked table-to-Graph materialization plus GraphDataSource capture. Both produce the same IDs/topology on this dense fixture. Fixture construction, assertions, metadata reads and all output destruction excluded. Graph materialization includes checked publication/history allocation; direct capture creates no Graph or history. Neither benchmark is the old incorrect FFI row loop. No historical speedup, RSS, Python/GIL/FFI budget or server/layout claim.",
            "cases":cases
        }))?,
    )?;
    Ok(())
}
