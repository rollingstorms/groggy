//! Owned graph-table feature conversion costs; fixture construction is excluded.
use groggy::{
    storage::table::{EdgesTable, GraphTable, NodesTable},
    AttrValue as V,
};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for rows in [1_000, 10_000, 100_000] {
        let mut nodes = NodesTable::new((0..rows).collect());
        let mut edges = EdgesTable::new((0..rows).map(|r| (r, r, r)).collect());
        for c in 0..8 {
            let values = (0..rows)
                .map(|r| (r, V::Float64((r * 8 + c) as f64)))
                .collect();
            nodes = nodes.with_attributes(format!("f{c}"), values)?;
            edges = edges.with_attributes(
                format!("f{c}"),
                (0..rows)
                    .map(|r| (r, V::Float64((r * 8 + c) as f64)))
                    .collect(),
            )?;
        }
        let table = GraphTable::new(nodes, edges);
        for domain in ["nodes", "edges"] {
            let mut samples = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                let matrix = if domain == "nodes" {
                    table.to_matrix_f64()?
                } else {
                    table.edges_to_matrix::<f64>()?
                };
                samples.push(start.elapsed().as_secs_f64());
                assert_eq!(matrix.shape(), (rows, 8));
                for c in 0..8 {
                    assert_eq!(matrix.get(rows - 1, c), Some(((rows - 1) * 8 + c) as f64));
                }
                std::hint::black_box(matrix);
            }
            cases.push(json!({"rows":rows,"features":8,"domain":domain,"seconds":samples}));
        }
    }
    let path = std::env::var("GROGGY_GRAPH_TABLE_FEATURES_PROFILE")
        .unwrap_or_else(|_| "graph_table_features.json".into());
    std::fs::write(
        path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized owned f64 feature conversion. Five observations, eight columns; ID validation, column resolution, conversion, allocation and matrix construction timed. Fixtures and output checks excluded. No matched baseline, RSS or FFI claims.","cases":cases}),
        )?,
    )?;
    Ok(())
}
