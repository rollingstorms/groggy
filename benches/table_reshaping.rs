//! Native current costs; fixtures/checks excluded, complete operations timed.
use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, NodesTable, Table},
    },
    AttrValue as V,
};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for n in [1_000, 10_000, 100_000] {
        let names = ["node_id", "g", "k", "x", "y"];
        let data = vec![
            (0..n).map(|r| V::Int(r as i64)).collect(),
            (0..n).map(|r| V::Int((r % 100) as i64)).collect(),
            (0..n).map(|r| V::Int((r % 4) as i64)).collect(),
            vec![V::Float64(1.25); n],
            vec![V::Int(2); n],
        ];
        let table = BaseTable::with_column_order(
            names
                .into_iter()
                .zip(data)
                .map(|(name, v)| (name.to_string(), BaseArray::from_attr_values(v)))
                .collect(),
            names.iter().map(|s| s.to_string()).collect(),
        )?;
        let nodes = NodesTable::from_base_table(table.clone())?;
        for op in ["typed_filter", "pivot", "melt"] {
            let mut times = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                let output = match op {
                    "typed_filter" => nodes.filter_by_attr("g", &V::Int(0))?.into_base(),
                    "pivot" => table.pivot_table(&["g".into()], "k", "x", "sum")?,
                    _ => table.melt(
                        Some(&["node_id".into()]),
                        Some(&["x".into(), "y".into()]),
                        None,
                        None,
                    )?,
                };
                times.push(start.elapsed().as_secs_f64());
                let expected = match op {
                    "typed_filter" => n / 100,
                    "pivot" => 100,
                    _ => n * 2,
                };
                assert_eq!(output.nrows(), expected);
                std::hint::black_box(output);
            }
            cases.push(json!({"rows":n,"operation":op,"seconds":times}));
        }
    }
    let path = std::env::var("GROGGY_TABLE_RESHAPING_PROFILE")
        .unwrap_or_else(|_| "table_reshaping.json".into());
    std::fs::write(
        path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized current cost: typed equality filter, stable pivot (100 groups, 4 keys), two-column melt. Five observations; complete operations including output allocation and validation timed. Fixture construction and checks excluded. No speedup, RSS or FFI latency claim.","cases":cases}),
        )?,
    )?;
    Ok(())
}
