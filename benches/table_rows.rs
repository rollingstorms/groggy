//! Native grouping and table-combination current costs.
use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, Table},
    },
    AttrValue as V,
};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        let table = BaseTable::with_column_order(
            HashMap::from([
                (
                    "k".into(),
                    BaseArray::from_attr_values(
                        (0..count).map(|i| V::Int((i % 100) as i64)).collect(),
                    ),
                ),
                (
                    "v".into(),
                    BaseArray::from_attr_values(vec![V::Float64(1.); count]),
                ),
            ]),
            vec!["k".into(), "v".into()],
        )?;
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let groups = table.groupby(&["k"])?;
            let group_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let dedup = table.drop_duplicates(None)?;
            let dedup_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let union = table.union(&table)?;
            let union_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let intersection = table.intersect(&table)?;
            let intersection_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let stack = table.stack(&table)?;
            let stack_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let concat = table.concat(&table)?;
            let concat_seconds = start.elapsed().as_secs_f64();
            if groups.len() != 100
                || dedup.nrows() != 100
                || union.nrows() != 100
                || intersection.nrows() != 100
                || stack.nrows() != count * 2
                || concat.nrows() != count
                || concat.column_names().len() != 4
            {
                return Err("row result shape mismatch".into());
            }
            if dedup.column("k").and_then(|c| c.get(99)) != Some(&V::Int(99)) {
                return Err("row order mismatch".into());
            }
            samples.push(json!({"group_seconds":group_seconds,"dedup_seconds":dedup_seconds,"union_seconds":union_seconds,"intersection_seconds":intersection_seconds,"stack_seconds":stack_seconds,"concat_seconds":concat_seconds}));
        }
        cases.push(json!({"input_rows":count,"samples":samples}));
    }
    let path =
        std::env::var("GROGGY_TABLE_ROW_PROFILE").unwrap_or_else(|_| "table_rows.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized grouping, dedup, self-union/intersection, vertical stack and horizontal concat. Two columns, 100 repeated keys, five observations. Full result construction included; fixture construction and correctness checks excluded. No matched baseline, RSS/allocation, FFI or speedup claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote {path}");
    Ok(())
}
