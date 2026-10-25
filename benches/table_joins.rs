//! Native table join costs; fixtures and correctness checks are outside timed regions.
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
        let make = |offset: usize| -> Result<BaseTable, groggy::GraphError> {
            BaseTable::from_columns(HashMap::from([
                (
                    "key".into(),
                    BaseArray::from_attr_values(
                        (0..count)
                            .map(|i| V::Int(((i + offset) / 2) as i64))
                            .collect(),
                    ),
                ),
                (
                    "tag".into(),
                    BaseArray::from_attr_values(
                        (0..count)
                            .map(|i| V::Text(format!("tag{}", i % 2)))
                            .collect(),
                    ),
                ),
            ]))
        };
        let left = make(0)?;
        let right = make(count)?;
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let inner = left.inner_join(&right, "key", "key")?;
            let inner_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let outer = left.outer_join(&right, "key", "key")?;
            let outer_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let composite = left.join_on(&right, &["key", "tag"], &["key", "tag"], "outer")?;
            let composite_seconds = start.elapsed().as_secs_f64();
            if inner.nrows() != 0 || outer.nrows() != count * 2 || composite.nrows() != count * 2 {
                return Err("join shape mismatch".into());
            }
            if outer.column("key").and_then(|c| c.get(count)) != Some(&V::Int((count / 2) as i64)) {
                return Err("unmatched outer key lost".into());
            }
            samples.push(json!({"inner_seconds":inner_seconds,"outer_seconds":outer_seconds,"composite_outer_seconds":composite_seconds}));
        }
        cases.push(json!({"rows_per_input":count,"samples":samples}));
    }
    let path =
        std::env::var("GROGGY_TABLE_JOIN_PROFILE").unwrap_or_else(|_| "table_joins.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Five observations of native optimized disjoint-key inner, outer, and composite outer joins. Duplicate keys within each input; fixture construction and output correctness checks excluded. Full result construction included. No matched baseline, FFI, memory/RSS or speedup claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote {path}");
    Ok(())
}
