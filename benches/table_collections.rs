//! Current native grouped collection reduction and flatten costs.
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
                    "x".into(),
                    BaseArray::from_attr_values(
                        (0..count).map(|i| V::Int((i % 7) as i64)).collect(),
                    ),
                ),
            ]),
            vec!["k".into(), "x".into()],
        )?;
        let groups = table.groupby(&["k"])?;
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let sum = groups.sum()?;
            let sum_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let mean = groups.mean()?;
            let mean_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let count_result = groups.count()?;
            let count_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let unique = groups.agg(HashMap::from([("x".into(), "nunique".into())]))?;
            let unique_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let flat = groups.concat()?;
            let flatten_seconds = start.elapsed().as_secs_f64();
            if sum.nrows() != 100
                || mean.nrows() != 100
                || unique.nrows() != 100
                || flat.nrows() != count
                || count_result.column("x").and_then(|c| c.get(0))
                    != Some(&V::Int((count / 100) as i64))
            {
                return Err("collection result mismatch".into());
            }
            samples.push(json!({"sum_seconds":sum_seconds,"mean_seconds":mean_seconds,"count_seconds":count_seconds,"unique_seconds":unique_seconds,"flatten_seconds":flatten_seconds}));
        }
        cases.push(json!({"input_rows":count,"groups":100,"samples":samples}));
    }
    let path = std::env::var("GROGGY_TABLE_COLLECTION_PROFILE")
        .unwrap_or_else(|_| "table_collections.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized pregrouped two-column collection: sum, mean, non-null count, explicit unique count and lossless flatten. Five observations per size, 100 groups. Metadata validation and complete output construction timed; fixture/group construction and correctness checks excluded. No matched speedup, RSS/allocation, Python bridge or FFI latency claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote {path}");
    Ok(())
}
