//! Native empty-collection ownership and schema transformation costs.
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
    for n in [1_000, 10_000] {
        let source = BaseTable::with_column_order(
            HashMap::from([
                (
                    "k".into(),
                    BaseArray::from_attr_values((0..n).map(|i| V::Int((i % 100) as i64)).collect()),
                ),
                (
                    "x".into(),
                    BaseArray::from_attr_values(vec![V::Float64(0.25); n]),
                ),
                (
                    "text".into(),
                    BaseArray::from_attr_values(vec![V::Text("sample".into()); n]),
                ),
            ]),
            vec!["k".into(), "x".into(), "text".into()],
        )?;
        let groups = source.groupby(&["k"])?;
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let empty = groups.take(0);
            let flat = empty.concat()?;
            let take_empty_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let projected = empty
                .select(&["x".into()])?
                .rename(&HashMap::from([("x".into(), "score".into())]))?
                .sum()?;
            let projection_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let filtered = groups.filter(|_| false).sum()?;
            let filter_empty_seconds = start.elapsed().as_secs_f64();
            if flat.nrows() != 0
                || flat.column_names() != source.column_names()
                || !projected.has_column("score")
                || projected.has_column("text")
                || filtered.has_column("text")
            {
                return Err("empty collection schema mismatch".into());
            }
            samples.push(json!({"take_empty_seconds":take_empty_seconds,"projection_seconds":projection_seconds,"filter_empty_seconds":filter_empty_seconds}));
        }
        cases.push(json!({"input_rows":n,"groups":100,"samples":samples}));
    }
    let path = std::env::var("GROGGY_COLLECTION_CONTEXT_PROFILE")
        .unwrap_or_else(|_| "collection_context.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized take(0)+flatten, empty projection/rename+sum, and filter-all+sum; three columns, 100 existing groups, five observations per size. Schema/role/source-context preparation and result construction included; source/group fixtures and checks excluded. No matched baseline, RSS/allocation or Python callback/FFI latency claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote {path}");
    Ok(())
}
