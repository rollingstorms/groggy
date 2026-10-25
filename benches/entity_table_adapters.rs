//! Native current-cost benchmark for validated row-aligned entity adapters.
use groggy::{
    storage::table::{EdgesTable, Table},
    AttrValue as V,
};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for n in [1_000, 10_000, 100_000] {
        let edges = EdgesTable::new((0..n).map(|r| (r, r, r + 1)).collect());
        for op in ["tuples", "attribute_column", "source_filter"] {
            let mut seconds = Vec::new();
            for _ in 0..5 {
                let entries = (0..n)
                    .step_by(2)
                    .map(|r| (r, V::Float64(r as f64 + 0.125)))
                    .collect();
                let start = Instant::now();
                match op {
                    "tuples" => {
                        let tuples = edges.as_tuples()?;
                        seconds.push(start.elapsed().as_secs_f64());
                        assert_eq!(tuples.len(), n);
                        assert_eq!(tuples[n - 1], (n - 1, n - 1, n));
                        std::hint::black_box(tuples);
                    }
                    "attribute_column" => {
                        let result = edges.with_attribute_entries("x".into(), entries)?;
                        seconds.push(start.elapsed().as_secs_f64());
                        assert_eq!(result.nrows(), n);
                        assert_eq!(result.get_column("x")?.get(1), Some(&V::Null));
                        std::hint::black_box(result);
                    }
                    _ => {
                        let result = edges.filter_by_sources(&[0, n - 1])?;
                        seconds.push(start.elapsed().as_secs_f64());
                        assert_eq!(result.nrows(), 2);
                        std::hint::black_box(result);
                    }
                }
            }
            cases.push(json!({"rows":n,"operation":op,"seconds":seconds}));
        }
    }
    let path = std::env::var("GROGGY_ENTITY_TABLE_PROFILE")
        .unwrap_or_else(|_| "entity_table_adapters.json".into());
    std::fs::write(
        path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized validated tuple reading, ID-mapped property-column replacement (half IDs supplied), and two-source filtering. Five observations. Fixture/entry construction, result assertions and destruction excluded. Complete validation and output allocation timed. Current costs, no matched speedup, RSS or FFI latency claim.","cases":cases}),
        )?,
    )?;
    Ok(())
}
