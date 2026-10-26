//! Cost of intrinsic JSON syntax checks in the shared graph validation boundary.
use groggy::{AttrValue, Graph};
use serde_json::json;
use std::{collections::HashMap, hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for (rows, body_bytes) in [(1000, 32), (1000, 16384), (1, 1024 * 1024)] {
        let mut graph = Graph::new_manual();
        graph.add_nodes(rows)?;
        let body = format!("{{\"payload\":\"{}\"}}", "x".repeat(body_bytes));
        for kind in ["json", "text", "f64"] {
            let values = (0..rows)
                .map(|id| {
                    (
                        id,
                        match kind {
                            "json" => AttrValue::Json(body.clone()),
                            "text" => AttrValue::Text(body.clone()),
                            _ => AttrValue::Float64(id as f64),
                        },
                    )
                })
                .collect();
            let attrs = HashMap::from([("payload".into(), values)]);
            let mut samples = Vec::new();
            for _ in 0..3 {
                let start = Instant::now();
                for _ in 0..100 {
                    black_box(graph.validate_node_attrs(black_box(&attrs), None))?;
                }
                samples.push(start.elapsed().as_secs_f64() / 100.0);
            }
            cases.push(json!({"rows":rows,"body_bytes_per_json_or_text":body.len(),"kind":kind,"seconds_per_validation_samples":samples}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"json_property_validation","scope":"release; prepared inputs; shared validation only, not full writes/FFI; text is an unchecked-text control, not equivalent JSON semantics; three batches of 100 checks","cases":cases})
        )?
    );
    Ok(())
}
