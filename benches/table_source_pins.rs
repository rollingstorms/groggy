//! Explicit frozen-table provenance, selected writeback and read-only join costs.
use groggy::{
    storage::{array::BaseArray, table::Table},
    AttrValue as V, Graph,
};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        let mut samples = Vec::new();
        for _ in 0..5 {
            let mut graph = Graph::new_publishing();
            graph.insert_nodes(count)?;
            graph.add_edges(&(0..count - 1).map(|id| (id, id + 1)).collect::<Vec<_>>())?;
            graph.set_node_attrs(HashMap::from([(
                "score".into(),
                (0..count).map(|id| (id, V::Float64(id as f64))).collect(),
            )]))?;
            let pin = graph.read_view()?;
            let head = graph.current_state_id();
            let started = Instant::now();
            let table = pin.nodes_table()?.into_base();
            let export_seconds = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let selected = table
                .sort_by("score", false)?
                .head(count / 2)
                .select(&["score".into()])?;
            let selection_seconds = started.elapsed().as_secs_f64();
            let edited = selected.with_column(
                "score".into(),
                BaseArray::from_attr_values(vec![V::Float64(0.25); count / 2]),
            )?;
            let started = Instant::now();
            let result = edited.update_source_graph(&mut graph, &["score".into()], None, None)?;
            let write_seconds = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let joined = table.inner_join(&table, "node_id", "node_id")?;
            let join_seconds = started.elapsed().as_secs_f64();
            if result.batch.state_id != head + 1
                || result.view.node_count() != count / 2
                || joined.source_views().len() != 1
                || joined.source_entity_ids().is_some()
                || table.source_entity_ids().map(|ids| ids.len()) != Some(count)
            {
                return Err("source identity mismatch".into());
            }
            for &node in selected.source_entity_ids().ok_or("missing rows")? {
                if graph.get_node_attr(node, &"score".into())? != Some(V::Float64(0.25))
                    || pin.node_attr(node, &"score".into())? != Some(V::Float64(node as f64))
                {
                    return Err("value/source pin mismatch".into());
                }
            }
            samples.push(json!({"export_seconds":export_seconds,"selection_seconds":selection_seconds,"write_seconds":write_seconds,"join_seconds":join_seconds}));
        }
        cases.push(json!({"nodes":count,"samples":samples}));
    }
    let path = std::env::var("GROGGY_TABLE_PIN_PROFILE")
        .unwrap_or_else(|_| "table_source_pins.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized frozen node-table export, descending sort/head/projection, half-row source writeback and read-only self join. Five observations. Fixture/initial pin/edit preparation/final verification excluded; write includes complete view-source validation, output capture and publication. No matched speedup, RSS/allocation, FFI or durable-fsync guarantee.","cases":cases}),
        )?,
    )?;
    println!("Wrote table source receipt to {path}");
    Ok(())
}
