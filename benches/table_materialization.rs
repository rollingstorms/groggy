//! Native checked table materialization receipt; no source history or SLA claim.
use groggy::{
    storage::table::{EdgesTable, GraphTable, NodesTable},
    types::GraphType,
    AttrValue, GraphConfig,
};
use serde_json::json;
use std::{collections::HashMap, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        for columns in [1, 4] {
            let mut samples = Vec::new();
            for _ in 0..5 {
                let ids: Vec<_> = (0..count).map(|row| 100_000 + count - row).collect();
                let mut nodes = NodesTable::new(ids.clone());
                let mut edges = EdgesTable::new(
                    (0..count - 1)
                        .map(|row| (200_000 + row, ids[row], ids[row + 1]))
                        .collect(),
                );
                for column in 0..columns {
                    nodes = nodes.with_attributes(
                        format!("value{column}"),
                        ids.iter()
                            .enumerate()
                            .map(|(row, &id)| (id, AttrValue::Float64(row as f64 + 0.125)))
                            .collect::<HashMap<_, _>>(),
                    )?;
                    edges = edges.with_attributes(
                        format!("value{column}"),
                        (0..count - 1)
                            .map(|row| (200_000 + row, AttrValue::Float64(row as f64 + 0.25)))
                            .collect::<HashMap<_, _>>(),
                    )?;
                }
                let table = GraphTable::new(nodes, edges);
                let mut config = GraphConfig::new();
                config.graph_type = GraphType::Directed;
                let started = Instant::now();
                let graph = table.to_graph_with_config(config)?;
                let seconds = started.elapsed().as_secs_f64();
                if graph.current_state_id() != 1
                    || graph.has_uncommitted_changes()
                    || !graph.automatic_publication()
                    || !graph.is_directed()
                    || graph.node_count() != count
                    || graph.edge_count() != count - 1
                {
                    return Err("Incorrect materialized graph publication/topology".into());
                }
                if graph.get_node_attr(count - 1, &"value0".into())?
                    != Some(AttrValue::Float64((count - 1) as f64 + 0.125))
                    || graph.edge_endpoints(count - 2)? != (count - 2, count - 1)
                {
                    return Err("Incorrect remapped row/property".into());
                }
                samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id(),"nodes":graph.node_count(),"edges":graph.edge_count()}));
            }
            cases.push(
                json!({"nodes":count,"property_columns_per_component":columns,"samples":samples}),
            );
        }
    }
    let output = std::env::var("GROGGY_MATERIALIZE_PROFILE")
        .unwrap_or_else(|_| "table_materialization.json".into());
    std::fs::write(
        &output,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release checked materialization from owned tables with descending sparse source IDs and directed chain edges; fixture construction excluded. Five observations per case. Verifies remapped topology/typed property and one clean accepted version. No before/after speedup, FFI latency, RSS/allocation or production budget claim.",
            "cases":cases
        }))?,
    )?;
    println!("Wrote materialization receipt to {output}");
    Ok(())
}
