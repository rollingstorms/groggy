//! Bounded native copying: current values, reference remapping and policy boundaries.
use groggy::{AttrValue as V, Graph};
use serde_json::json;
use std::{collections::HashMap, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        let mut source = Graph::new_directed();
        source.add_nodes(count + 1);
        source.remove_node(0)?;
        let edges =
            source.add_edges(&(1..count).map(|node| (node, node + 1)).collect::<Vec<_>>())?;
        source.set_node_attrs(HashMap::from([
            (
                "weight".into(),
                (1..=count)
                    .map(|node| (node, V::Float64(node as f64 + 0.125)))
                    .collect(),
            ),
            (
                "embedding".into(),
                (1..=count)
                    .map(|node| (node, V::Float64Vec(vec![node as f64; 16])))
                    .collect(),
            ),
            (
                "refs".into(),
                (1..=count)
                    .map(|node| (node, V::NodeArray(vec![1, count])))
                    .collect(),
            ),
        ]))?;
        source.set_edge_attrs(HashMap::from([(
            "refs".into(),
            edges
                .iter()
                .map(|&edge| (edge, V::EdgeArray(vec![edges[0], edges[edges.len() - 1]])))
                .collect(),
        )]))?;
        for publishing in [false, true] {
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut target = Graph::new();
                target.set_automatic_publication(publishing)?;
                target.insert_nodes(3)?;
                target.add_edge(0, 0)?;
                target.remove_node(2)?;
                target.commit("fixture".into(), "benchmark".into())?;
                let before = target.current_state_id();
                let started = Instant::now();
                target.add_graph(&source)?;
                let seconds = started.elapsed().as_secs_f64();
                if target.current_state_id() != before + u64::from(publishing)
                    || target.has_uncommitted_changes() == publishing
                    || target.node_count() != count + 2
                    || target.edge_count() != count
                {
                    return Err("copy boundary mismatch".into());
                }
                for source_node in 1..=count {
                    let target_node = source_node + 2;
                    if target.get_node_attr(target_node, &"weight".into())?
                        != Some(V::Float64(source_node as f64 + 0.125))
                        || target.get_node_attr(target_node, &"embedding".into())?
                            != Some(V::Float64Vec(vec![source_node as f64; 16]))
                        || target.get_node_attr(target_node, &"refs".into())?
                            != Some(V::NodeArray(vec![3, count + 2]))
                    {
                        return Err("copied node mismatch".into());
                    }
                }
                for &edge in &edges {
                    if target.edge_endpoints(edge + 1)? != (edge + 3, edge + 4)
                        || target.get_edge_attr(edge + 1, &"refs".into())?
                            != Some(V::EdgeArray(vec![1, count - 1]))
                    {
                        return Err("copied edge mismatch".into());
                    }
                }
                samples.push(json!({"seconds":seconds,"state_id":target.current_state_id(),"nodes":target.node_count(),"edges":target.edge_count()}));
            }
            cases.push(json!({"nodes_copied":count,"publishing":publishing,"samples":samples}));
        }
    }
    let path =
        std::env::var("GROGGY_GRAPH_COPY_PROFILE").unwrap_or_else(|_| "graph_copy.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native optimized active graph copying; source has sparse IDs, f64 scalar/vector and typed references. Five observations per case; complete validation, capture, remapping, bulk installation and publication (when enabled) measured. Setup and exhaustive verification excluded. No historical/pool copy, matched speedup, RSS/allocation or FFI claim.","cases":cases
        }))?,
    )?;
    println!("Wrote graph copy receipt to {path}");
    Ok(())
}
