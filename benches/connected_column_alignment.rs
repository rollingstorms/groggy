//! Aligned bulk selected reads versus equivalent scalar reads; setup and ID ordering excluded.
use groggy::ownership::OwnerCell;
use groggy::{subgraphs::Subgraph, AttrValue as V, Graph};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 100_000] {
        let mut graph = Graph::new_manual();
        graph.add_nodes(count)?;
        for id in 0..count {
            if id % 3 != 2 {
                graph.set_node_attr(
                    id,
                    "v".into(),
                    if id % 3 == 0 {
                        V::Int(id as i64)
                    } else {
                        V::Null
                    },
                )?;
            }
        }
        let owner = Arc::new(OwnerCell::new(graph));
        let selected = Subgraph::new(
            owner.clone(),
            (0..count).collect(),
            Default::default(),
            "benchmark".into(),
        );
        let ids = selected.node_ids();
        let name = "v".to_owned();
        let mut bulk = Vec::new();
        let mut scalar = Vec::new();
        for sample in 0..5 {
            let mut rows = [Vec::new(), Vec::new()];
            for kind in if sample % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                rows[kind] = if kind == 0 {
                    selected.get_node_attribute_column(&name)?
                } else {
                    let graph = owner.borrow();
                    ids.iter()
                        .map(|&id| {
                            graph
                                .get_node_attr(id, &name)
                                .map(|value| value.unwrap_or(V::Null))
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };
                let elapsed = start.elapsed().as_secs_f64();
                if kind == 0 {
                    bulk.push(elapsed)
                } else {
                    scalar.push(elapsed)
                }
            }
            assert_eq!(rows[0], rows[1]);
            assert_eq!(rows[0].len(), count);
        }
        cases.push(json!({"nodes":count,"bulk_column_seconds":bulk,"equivalent_scalar_column_seconds":scalar}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"manual isolates; integer/null/absent column; precomputed sorted IDs; five alternating samples; both complete row vectors compared; no Python or capture","cases":cases})
        )?
    );
    Ok(())
}
