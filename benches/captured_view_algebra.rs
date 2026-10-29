//! Captured/connected grouping and exact algebra costs; capture is a separate measurement.
use groggy::{subgraphs::Subgraph, AttrValue as V, Graph, GraphView};
use serde_json::json;
use std::{cell::RefCell, hint::black_box, rc::Rc, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 10_000] {
        for groups in [1, 100, count] {
            let mut graph = Graph::new_manual();
            graph.add_nodes(count)?;
            for node in 0..count {
                graph.set_node_attr(node, "group".into(), V::Int((node % groups) as i64))?;
                if node > 0 {
                    graph.add_edge(node - 1, node)?;
                }
            }
            let start = Instant::now();
            let root = graph.read_view()?;
            let capture = start.elapsed().as_secs_f64();
            let selected = Subgraph::new(
                Rc::new(RefCell::new(graph)),
                root.nodes().clone(),
                root.edges().clone(),
                "bench".into(),
            );
            let left = root.select_nodes(&(0..count / 2).collect::<Vec<_>>())?;
            let right = root.select_nodes(&(count / 4..count).collect::<Vec<_>>())?;
            let mut frozen = Vec::new();
            let mut connected = Vec::new();
            let mut algebra = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                let values = root.group_by_nodes(&"group".into())?;
                assert_eq!(
                    values.iter().map(GraphView::node_count).sum::<usize>(),
                    count
                );
                assert_eq!(values.len(), groups);
                black_box(values);
                frozen.push(start.elapsed().as_secs_f64());
                let start = Instant::now();
                let values = selected.group_by_nodes(&"group".into())?;
                assert_eq!(values.len(), groups);
                black_box(values);
                connected.push(start.elapsed().as_secs_f64());
                let start = Instant::now();
                assert_eq!(left.merge_with(&right)?.node_count(), count);
                assert_eq!(left.intersect_with(&right)?.node_count(), count / 4);
                assert_eq!(left.subtract_from(&right)?.node_count(), count / 4);
                algebra.push(start.elapsed().as_secs_f64());
            }
            cases.push(json!({"nodes":count,"edges":root.edge_count(),"groups":groups,"capture_seconds":capture,"frozen_group_seconds":frozen,"connected_group_seconds":connected,"three_set_operations_seconds":algebra}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"chain, one integer column; complete membership and per-group edge closure asserted; five samples; cold capture reported separately","cases":cases})
        )?
    );
    Ok(())
}
