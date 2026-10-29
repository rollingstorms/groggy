//! Structural summary setup/reuse costs, with cold immutable capture reported separately.
use groggy::{subgraphs::Subgraph, Graph};
use serde_json::json;
use std::{cell::RefCell, rc::Rc, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for (kind, count) in [
        ("chain", 1_000usize),
        ("chain", 100_000),
        ("star", 100_000),
        ("clique", 300),
    ] {
        let mut graph = Graph::new_manual();
        graph.add_nodes(count)?;
        let mut edges = 0;
        match kind {
            "clique" => {
                for s in 0..count {
                    for t in s + 1..count {
                        graph.add_edge(s, t)?;
                        edges += 1;
                    }
                }
            }
            "star" => {
                for t in 1..count {
                    graph.add_edge(0, t)?;
                    edges += 1;
                }
            }
            _ => {
                for t in 1..count {
                    graph.add_edge(t - 1, t)?;
                    edges += 1;
                }
            }
        }
        let owner = Rc::new(RefCell::new(graph));
        let selected = Subgraph::new(
            owner.clone(),
            (0..count).collect(),
            (0..edges).collect(),
            "benchmark".into(),
        );
        let start = Instant::now();
        let input = selected.structural_read()?;
        let live_prepare = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let view = owner.borrow().read_view()?;
        let capture = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let frozen = view.structural_read()?;
        let frozen_prepare = start.elapsed().as_secs_f64();
        let mut all = Vec::new();
        let mut global = Vec::new();
        let mut local = Vec::new();
        let mut repeated_prepare = Vec::new();
        let expected = if kind == "clique" { 1. } else { 0. };
        for _ in 0..5 {
            let start = Instant::now();
            let rows = input.clustering_coefficients()?;
            assert_eq!(rows.len(), count);
            assert!(rows.iter().all(|(_, v)| (*v - expected).abs() < 1e-12));
            all.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            assert!((input.transitivity()? - expected).abs() < 1e-12);
            global.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            assert!((input.clustering_coefficient(Some(0))? - expected).abs() < 1e-12);
            local.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            let repeated = selected.structural_read()?;
            repeated_prepare.push(start.elapsed().as_secs_f64());
            drop(repeated);
        }
        assert!((frozen.transitivity()? - expected).abs() < 1e-12);
        cases.push(json!({"fixture":kind,"nodes":count,"edges":edges,"connected_prepare_seconds":live_prepare,"capture_seconds":capture,"captured_prepare_seconds":frozen_prepare,"all_clustering_seconds":all,"transitivity_seconds":global,"local_clustering_seconds":local,"connected_cached_prepare_seconds":repeated_prepare}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"undirected unweighted simple topology, no properties; setup and five reusable computations measured separately; all rows/objectives checked","cases":cases})
        )?
    );
    Ok(())
}
