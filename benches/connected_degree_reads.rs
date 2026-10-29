//! Topology-only compatibility degree and warmed neighbor costs; no property capture.
use groggy::{
    subgraphs::{DegreeMode, Subgraph},
    traits::SubgraphOperations,
    Graph,
};
use serde_json::json;
use std::{cell::RefCell, hint::black_box, rc::Rc, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 100_000] {
        let mut graph = Graph::new_manual();
        graph.add_nodes(count)?;
        for node in 1..count {
            graph.add_edge(node - 1, node)?;
        }
        let owner = Rc::new(RefCell::new(graph));
        let selected = Subgraph::new(
            owner,
            (0..count).collect(),
            (0..count - 1).collect(),
            "chain".into(),
        );
        let ids: Vec<_> = (0..count).collect();
        let start = Instant::now();
        let read = selected.degree_read(DegreeMode::Incident, false)?;
        let cold = start.elapsed().as_secs_f64();
        assert_eq!(read.counts(&[0, count - 1])?, vec![1, 1]);
        let mut preparation = Vec::new();
        let mut bulk = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            black_box(selected.degree_read(DegreeMode::Incident, false)?);
            preparation.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            let counts = read.counts(&ids)?;
            bulk.push(start.elapsed().as_secs_f64());
            assert_eq!(counts.iter().sum::<usize>(), 2 * (count - 1));
        }
        selected.neighbors(0)?;
        let start = Instant::now();
        for _ in 0..10_000 {
            black_box(selected.neighbors(0)?);
        }
        let neighbor = start.elapsed().as_secs_f64() / 10_000.;
        let start = Instant::now();
        for _ in 0..10_000 {
            black_box(selected.degree(0)?);
        }
        let degree = start.elapsed().as_secs_f64() / 10_000.;
        cases.push(json!({"nodes":count,"edges":count-1,"cold_prepare_seconds":cold,"warm_prepare_seconds":preparation,"bulk_count_seconds":bulk,"warm_neighbor_seconds_per_call":neighbor,"warm_native_degree_seconds_per_call":degree}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"undirected chain, no properties; topology-only, checksum-verified; absolute timings, no baseline speedup claim","cases":cases})
        )?
    );
    Ok(())
}
