//! Cold/warm selected topology preparation and owner-free adjacency export.
use groggy::{subgraphs::Subgraph, Graph};
use serde_json::json;
use std::{cell::RefCell, hint::black_box, rc::Rc, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 100_000] {
        let mut graph = Graph::new_directed_manual();
        graph.add_nodes(count)?;
        graph.add_edges(&(1..count).map(|id| (id - 1, id)).collect::<Vec<_>>())?;
        let owner = Rc::new(RefCell::new(graph));
        let connected = Subgraph::new(
            owner,
            (0..count).collect(),
            (0..count - 1).collect(),
            "full".into(),
        );
        let started = Instant::now();
        let read = connected.adjacency_read()?;
        let cold = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let frozen = connected.pin_view()?;
        let capture = started.elapsed().as_secs_f64();
        let mut warm = Vec::new();
        let mut captured = Vec::new();
        let mut exported = Vec::new();
        for _ in 0..5 {
            let started = Instant::now();
            black_box(connected.adjacency_read()?);
            warm.push(started.elapsed().as_secs_f64());
            let started = Instant::now();
            black_box(frozen.adjacency_read()?);
            captured.push(started.elapsed().as_secs_f64());
            let started = Instant::now();
            let rows = read.adjacency_list();
            exported.push(started.elapsed().as_secs_f64());
            assert_eq!(rows.len(), count);
            for (node, neighbors) in &rows {
                assert_eq!(
                    neighbors,
                    &if *node + 1 < count {
                        vec![node + 1]
                    } else {
                        vec![]
                    }
                );
            }
        }
        cases.push(json!({"nodes":count,"edges":count-1,"cold_connected_preparation_seconds":cold,"immutable_capture_seconds":capture,"warm_connected_preparation_seconds":warm,"captured_preparation_seconds":captured,"detached_export_seconds":exported}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"fixture":"release native directed chain; five samples; owner setup excluded; capture and CSR preparation separate from row export; exact analytic row checks; no Python or RSS claim", "cases":cases})
        )?
    );
    Ok(())
}
