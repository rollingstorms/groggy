//! Specialized wrappers reuse canonical scoped traversal and share large membership.
use groggy::{
    subgraphs::{ComponentSubgraph, FilteredSubgraph, NeighborhoodSubgraph, Subgraph},
    traits::{filter_operations::FilterCriteria, SubgraphOperations},
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
        let base = Subgraph::new(
            owner.clone(),
            (0..count).collect(),
            (0..count - 1).collect(),
            "base".into(),
        );
        let filter = FilteredSubgraph::new(
            owner.clone(),
            base.nodes().clone(),
            base.edges().clone(),
            FilterCriteria::And(Vec::new()),
            count,
            count - 1,
        );
        let component = ComponentSubgraph::new(
            owner.clone(),
            base.nodes().clone(),
            base.edges().clone(),
            0,
            true,
            1,
        );
        let neighborhood = NeighborhoodSubgraph::new(
            owner,
            vec![0],
            3,
            base.nodes().clone(),
            base.edges().clone(),
        );
        let mut timings = Vec::new();
        for view in [
            &base as &dyn SubgraphOperations,
            &filter,
            &component,
            &neighborhood,
        ] {
            assert_eq!(view.bfs(0, Some(3))?.node_count(), 4);
            let mut samples = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                for _ in 0..1000 {
                    let result = view.bfs(0, Some(3))?;
                    assert_eq!(result.node_count(), 4);
                    assert_eq!(result.edge_count(), 3);
                    black_box(result);
                }
                samples.push(start.elapsed().as_secs_f64() / 1000.);
            }
            let mut induction = Vec::new();
            let mut edge_selection = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                let child = view.induced_subgraph(&[0, 1, 2, 3])?;
                induction.push(start.elapsed().as_secs_f64());
                assert_eq!(child.node_count(), 4);
                assert_eq!(child.edge_count(), 3);
                let start = Instant::now();
                let child = view.subgraph_from_edges(&[0, 1, 2])?;
                edge_selection.push(start.elapsed().as_secs_f64());
                assert_eq!(child.node_count(), 4);
                assert_eq!(child.edge_count(), 3);
            }
            timings.push(json!({"kind":view.entity_type(),"warm_three_hop_bfs_seconds":samples,"four_node_induction_seconds":induction,"three_edge_selection_seconds":edge_selection}));
        }
        let start = Instant::now();
        for _ in 0..1000 {
            let clone = filter.clone();
            assert!(std::ptr::eq(filter.node_set(), clone.node_set()));
            black_box(clone);
        }
        cases.push(json!({"nodes":count,"edges":count-1,"timings":timings,"filter_clone_seconds_per_call":start.elapsed().as_secs_f64()/1000.,"object_sizes_bytes":{"base":std::mem::size_of::<Subgraph>(),"filtered":std::mem::size_of::<FilteredSubgraph>(),"component":std::mem::size_of::<ComponentSubgraph>(),"neighborhood":std::mem::size_of::<NeighborhoodSubgraph>()}}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"undirected chain; no properties; absolute warmed topology-only timings with output and membership-sharing checks; no before/after speedup claim","cases":cases})
        )?
    );
    Ok(())
}
