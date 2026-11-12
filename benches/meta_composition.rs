//! Native composition receipt: fixture construction excluded; no SLA claim.
use groggy::{
    subgraphs::{MetaNodePlan, Subgraph},
    traits::NodeStrategy,
    Graph,
};
use serde_json::json;
use std::{cell::RefCell, collections::HashSet, rc::Rc, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|arg| arg == "--collection") {
        return collection_receipt();
    }
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        for selected in [1, count / 2] {
            for collapse in [false, true] {
                let mut samples = Vec::new();
                for _ in 0..5 {
                    let mut graph = Graph::new_publishing();
                    graph.insert_nodes(count)?;
                    graph.add_edges(&(0..count - 1).map(|id| (id, id + 1)).collect::<Vec<_>>())?;
                    let shared = Rc::new(RefCell::new(graph));
                    let nodes: HashSet<_> = (0..selected).collect();
                    let edges: HashSet<_> = (0..selected.saturating_sub(1)).collect();
                    let selection =
                        Subgraph::new(shared.clone(), nodes.clone(), edges.clone(), "bench".into());
                    let mut plan = MetaNodePlan::new(nodes, edges).with_node_agg(
                        "size".into(),
                        "count".into(),
                        None,
                    );
                    plan.node_strategy = if collapse {
                        NodeStrategy::Collapse
                    } else {
                        NodeStrategy::Extract
                    };
                    let started = Instant::now();
                    let meta = plan.add_to_graph(&selection)?;
                    std::hint::black_box(meta);
                    let seconds = started.elapsed().as_secs_f64();
                    let graph = shared.borrow();
                    let expected_nodes = count + 1 - if collapse { selected } else { 0 };
                    let expected_edges = count - 1 + 1 - if collapse { selected } else { 0 };
                    if graph.current_state_id() != 3
                        || graph.has_uncommitted_changes()
                        || graph.node_count() != expected_nodes
                        || graph.edge_count() != expected_edges
                    {
                        return Err("Unexpected composition publication/topology".into());
                    }
                    samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id(),"nodes":graph.node_count(),"edges":graph.edge_count()}));
                }
                cases.push(json!({"graph_nodes":count,"selected_nodes":selected,"strategy":if collapse {"collapse"} else {"extract"},"samples":samples}));
            }
        }
    }
    let output = std::env::var("GROGGY_COMPOSITION_PROFILE")
        .unwrap_or_else(|_| "meta_composition.json".into());
    std::fs::write(
        &output,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release chain graphs; fixture construction excluded. Includes planning, deterministic sorting, retained group storage and checked publication. Five observations per case; no before/after speedup, FFI latency, allocation/RSS or SLA claim.",
            "cases":cases
        }))?,
    )?;
    println!("Wrote composition receipt to {output}");
    Ok(())
}

fn collection_receipt() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [10, 100, 500, 2000] {
        let mut samples = Vec::new();
        for _ in 0..5 {
            let mut graph = Graph::new();
            graph.add_nodes(2 * count)?;
            graph.add_edges(&(0..count).map(|i| (2 * i, 2 * i + 1)).collect::<Vec<_>>())?;
            let head = graph.current_state_id();
            let shared = Rc::new(RefCell::new(graph));
            let plans = (0..count)
                .map(|i| {
                    let nodes = HashSet::from([2 * i, 2 * i + 1]);
                    let edges = HashSet::from([i]);
                    let selection =
                        Subgraph::new(shared.clone(), nodes.clone(), edges.clone(), "bench".into())
                            .freeze()?;
                    let mut plan = MetaNodePlan::new(nodes, edges).with_node_agg(
                        "size".into(),
                        "count".into(),
                        None,
                    );
                    plan.node_strategy = NodeStrategy::Collapse;
                    Ok((plan, selection))
                })
                .collect::<groggy::GraphResult<Vec<_>>>()?;
            let start = Instant::now();
            let metas = MetaNodePlan::execute_many(&plans, false, None)?;
            samples.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&metas);
            let graph = shared.borrow();
            assert_eq!(metas.len(), count);
            assert_eq!(graph.node_count(), count);
            assert_eq!(graph.edge_count(), 0);
            assert_eq!(graph.current_state_id(), head + 1);
            assert!(!graph.has_uncommitted_changes());
        }
        cases.push(json!({"components":count,"source_nodes":2*count,"source_edges":count,"seconds":samples}));
    }
    let output = std::env::var("GROGGY_COMPOSITION_PROFILE")
        .unwrap_or_else(|_| "meta_composition_collection.json".into());
    std::fs::write(
        &output,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release disconnected pairs; captured selections and graph construction excluded; count aggregation; destructive collection collapse; five samples; validation excluded; no memory/SLA claim", "cases":cases
        }))?,
    )?;
    println!("Wrote collection composition receipt to {output}");
    Ok(())
}
