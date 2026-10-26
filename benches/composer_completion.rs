//! Checked composer presets, selectors and bounded connectivity contraction receipt.
use groggy::{
    subgraphs::{ComposerOptions, EdgeStrategy, MetaNodePlan, Subgraph},
    traits::SubgraphOperations,
    AttrValue, Graph,
};
use serde_json::json;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
    time::Instant,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        for mode in [
            "preset",
            "selector_include",
            "selector_empty",
            "contract_directed",
            "contract_undirected",
        ] {
            let contract = mode.starts_with("contract_");
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut graph = if mode == "contract_directed" {
                    Graph::new_directed_manual().into_publishing()?
                } else {
                    Graph::new_publishing()
                };
                graph.insert_nodes(count)?;
                graph.add_edges_with_attrs(
                    (0..count - 1)
                        .map(|node| {
                            (
                                node,
                                node + 1,
                                HashMap::from([
                                    ("capacity".into(), AttrValue::Int(1)),
                                    ("flow".into(), AttrValue::Int(1)),
                                ]),
                            )
                        })
                        .collect(),
                )?;
                let shared = Rc::new(RefCell::new(graph));
                let outer = if contract {
                    None
                } else {
                    let last = Subgraph::new(
                        shared.clone(),
                        [count - 1].into_iter().collect(),
                        HashSet::new(),
                        "outer".into(),
                    );
                    let outer = MetaNodePlan::new(last.node_set().clone(), HashSet::new())
                        .with_edge_strategy(EdgeStrategy::DropAll)
                        .add_to_graph(&last)?
                        .id();
                    shared.borrow_mut().add_edge(count / 2 - 1, outer)?;
                    Some(outer)
                };
                let (nodes, edges): (HashSet<_>, HashSet<_>) = if contract {
                    ((1..count - 1).collect(), (1..count - 2).collect())
                } else {
                    ((0..count / 2).collect(), (0..count / 2 - 1).collect())
                };
                let selected = Subgraph::new(shared.clone(), nodes, edges, "bench".into());
                let plan = selected.collapse_with_options(ComposerOptions {
                    preset: Some(
                        if contract {
                            "flow_network"
                        } else {
                            "org_hierarchy"
                        }
                        .into(),
                    ),
                    node_strategy: Some(groggy::traits::NodeStrategy::Collapse),
                    ..Default::default()
                })?;
                let targets = match mode {
                    "selector_include" => Some(vec![outer.ok_or("missing outer")?]),
                    "selector_empty" => Some(Vec::new()),
                    _ => None,
                };
                let before = shared.borrow().current_state_id();
                let start = Instant::now();
                let meta = plan.execute_configured(&selected, false, targets.as_deref())?;
                let seconds = start.elapsed().as_secs_f64();
                let graph = shared.borrow();
                if graph.current_state_id() != before + 1
                    || graph.has_uncommitted_changes()
                    || !graph.is_meta_node(meta.id())
                {
                    return Err("publication/identity mismatch".into());
                }
                if graph.node_labels(meta.id())?
                    != vec![if contract { "junction" } else { "department" }.to_string()]
                {
                    return Err("label mismatch".into());
                }
                if contract {
                    if graph.node_count() != 3 || graph.edge_count() != 1 {
                        return Err("contraction shape mismatch".into());
                    }
                    let edge = graph.edge_ids()[0];
                    if graph.edge_endpoints(edge)? != (0, count - 1)
                        || graph.get_edge_attr(edge, &"capacity".into())? != Some(AttrValue::Int(1))
                        || graph.get_edge_attr(edge, &"flow".into())?
                            != Some(AttrValue::Int((count - 1) as i64))
                    {
                        return Err("contraction attributes mismatch".into());
                    }
                } else {
                    let mut actual: Vec<_> = graph
                        .edge_ids()
                        .into_iter()
                        .filter_map(|id| {
                            graph
                                .edge_endpoints(id)
                                .ok()
                                .filter(|pair| pair.0 == meta.id() || pair.1 == meta.id())
                        })
                        .collect();
                    actual.sort_unstable();
                    let mut expected = vec![(meta.id(), count / 2)];
                    if mode != "selector_empty" {
                        expected.push((meta.id(), outer.ok_or("missing outer")?));
                    }
                    expected.sort_unstable();
                    if actual != expected {
                        return Err("selected boundary mismatch".into());
                    }
                }
                samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id(),"nodes":graph.node_count(),"edges":graph.edge_count()}));
            }
            cases.push(json!({"mode":mode,"fixture_nodes":count,"samples":samples}));
        }
    }
    let path = std::env::var("GROGGY_COMPOSER_PROFILE")
        .unwrap_or_else(|_| "composer_completion.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized chain with bounded external ports, presets/explicit meta targets and directed/undirected contraction. Five observations; fixture/configuration preparation and verification excluded. Includes complete planning, attribute aggregation, retained grouping, source removal and one checked publication. Every generated identity/label/topology/value and clean state checked. No matched speedup, FFI, allocation/RSS or dense-boundary scale claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote composer completion receipt to {path}");
    Ok(())
}
