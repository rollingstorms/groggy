//! Bounded release receipt for full logical diffs and atomic-capable diff masks.
use groggy::{
    algorithms::{
        steps::{
            temporal::{DiffEdgesStep, DiffNodesStep},
            Step, StepScope, StepVariables,
        },
        temporal::TemporalDelta,
        AlgorithmParamValue, Context,
    },
    subgraphs::Subgraph,
    AttrValue, Graph,
};
use serde_json::json;
use std::{cell::RefCell, collections::HashMap, rc::Rc, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        let mut graph = Graph::new_manual();
        graph.add_nodes(count).unwrap();
        for id in 0..count {
            graph.add_edge(id, (id + 1) % count)?;
        }
        graph.set_node_attrs(HashMap::from([(
            "score".into(),
            (0..count).map(|id| (id, AttrValue::Int(0))).collect(),
        )]))?;
        graph.set_edge_attrs(HashMap::from([(
            "weight".into(),
            (0..count).map(|id| (id, AttrValue::Int(0))).collect(),
        )]))?;
        let first = graph.commit("before".into(), "bench".into())?;
        let before = graph.snapshot_at_commit(first)?;
        graph.set_node_attrs(HashMap::from([(
            "score".into(),
            (0..count).map(|id| (id, AttrValue::Int(2))).collect(),
        )]))?;
        graph.set_edge_attrs(HashMap::from([(
            "weight".into(),
            (0..count).map(|id| (id, AttrValue::Int(2))).collect(),
        )]))?;
        let head = graph.commit("after".into(), "bench".into())?;
        let after = graph.snapshot_at_commit(head)?;
        let selected = Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            (0..count).collect(),
            "diff".into(),
        )?;
        for mode in ["logical_delta", "node_masks", "edge_masks"] {
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut vars = StepVariables::default();
                vars.set_snapshot("before", before.clone());
                vars.set_snapshot("after", after.clone());
                let mut scope = StepScope::new(&selected, &mut vars);
                let mut ctx = Context::new();
                let step: Box<dyn Step> = if mode == "edge_masks" {
                    Box::new(DiffEdgesStep::new(
                        Some("before".into()),
                        Some("after".into()),
                        "diff",
                    ))
                } else {
                    Box::new(DiffNodesStep::new(
                        Some("before".into()),
                        Some("after".into()),
                        "diff",
                    ))
                };
                let start = Instant::now();
                let delta = if mode == "logical_delta" {
                    Some(TemporalDelta::compute(&before, &after)?)
                } else {
                    step.apply(&mut ctx, &mut scope)?;
                    None
                };
                let seconds = start.elapsed().as_secs_f64();
                if let Some(delta) = delta {
                    if !delta.nodes_added.is_empty()
                        || !delta.nodes_removed.is_empty()
                        || !delta.edges_added.is_empty()
                        || !delta.edges_removed.is_empty()
                        || delta.node_attr_changes.len() != count
                        || delta.edge_attr_changes.len() != count
                    {
                        return Err("invalid delta shape".into());
                    }
                    for (id, change) in delta.node_attr_changes.iter().enumerate() {
                        if change.node_id != id
                            || change.attr_name != "score"
                            || change.old_value != Some(AttrValue::Int(0))
                            || change.new_value != Some(AttrValue::Int(2))
                        {
                            return Err("invalid node delta".into());
                        }
                    }
                    for (id, change) in delta.edge_attr_changes.iter().enumerate() {
                        if change.edge_id != id
                            || change.attr_name != "weight"
                            || change.old_value != Some(AttrValue::Int(0))
                            || change.new_value != Some(AttrValue::Int(2))
                        {
                            return Err("invalid edge delta".into());
                        }
                    }
                } else {
                    for suffix in ["added", "removed", "modified"] {
                        let name = format!(
                            "diff_{}_{suffix}",
                            if mode == "node_masks" {
                                "nodes"
                            } else {
                                "edges"
                            }
                        );
                        let map = if mode == "node_masks" {
                            scope.variables().node_map(&name)?
                        } else {
                            scope.variables().edge_map(&name)?
                        };
                        if map.len() != if suffix == "modified" { count } else { 0 }
                            || map.iter().any(|(&id, value)| {
                                id >= count || *value != AlgorithmParamValue::Int(1)
                            })
                        {
                            return Err("invalid masks".into());
                        }
                    }
                }
                if selected.graph().borrow().current_state_id() != head
                    || selected.graph().borrow().has_uncommitted_changes()
                {
                    return Err("diff changed owner".into());
                }
                samples.push(json!({"seconds": seconds}));
            }
            cases.push(json!({"mode": mode, "nodes": count, "edges": count, "node_property_changes": count, "edge_property_changes": count, "samples": samples}));
        }
    }
    let path =
        std::env::var("GROGGY_DIFF_PROFILE").unwrap_or_else(|_| "temporal_diff_steps.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope": "Native optimized retained logical diff and node/edge mask kernels; ring with all node and edge scalar properties changed, five observations. Excludes fixture/snapshot/variable preparation and validation; includes provenance checking, property comparison, stable sorting and result installation. Each mask helper currently computes the full logical diff. Validate every old/new value or mask plus unchanged owner. No property publication, FFI, memory, matched speedup or scale claim.", "cases": cases}),
        )?,
    )?;
    println!("Wrote diff receipt to {path}");
    Ok(())
}
