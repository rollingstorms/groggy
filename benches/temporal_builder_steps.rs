//! Bounded retained snapshot and checked decay kernel receipt.
use groggy::{
    algorithms::{
        steps::{
            temporal::{DecayStep, SnapshotAtStep},
            NodeColumn, Step, StepScope, StepVariables,
        },
        AlgorithmParamValue, Context,
    },
    subgraphs::Subgraph,
    Graph,
};
use serde_json::json;
use std::{cell::RefCell, rc::Rc, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        let mut graph = Graph::new_manual();
        graph.add_nodes(count).unwrap();
        for id in 0..count {
            graph.add_edge(id, (id + 1) % count)?;
        }
        let state = graph.commit("fixture".into(), "bench".into())?;
        let selected = Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            (0..count).collect(),
            "temporal".into(),
        )?;
        for mode in ["node_map", "edge_map", "node_column", "retained_snapshot"] {
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut vars = StepVariables::default();
                if mode == "node_column" {
                    vars.set_node_column(
                        "values",
                        NodeColumn::new(
                            (0..count).rev().collect(),
                            vec![AlgorithmParamValue::Int(8); count],
                        ),
                    );
                } else if mode != "retained_snapshot" {
                    let map = (0..count)
                        .map(|id| (id, AlgorithmParamValue::Int(8)))
                        .collect();
                    if mode == "edge_map" {
                        vars.set_edge_map("values", map);
                    } else {
                        vars.set_node_map("values", map);
                    }
                }
                let step: Box<dyn Step> = if mode == "retained_snapshot" {
                    Box::new(SnapshotAtStep::new_at_commit(state, "result"))
                } else {
                    Box::new(DecayStep::new("values", 2.0, "result").with_time_delta(4.0))
                };
                let mut ctx = Context::new();
                let mut scope = StepScope::new(&selected, &mut vars);
                let start = Instant::now();
                step.apply(&mut ctx, &mut scope)?;
                let seconds = start.elapsed().as_secs_f64();
                if mode == "retained_snapshot" {
                    let snapshot = scope.variables().snapshot("result")?;
                    if snapshot.lineage().commit_id != state
                        || (0..count)
                            .any(|id| !snapshot.node_exists(id) || !snapshot.edge_exists(id))
                    {
                        return Err("invalid snapshot".into());
                    }
                } else {
                    let rows: Vec<_> = if mode == "node_column" {
                        scope
                            .variables()
                            .node_column("result")?
                            .iter()
                            .map(|(id, v)| (id, v.clone()))
                            .collect()
                    } else {
                        let map = if mode == "edge_map" {
                            scope.variables().edge_map("result")?
                        } else {
                            scope.variables().node_map("result")?
                        };
                        map.iter().map(|(&id, v)| (id, v.clone())).collect()
                    };
                    if rows.len() != count
                        || rows.iter().any(|(id, value)| {
                            *id >= count || *value != AlgorithmParamValue::Float(2.0)
                        })
                    {
                        return Err("invalid decay".into());
                    }
                    if mode == "node_column"
                        && rows
                            .iter()
                            .enumerate()
                            .any(|(index, (id, _))| *id != count - index - 1)
                    {
                        return Err("column order changed".into());
                    }
                }
                if selected.graph().borrow().current_state_id() != state
                    || selected.graph().borrow().has_uncommitted_changes()
                {
                    return Err("read-only step changed graph".into());
                }
                samples.push(json!({"seconds": seconds}));
            }
            cases.push(json!({"mode": mode, "nodes": count, "edges": count, "samples": samples}));
        }
    }
    let path = std::env::var("GROGGY_TEMPORAL_PROFILE")
        .unwrap_or_else(|_| "temporal_builder_steps.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope": "Native release kernel-only decay maps/ordered columns and retained full-owner snapshots on a ring, five observations per workload. Fixture/variable setup and validation excluded; includes validation, calculation/reconstruction and private result installation. No property publication, FFI, RSS, allocation, speedup or production-scale claim. Every value, column order, snapshot identity/existence and unchanged owner state validated.", "cases": cases}),
        )?,
    )?;
    println!("Wrote temporal receipt to {path}");
    Ok(())
}
