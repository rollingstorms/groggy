//! Explicit-index reuse versus ephemeral retained temporal kernels.
use groggy::{
    algorithms::{
        steps::{
            temporal::{
                AggregateFunction, MarkChangedNodesStep, TemporalFilterStep, TemporalWindowStep,
                WindowAggregateStep,
            },
            Step, StepScope, StepVariables,
        },
        AlgorithmParamValue, Context,
    },
    subgraphs::Subgraph,
    AttrValue, Graph,
};
use serde_json::json;
use std::{cell::RefCell, collections::HashMap, rc::Rc, sync::Arc, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        let mut graph = Graph::new_manual();
        graph.add_nodes(count).unwrap();
        graph.set_node_attrs(HashMap::from([(
            "score".into(),
            (0..count).map(|id| (id, AttrValue::Int(1))).collect(),
        )]))?;
        let first = graph.commit("first".into(), "bench".into())?;
        graph.set_node_attrs(HashMap::from([(
            "score".into(),
            (0..count).map(|id| (id, AttrValue::Int(3))).collect(),
        )]))?;
        let head = graph.commit("second".into(), "bench".into())?;
        let index = Arc::new(graph.build_temporal_index()?);
        let selected = Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            (0..count).collect(),
            "windows".into(),
        )?;
        for mode in [
            "index_build",
            "aggregate_reused",
            "aggregate_ephemeral",
            "filter_reused",
            "mark_reused",
            "window_reused",
        ] {
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut vars = StepVariables::default();
                if mode.ends_with("reused") {
                    vars.set_temporal_index("temporal_index", Arc::clone(&index));
                }
                let step: Box<dyn Step> = match mode {
                    "filter_reused" => Box::new(TemporalFilterStep::created_after(0, "result")),
                    "mark_reused" => Box::new(
                        MarkChangedNodesStep::modified_only("result").with_window(head, head),
                    ),
                    "window_reused" => Box::new(TemporalWindowStep::new(first, head, "result")),
                    _ => Box::new(
                        WindowAggregateStep::new(
                            "score",
                            AggregateFunction::Sum,
                            "result",
                            "temporal_index",
                        )
                        .with_window(first, head),
                    ),
                };
                let mut ctx = Context::new();
                let mut scope = StepScope::new(&selected, &mut vars);
                let start = Instant::now();
                let built = if mode == "index_build" {
                    Some(selected.graph().borrow().build_temporal_index()?)
                } else {
                    step.apply(&mut ctx, &mut scope)?;
                    None
                };
                let seconds = start.elapsed().as_secs_f64();
                if let Some(built) = built {
                    if built.statistics().total_nodes != count
                        || built.statistics().total_commits != 2
                    {
                        return Err("index shape mismatch".into());
                    }
                    for id in 0..count {
                        if built.node_attr_history(id, &"score".into(), first, head)
                            != vec![
                                (first, Some(AttrValue::Int(1))),
                                (head, Some(AttrValue::Int(3))),
                            ]
                        {
                            return Err("index history mismatch".into());
                        }
                    }
                } else {
                    let map = scope.variables().node_map("result")?;
                    let expected = if mode.starts_with("aggregate") {
                        AlgorithmParamValue::Float(4.0)
                    } else {
                        AlgorithmParamValue::Int(1)
                    };
                    if map.len() != count
                        || map
                            .iter()
                            .any(|(&id, value)| id >= count || *value != expected)
                    {
                        return Err("temporal result mismatch".into());
                    }
                    if mode == "aggregate_ephemeral" && scope.variables().contains("temporal_index")
                    {
                        return Err("ephemeral index leaked into variables".into());
                    }
                }
                if selected.graph().borrow().current_state_id() != head
                    || selected.graph().borrow().has_uncommitted_changes()
                {
                    return Err("read changed owner".into());
                }
                samples.push(json!({"seconds":seconds}));
            }
            cases.push(json!({"mode":mode,"nodes":count,"states":2,"attribute_events":2*count,"samples":samples}));
        }
    }
    let path = std::env::var("GROGGY_WINDOW_PROFILE")
        .unwrap_or_else(|_| "temporal_window_steps.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native release retained branch index build and four temporal kernels; explicit reuse compared with local ephemeral aggregation index. Five observations; fixture/variable preparation and validation excluded. Every history/result and unchanged owner validated. Includes kernel validation, reductions, allocation/result installation and index build only where named. No publication, FFI, RSS/allocation, matched speedup or scale claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote temporal window receipt to {path}");
    Ok(())
}
