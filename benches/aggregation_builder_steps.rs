//! Bounded native receipt for the combined checked aggregation builder bundle.
use groggy::{
    algorithms::{
        builder::register_algorithms, AlgorithmOutput, AlgorithmParamValue, AlgorithmParams,
        AlgorithmSpec, Context, Registry,
    },
    subgraphs::Subgraph,
    AttrValue, Graph,
};
use serde_json::json;
use std::{cell::RefCell, rc::Rc, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        for mode in [
            "sum",
            "mean",
            "std",
            "median",
            "quantile",
            "mode",
            "entropy",
            "histogram",
            "neighbor_sum",
            "neighbor_mean",
            "neighbor_min",
            "neighbor_max",
            "neighbor_mode",
            "neighbor_weighted_sum",
        ] {
            let neighbor = mode.starts_with("neighbor_");
            let histogram = mode == "histogram";
            let operation = match mode {
                "sum" | "mean" => {
                    json!({"id":"core.reduce_nodes","params":{"source":"values","target":"stat","reducer":mode}})
                }
                "std" | "median" | "mode" | "entropy" => {
                    json!({"id":format!("core.{mode}"),"params":{"source":"values","target":"stat"}})
                }
                "quantile" => {
                    json!({"id":"core.quantile","params":{"source":"values","target":"stat","q":0.25}})
                }
                "histogram" => {
                    json!({"id":"core.histogram","params":{"source":"values","target":"hist","bins":16}})
                }
                "neighbor_weighted_sum" => {
                    json!({"id":"core.neighbor_agg","params":{"source":"values","weights":"weights","target":"result","agg":"sum","direction":"in"}})
                }
                _ => {
                    json!({"id":"core.neighbor_agg","params":{"source":"values","target":"result","agg":mode.trim_start_matches("neighbor_"),"direction":"in"}})
                }
            };
            let mut steps = vec![
                json!({"id":"core.load_node_attr","params":{"attr":"score","target":"values"}}),
                json!({"id":"core.init_nodes","params":{"target":"weights","value":2}}),
                operation,
            ];
            if !neighbor && !histogram {
                steps.push(json!({"id":"core.broadcast_scalar","params":{"scalar":"stat","reference":"values","target":"result"}}));
            }
            if !histogram {
                steps.push(json!({"id":"core.attach_node_attr","params":{"source":"result","attr":"answer"}}));
            }
            let mut params = AlgorithmParams::new();
            params.insert(
                "steps",
                AlgorithmParamValue::Json(json!({"atomic":true,"steps":steps})),
            );
            let algorithm = registry.instantiate(&AlgorithmSpec {
                id: "builder.step_pipeline".into(),
                params,
            })?;
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut graph = Graph::new_directed_manual();
                graph.add_nodes(count).unwrap();
                for node in 0..count {
                    graph.add_edge(node, (node + 1) % count)?;
                    graph.set_node_attr(node, "score".into(), AttrValue::Int(1))?;
                }
                graph.commit("fixture".into(), "bench".into())?;
                let before = graph.current_state_id();
                let shared = Rc::new(RefCell::new(graph));
                let selection = Subgraph::from_nodes(
                    shared.clone(),
                    (0..count).collect(),
                    "aggregation".into(),
                )?;
                let mut ctx = Context::new();
                let start = Instant::now();
                algorithm.execute(&mut ctx, selection)?;
                let seconds = start.elapsed().as_secs_f64();
                let graph = shared.borrow();
                if graph.current_state_id() != before + u64::from(!histogram)
                    || graph.has_uncommitted_changes()
                {
                    return Err("invalid publication".into());
                }
                if histogram {
                    let outputs = ctx.take_outputs();
                    let Some(AlgorithmOutput::Histogram(counts)) = outputs.get("histograms.hist")
                    else {
                        return Err("missing histogram output".into());
                    };
                    if counts.len() != 16
                        || counts[0] != count as i64
                        || counts.iter().sum::<i64>() != count as i64
                    {
                        return Err("invalid histogram counts".into());
                    }
                } else {
                    let expected = match mode {
                        "mode" | "neighbor_mode" => AttrValue::Int(1),
                        "std" | "entropy" => AttrValue::Float64(0.0),
                        "sum" => AttrValue::Float64(count as f64),
                        "neighbor_weighted_sum" => AttrValue::Float64(2.0),
                        _ => AttrValue::Float64(1.0),
                    };
                    for node in 0..count {
                        if graph.get_node_attr(node, &"answer".into())? != Some(expected.clone()) {
                            return Err("invalid aggregation result".into());
                        }
                    }
                }
                samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id()}));
            }
            cases.push(json!({"mode":mode,"nodes":count,"edges":count,"samples":samples}));
        }
    }
    let path = std::env::var("GROGGY_AGGREGATION_PROFILE")
        .unwrap_or_else(|_| "aggregation_builder_steps.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native release ring workloads; fixture/spec construction excluded. Includes reduction or neighbor calculation, owned histogram or property attachment and checked publication. Five samples per workload; all outputs checked. No matched speedup, FFI, RSS/allocation or production budget claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote aggregation receipt to {path}");
    Ok(())
}
