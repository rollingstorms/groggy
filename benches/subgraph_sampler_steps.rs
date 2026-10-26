//! Bounded native release selector construction/expansion/emission receipt.
use groggy::{
    algorithms::{
        builder::StepSamplePipeline, AlgorithmParamValue, AlgorithmParams, AlgorithmSpec, Context,
    },
    subgraphs::Subgraph,
    Graph,
};
use serde_json::json;
use std::{cell::RefCell, rc::Rc, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        for mode in [
            "iterate_nodes",
            "iterate_edges",
            "neighbors_zero",
            "neighbors_one",
            "mask_unified",
            "array_unified",
        ] {
            let mut steps = Vec::new();
            let mut source = "seeds";
            match mode {
                "iterate_edges" | "array_unified" => steps.push(json!({"id":"sample.iterate_edges","params":{"target":"seeds"}})),
                "mask_unified" => steps.push(json!({"id":"core.sample_nodes","params":{"target":"seeds","count":count/10,"seed":19}})),
                _ => steps.push(json!({"id":"sample.iterate_nodes","params":{"target":"seeds"}})),
            }
            if mode.starts_with("neighbors_") {
                source = "expanded";
                steps.push(json!({"id":"sample.neighbors","params":{"source":"seeds","target":"expanded","hops":usize::from(mode=="neighbors_one")}}));
            }
            steps.push(json!({"id":"sample.emit_subgraphs","params":{"source":source,"target":"__sample_output__","mode":if mode.ends_with("unified") {"unified"} else {"per_item"},"induced":mode.ends_with("unified")}}));
            let mut params = AlgorithmParams::new();
            params.insert("steps", AlgorithmParamValue::Json(json!({"steps":steps})));
            let pipeline = StepSamplePipeline::try_from_spec(&AlgorithmSpec {
                id: "builder.sample_pipeline".into(),
                params,
            })?;
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut graph = Graph::new_directed_manual();
                graph.add_nodes(count).unwrap();
                for node in 0..count {
                    graph.add_edge(node, (node + 1) % count)?;
                }
                graph.commit("fixture".into(), "bench".into())?;
                let before = graph.current_state_id();
                let shared = Rc::new(RefCell::new(graph));
                let selected =
                    Subgraph::from_nodes(shared.clone(), (0..count).collect(), "ring".into())?;
                let start = Instant::now();
                let outputs = pipeline.run(&mut Context::new(), selected)?;
                let seconds = start.elapsed().as_secs_f64();
                if shared.borrow().current_state_id() != before
                    || shared.borrow().has_uncommitted_changes()
                {
                    return Err("unexpected mutation".into());
                }
                if outputs.len() != if mode.ends_with("unified") { 1 } else { count } {
                    return Err("invalid output count".into());
                }
                for (index, output) in outputs.iter().enumerate() {
                    let mut nodes: Vec<_> = output.nodes().iter().copied().collect();
                    let mut edges: Vec<_> = output.edges().iter().copied().collect();
                    nodes.sort_unstable();
                    edges.sort_unstable();
                    let expected_nodes = if mode == "array_unified" {
                        count
                    } else if mode == "mask_unified" {
                        count / 10
                    } else if mode == "iterate_edges" || mode == "neighbors_one" {
                        2
                    } else {
                        1
                    };
                    if nodes.len() != expected_nodes {
                        return Err("invalid nodes".into());
                    }
                    if mode == "mask_unified" || mode == "array_unified" {
                        for node in 0..count {
                            let internal = nodes.binary_search(&node).is_ok()
                                && nodes.binary_search(&((node + 1) % count)).is_ok();
                            if edges.binary_search(&node).is_ok() != internal {
                                return Err("invalid induced edge set".into());
                            }
                        }
                    } else {
                        let expected_edges = if mode == "iterate_edges" || mode == "neighbors_one" {
                            vec![index]
                        } else {
                            vec![]
                        };
                        if edges.as_slice() != expected_edges {
                            return Err("invalid selected edges".into());
                        }
                        if nodes.binary_search(&index).is_err() {
                            return Err("unstable output order".into());
                        }
                    }
                }
                samples
                    .push(json!({"seconds":seconds,"state_id":shared.borrow().current_state_id()}));
            }
            cases.push(json!({"mode":mode,"nodes":count,"edges":count,"samples":samples}));
        }
    }
    let path = std::env::var("GROGGY_SUBGRAPH_SAMPLER_PROFILE")
        .unwrap_or_else(|_| "subgraph_sampler_steps.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native release directed-ring selector construction, bounded expansion and emission. Fixture/spec construction excluded; five observations per workload. Validate all result memberships, induction, output order/cardinality and zero graph publications. Legacy connected selectors, not pinned snapshots. No matched speedup, FFI/RSS/allocation or production-scale claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote subgraph sampler receipt to {path}");
    Ok(())
}
