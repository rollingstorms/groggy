//! Bounded native release receipt for checked mask samplers and publication.
use groggy::{
    algorithms::{
        builder::register_algorithms, AlgorithmParamValue, AlgorithmParams, AlgorithmSpec, Context,
        Registry,
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
            "nodes_count",
            "nodes_fraction",
            "edges_count",
            "edges_fraction",
            "nodes_reservoir",
            "edges_reservoir",
        ] {
            let edges = mode.starts_with("edges");
            let reservoir = mode.ends_with("reservoir");
            let kind = if edges { "edges" } else { "nodes" };
            let mut steps = Vec::new();
            if reservoir {
                steps.push(json!({"id":format!("core.init_{kind}"),"params":{"target":"source","value":0}}));
                steps.push(json!({"id":"core.reservoir_sample","params":{"source":"source","target":"sample","k":count/10,"seed":19,"entity_type":kind}}));
            } else {
                let mut params = json!({"target":"sample","seed":19});
                if mode.ends_with("count") {
                    params["count"] = json!(count / 10);
                } else {
                    params["fraction"] = json!(0.1);
                }
                steps.push(json!({"id":format!("core.sample_{kind}"),"params":params}));
            }
            steps.push(json!({"id":if edges {"core.attach_edge_attr"} else {"core.attach_node_attr"},"params":{"source":"sample","attr":"sampled"}}));
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
                let mut graph = Graph::new_directed();
                graph.add_nodes(count);
                for node in 0..count {
                    graph.add_edge(node, (node + 1) % count)?;
                }
                graph.commit("fixture".into(), "bench".into())?;
                let before = graph.current_state_id();
                let shared = Rc::new(RefCell::new(graph));
                let selection =
                    Subgraph::from_nodes(shared.clone(), (0..count).collect(), "sample".into())?;
                let start = Instant::now();
                algorithm.execute(&mut Context::new(), selection)?;
                let seconds = start.elapsed().as_secs_f64();
                let graph = shared.borrow();
                if graph.current_state_id() != before + 1 || graph.has_uncommitted_changes() {
                    return Err("invalid publication".into());
                }
                let mut selected = 0;
                for id in 0..count {
                    let value = if edges {
                        graph.get_edge_attr(id, &"sampled".into())?
                    } else {
                        graph.get_node_attr(id, &"sampled".into())?
                    };
                    match value {
                        Some(AttrValue::Int(1) | AttrValue::SmallInt(1)) => selected += 1,
                        None | Some(AttrValue::Null) => {}
                        other => {
                            return Err(format!("invalid mask {mode} at {id}: {other:?}").into())
                        }
                    }
                }
                if selected != count / 10 {
                    return Err("invalid sample size".into());
                }
                samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id()}));
            }
            cases.push(json!({"mode":mode,"nodes":count,"edges":count,"samples":samples}));
        }
    }
    let path = std::env::var("GROGGY_SAMPLING_PROFILE")
        .unwrap_or_else(|_| "sampling_builder_steps.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native release directed rings, 10% mask sampling plus property attachment and one checked publication. Fixture/spec construction excluded. Five observations per workload; all mask cells, cardinality and publication checked. No before/after speedup, FFI/RSS/allocation or production-scale claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote sampling receipt to {path}");
    Ok(())
}
