//! Bounded native receipt for checked expression and transformation builders.
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
            "scalar_map",
            "staged_attr_map",
            "neighbor_map_sync",
            "neighbor_map_async",
            "neighbor_mode",
            "update_in_place",
        ] {
            let expr = match mode {
                "scalar_map" => {
                    json!({"type":"binary_op","op":"mul","left":{"type":"var","name":"value"},"right":{"type":"const","value":2}})
                }
                "staged_attr_map" => json!({"type":"attr","name":"staged"}),
                _ => {
                    json!({"type":"call","func":"sum","args":[{"type":"call","func":"neighbor_values","args":[{"type":"var","name":"values"}]}]})
                }
            };
            let operation = match mode {
                "neighbor_mode" => {
                    json!({"id":"core.neighbor_mode_update","params":{"target":"values","output":"result","include_self":false,"ordered":true}})
                }
                "update_in_place" => {
                    json!({"id":"core.update_in_place","params":{"source":"values","target":"destination","output":"result","ordered":true}})
                }
                _ => {
                    json!({"id":"core.map_nodes","params":{"source":"values","target":"result","expr":expr,"async_update":mode=="neighbor_map_async"}})
                }
            };
            let mut steps = vec![
                json!({"id":"core.load_node_attr","params":{"attr":"score","target":"values"}}),
            ];
            if mode == "staged_attr_map" {
                steps.push(json!({"id":"core.attach_node_attr","params":{"source":"values","attr":"staged"}}));
            }
            if mode == "update_in_place" {
                steps.push(
                    json!({"id":"core.init_nodes","params":{"target":"destination","value":0}}),
                );
            }
            steps.push(operation);
            steps.push(
                json!({"id":"core.attach_node_attr","params":{"source":"result","attr":"answer"}}),
            );
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
                let selection =
                    Subgraph::from_nodes(shared.clone(), (0..count).collect(), "transform".into())?;
                let mut ctx = Context::new();
                let start = Instant::now();
                algorithm.execute(&mut ctx, selection)?;
                let seconds = start.elapsed().as_secs_f64();
                let graph = shared.borrow();
                if graph.current_state_id() != before + 1 || graph.has_uncommitted_changes() {
                    return Err("invalid publication".into());
                }
                let expected = AttrValue::Int(if mode == "scalar_map" { 2 } else { 1 });
                for node in 0..count {
                    if graph.get_node_attr(node, &"answer".into())? != Some(expected.clone()) {
                        return Err("invalid result".into());
                    }
                }
                samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id()}));
            }
            cases.push(json!({"mode":mode,"nodes":count,"edges":count,"samples":samples}));
        }
    }
    let path = std::env::var("GROGGY_TRANSFORMATION_PROFILE")
        .unwrap_or_else(|_| "transformation_builder_steps.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native release directed ring workloads; fixture/spec construction excluded. Includes transformation, property attachment and one checked publication. Five samples per workload; all outputs checked. No matched speedup, FFI, RSS/allocation or production budget claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote transformation receipt to {path}");
    Ok(())
}
