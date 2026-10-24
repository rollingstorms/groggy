//! Bounded release receipt for staged fused/message-pass loops on a ring.
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
        for mode in ["fused", "message_pass"] {
            let operation = if mode == "fused" {
                json!({"id":"graph.fused_neighbor_mul_agg","params":{"values":"values","scalars":"ones","target":"values","direction":"in"}})
            } else {
                json!({"id":"core.execution_block","params":{
                    "mode":"message_pass","target":"values","options":{"include_self":false,"direction":"in"},
                    "body":{"nodes":[{"id":"sum","domain":"core","op_type":"neighbor_agg","inputs":["values"],"output":"sum","metadata":{"agg":"sum"}}]}
                }})
            };
            let steps = json!([
                {"id":"core.init_nodes","params":{"target":"values","value":1.0}},
                {"id":"core.init_nodes","params":{"target":"ones","value":1.0}},
                {"id":"iter.loop","params":{"iterations":5,"body":[
                    operation,
                    {"id":"core.attach_node_attr","params":{"source":"values","attr":"answer"}},
                    {"id":"core.load_node_attr","params":{"target":"values","attr":"answer"}}
                ]}}
            ]);
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
                let mut graph = Graph::new();
                graph.add_nodes(count);
                for node in 0..count {
                    graph.add_edge(node, (node + 1) % count)?;
                }
                graph.commit("fixture".into(), "bench".into())?;
                let before = graph.current_state_id();
                let shared = Rc::new(RefCell::new(graph));
                let selection =
                    Subgraph::from_nodes(shared.clone(), (0..count).collect(), "bench".into())?;
                let mut context = Context::new();
                let started = Instant::now();
                algorithm.execute(&mut context, selection)?;
                let seconds = started.elapsed().as_secs_f64();
                let graph = shared.borrow();
                if graph.current_state_id() != before + 1 || graph.has_uncommitted_changes() {
                    return Err("invalid atomic publication".into());
                }
                for node in 0..count {
                    if graph.get_node_attr(node, &"answer".into())? != Some(AttrValue::Float64(1.0))
                    {
                        return Err("invalid ring result".into());
                    }
                }
                samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id()}));
            }
            cases.push(
                json!({"mode":mode,"nodes":count,"edges":count,"iterations":5,"samples":samples}),
            );
        }
    }
    let path = std::env::var("GROGGY_ATOMIC_BLOCK_PROFILE")
        .unwrap_or_else(|_| "atomic_builder_blocks.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release ring workloads; fixture and algorithm construction excluded; includes five ordered iterations, topology preparation, property staging and one publication. Five samples per case. No before/after speedup, FFI, RSS/allocation or production budget claim.", "cases":cases
        }))?,
    )?;
    println!("Wrote atomic block receipt to {path}");
    Ok(())
}
