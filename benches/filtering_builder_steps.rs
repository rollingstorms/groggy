//! Bounded release receipt for staged filtering and deterministic rank outputs.
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
            "core.sort_nodes_by_attr",
            "core.top_k",
            "core.filter_nodes_by_attr",
            "core.filter_edges_by_attr",
        ] {
            let is_edge = mode == "core.filter_edges_by_attr";
            let operation = if mode == "core.top_k" {
                json!({"id":mode,"params":{"source":"scores","k":count/2,"order":"asc","target":"values"}})
            } else if mode == "core.sort_nodes_by_attr" {
                json!({"id":mode,"params":{"attr":"score","order":"asc","target":"values"}})
            } else {
                json!({"id":mode,"params":{"attr":if is_edge {"weight"} else {"score"},"target":"values","predicate":{"type":"ge","value":{"type":"Int","value":2}}}})
            };
            let steps = json!([
                {"id":"core.init_nodes","params":{"target":"scores","value":2}},
                {"id":"core.attach_node_attr","params":{"source":"scores","attr":"score"}},
                {"id":"core.init_edges","params":{"target":"weights","value":2}},
                {"id":"core.attach_edge_attr","params":{"source":"weights","attr":"weight"}},
                operation,
                {"id":if is_edge {"core.attach_edge_attr"} else {"core.attach_node_attr"},"params":{"source":"values","attr":"answer"}}
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
                let mut graph = Graph::new_directed_manual();
                graph.add_nodes(count).unwrap();
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
                    let expected = if mode == "core.top_k" && node >= count / 2 {
                        None
                    } else if mode == "core.sort_nodes_by_attr" || mode == "core.top_k" {
                        Some(AttrValue::Int(node as i64))
                    } else {
                        Some(AttrValue::Int(1))
                    };
                    let actual = if is_edge {
                        graph.get_edge_attr(node, &"answer".into())?
                    } else {
                        graph.get_node_attr(node, &"answer".into())?
                    };
                    if actual != expected {
                        return Err("invalid filtering/rank result".into());
                    }
                }
                samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id()}));
            }
            cases.push(json!({"mode":mode,"nodes":count,"edges":count,"samples":samples}));
        }
    }
    let path = std::env::var("GROGGY_FILTERING_BUILDER_PROFILE")
        .unwrap_or_else(|_| "filtering_builder_steps.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release ring workloads; fixture and algorithm construction excluded; includes staged node/edge assignments, filtering or ranking, result staging and one publication. Five samples per case. No before/after speedup, FFI, RSS/allocation or production budget claim.", "cases":cases
        }))?,
    )?;
    println!("Wrote filtering builder receipt to {path}");
    Ok(())
}
