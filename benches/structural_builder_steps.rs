//! Bounded release receipt for checked structural builder steps on a ring.
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
            "core.node_degree",
            "core.weighted_degree",
            "core.edge_weight_sum",
            "core.triangle_count",
            "core.k_core_mark",
        ] {
            let operation = if mode == "core.node_degree" || mode == "core.triangle_count" {
                json!({"id":mode,"params":{"target":"values"}})
            } else if mode == "core.k_core_mark" {
                json!({"id":mode,"params":{"target":"values","k":2}})
            } else {
                json!({"id":mode,"params":{"target":"values","weight_attr":"weight"}})
            };
            let steps = json!([
                {"id":"core.init_edges","params":{"target":"weights","value":2.0}},
                {"id":"core.attach_edge_attr","params":{"source":"weights","attr":"weight"}},
                operation,
                {"id":"core.attach_node_attr","params":{"source":"values","attr":"answer"}}
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
                let mut graph = if mode == "core.triangle_count" || mode == "core.k_core_mark" {
                    Graph::new_undirected()
                } else {
                    Graph::new_directed()
                };
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
                    let expected = if mode == "core.triangle_count" {
                        AttrValue::Int(0)
                    } else if mode == "core.node_degree" || mode == "core.k_core_mark" {
                        AttrValue::Int(1)
                    } else {
                        AttrValue::Float64(4.0)
                    };
                    if graph.get_node_attr(node, &"answer".into())? != Some(expected) {
                        return Err("invalid ring result".into());
                    }
                }
                samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id()}));
            }
            cases.push(json!({"mode":mode,"nodes":count,"edges":count,"samples":samples}));
        }
    }
    let path = std::env::var("GROGGY_STRUCTURAL_BUILDER_PROFILE")
        .unwrap_or_else(|_| "structural_builder_steps.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release ring workloads; fixture and algorithm construction excluded; includes one structural calculation, staged edge weight assignment, node result staging and one publication. Five samples per case. No before/after speedup, FFI, RSS/allocation or production budget claim.", "cases":cases
        }))?,
    )?;
    println!("Wrote structural builder receipt to {path}");
    Ok(())
}
