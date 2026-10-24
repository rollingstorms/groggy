//! Bounded native release receipt for community helpers plus checked publication.
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
            "singleton",
            "degree",
            "random",
            "labels_undirected",
            "labels_directed",
            "modularity_undirected",
            "modularity_directed",
        ] {
            let directed = mode.ends_with("_directed");
            let propagation = mode.starts_with("labels_");
            let modularity = mode.starts_with("modularity_");
            let mut steps = Vec::new();
            if propagation {
                steps.push(json!({"id":"core.init_nodes","params":{"target":"labels","value":5}}));
                steps.push(json!({"id":"core.label_propagate_step","params":{"labels":"labels","target":"result"}}));
            } else {
                steps.push(json!({"id":"core.community_seed","params":{"strategy":if modularity {"singleton"} else {mode},"target":"labels","k":7,"seed":19}}));
                if modularity {
                    steps.push(json!({"id":"core.modularity_gain","params":{"partition":"labels","target":"result"}}));
                }
            }
            steps.push(json!({"id":"core.attach_node_attr","params":{"source":if modularity || propagation {"result"} else {"labels"},"attr":"answer"}}));
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
                let mut graph = if directed {
                    Graph::new_directed()
                } else {
                    Graph::new_undirected()
                };
                graph.add_nodes(count);
                for node in 0..count {
                    graph.add_edge(node, (node + 1) % count)?;
                }
                graph.commit("fixture".into(), "bench".into())?;
                let before = graph.current_state_id();
                let shared = Rc::new(RefCell::new(graph));
                let selected =
                    Subgraph::from_nodes(shared.clone(), (0..count).collect(), "community".into())?;
                let start = Instant::now();
                algorithm.execute(&mut Context::new(), selected)?;
                let seconds = start.elapsed().as_secs_f64();
                let graph = shared.borrow();
                if graph.current_state_id() != before + 1 || graph.has_uncommitted_changes() {
                    return Err("invalid publication".into());
                }
                for node in 0..count {
                    let value = graph.get_node_attr(node, &"answer".into())?;
                    if modularity {
                        let expected = 1.0 / count as f64 - 2.0 / (count as f64).powi(2);
                        if !matches!(value, Some(AttrValue::Float64(value)) if (value-expected).abs()<1e-12)
                        {
                            return Err(format!("invalid gain {value:?}").into());
                        }
                    } else {
                        let integer = match value {
                            Some(AttrValue::Int(value)) => value,
                            Some(AttrValue::SmallInt(value)) => i64::from(value),
                            _ => return Err("invalid label type".into()),
                        };
                        let valid = if propagation {
                            integer == 5
                        } else if mode == "random" {
                            (0..7).contains(&integer)
                        } else if mode == "degree" {
                            let hubs = count.div_ceil(10);
                            integer
                                == if node < hubs {
                                    node as i64
                                } else if node == count - 1 {
                                    0
                                } else {
                                    (hubs - 1) as i64
                                }
                        } else {
                            integer == node as i64
                        };
                        if !valid {
                            return Err("invalid label".into());
                        }
                    }
                }
                samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id()}));
            }
            cases.push(json!({"mode":mode,"nodes":count,"edges":count,"samples":samples}));
        }
    }
    let path = std::env::var("GROGGY_COMMUNITY_PROFILE")
        .unwrap_or_else(|_| "community_builder_steps.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native release rings: seeding, synchronous propagation and directed/undirected local modularity gains, attachment and one checked publication. Fixture/spec construction excluded; five observations per workload. Every result/publication checked. No matched before/after speedup, FFI/RSS/allocation or production-scale claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote community receipt to {path}");
    Ok(())
}
